# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--bc_checkpoint",
    type=str,
    default=None,
    help="Warm-start compatible actor/vision weights from a behavior-cloning checkpoint.",
)
parser.add_argument(
    "--warm_start_checkpoint",
    type=str,
    default=None,
    help=(
        "Warm-start all compatible policy tensors from an RL checkpoint without "
        "loading the optimizer or runner iteration state."
    ),
)
parser.add_argument(
    "--allow_multi_env_vision",
    action="store_true",
    default=False,
    help=(
        "Allow visual dual-camera tasks to train with num_envs > 1. "
        "Use only after a camera isolation check passes; otherwise RGB observations can contain other envs."
    ),
)
parser.add_argument(
    "--vision_acceptance_summary",
    type=str,
    default=None,
    help="Path to a passing Room60 visual_isolation_summary.json required for multi-env Toyota RGB training.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def install_policy_iteration_hook(runner) -> None:
    """Bridge outer PPO iteration count into custom policy modules.

    Some task-specific policies need to react to the *outer* policy update
    schedule, while the default library only exposes finer-grained callbacks.
    If the policy implements ``on_policy_iteration_end(iteration)``, call it
    once after every ``alg.update()``.
    """

    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    if policy is None or not hasattr(policy, "on_policy_iteration_end"):
        return

    original_update = runner.alg.update
    policy_iteration = int(runner.current_learning_iteration)

    def wrapped_update(*args, **kwargs):
        nonlocal policy_iteration
        result = original_update(*args, **kwargs)
        policy_iteration += 1
        policy.on_policy_iteration_end(policy_iteration)
        return result

    runner.alg.update = wrapped_update
    print("[INFO]: Installed policy iteration hook for custom actor-critic schedule.")


def _set_policy_action_std(runner, action_std: float) -> bool:
    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    if policy is None:
        return False

    value = max(float(action_std), 1e-6)
    with torch.no_grad():
        if hasattr(policy, "log_std"):
            policy.log_std.fill_(torch.log(torch.tensor(value, device=policy.log_std.device)))
            return True
        if hasattr(policy, "std"):
            policy.std.fill_(value)
            return True
    return False


def install_ppo_late_phase_schedule(runner) -> None:
    """Apply an optional low-drift PPO schedule after an early learning window."""

    cfg = getattr(runner, "cfg", {})
    schedule = cfg.get("late_phase_schedule") if isinstance(cfg, dict) else None
    if not schedule or not bool(schedule.get("enable", False)):
        return

    start_iter = int(schedule.get("start_iter", 0))
    ramp_iters = max(int(schedule.get("ramp_iters", 1)), 1)
    final_lr = float(schedule.get("learning_rate", runner.alg.learning_rate))
    final_entropy = float(schedule.get("entropy_coef", runner.alg.entropy_coef))
    final_desired_kl = float(schedule.get("desired_kl", runner.alg.desired_kl or 0.0))
    final_clip = float(schedule.get("clip_param", runner.alg.clip_param))
    final_max_grad_norm = float(schedule.get("max_grad_norm", runner.alg.max_grad_norm))
    final_epochs = int(schedule.get("num_learning_epochs", runner.alg.num_learning_epochs))
    final_batches = int(schedule.get("num_mini_batches", runner.alg.num_mini_batches))
    final_std = schedule.get("action_std", None)
    final_schedule = schedule.get("schedule", None)
    freeze_normalization = bool(schedule.get("freeze_normalization", False))
    freeze_actor = bool(schedule.get("freeze_actor", False))
    freeze_actor_start_iter = int(schedule.get("freeze_actor_start_iter", start_iter))

    base = {
        "learning_rate": float(runner.alg.learning_rate),
        "entropy_coef": float(runner.alg.entropy_coef),
        "desired_kl": float(runner.alg.desired_kl or 0.0),
        "clip_param": float(runner.alg.clip_param),
        "max_grad_norm": float(runner.alg.max_grad_norm),
        "num_learning_epochs": int(runner.alg.num_learning_epochs),
        "num_mini_batches": int(runner.alg.num_mini_batches),
        "schedule": getattr(runner.alg, "schedule", None),
    }

    original_update = runner.alg.update
    original_act = runner.alg.act
    original_process_env_step = runner.alg.process_env_step
    policy_iteration = int(runner.current_learning_iteration)
    std_applied = False
    norm_frozen = False

    def freeze_policy_normalization() -> bool:
        policy = getattr(runner.alg, "actor_critic", None)
        if policy is None:
            policy = getattr(runner.alg, "policy", None)
        if policy is None:
            return False
        frozen_any = False
        for attr in ("actor_obs_normalizer", "critic_obs_normalizer"):
            module = getattr(policy, attr, None)
            if module is not None and hasattr(module, "eval"):
                module.eval()
                frozen_any = True
        return frozen_any

    def lerp(key: str, target: float, alpha: float) -> float:
        return base[key] + (target - base[key]) * alpha

    def apply_schedule_if_due() -> float:
        nonlocal policy_iteration, std_applied, norm_frozen
        policy_iteration = max(policy_iteration, int(getattr(runner, "current_learning_iteration", 0)))
        alpha = 0.0
        if policy_iteration >= start_iter:
            alpha = min(1.0, float(policy_iteration - start_iter + 1) / float(ramp_iters))
        if alpha > 0.0:
            runner.alg.learning_rate = lerp("learning_rate", final_lr, alpha)
            runner.alg.entropy_coef = lerp("entropy_coef", final_entropy, alpha)
            runner.alg.desired_kl = lerp("desired_kl", final_desired_kl, alpha)
            runner.alg.clip_param = lerp("clip_param", final_clip, alpha)
            runner.alg.max_grad_norm = lerp("max_grad_norm", final_max_grad_norm, alpha)
            runner.alg.num_learning_epochs = final_epochs
            runner.alg.num_mini_batches = final_batches
            if final_schedule is not None and hasattr(runner.alg, "schedule"):
                runner.alg.schedule = final_schedule
            if freeze_normalization and not norm_frozen:
                norm_frozen = freeze_policy_normalization()
                print(
                    "[INFO]: PPO late-phase normalization freeze "
                    f"{'enabled' if norm_frozen else 'requested but no compatible normalizer found'}"
                )
            if final_std is not None and not std_applied:
                std_applied = _set_policy_action_std(runner, float(final_std))
                print(
                    "[INFO]: PPO late-phase action std "
                    f"set to {float(final_std):.6f}: {'ok' if std_applied else 'no compatible policy field'}"
                )
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = runner.alg.learning_rate
        return alpha

    def wrapped_act(*args, **kwargs):
        apply_schedule_if_due()
        return original_act(*args, **kwargs)

    def wrapped_process_env_step(*args, **kwargs):
        apply_schedule_if_due()
        return original_process_env_step(*args, **kwargs)

    def wrapped_update(*args, **kwargs):
        nonlocal policy_iteration
        apply_schedule_if_due()
        if freeze_actor and policy_iteration >= freeze_actor_start_iter:
            runner.alg.storage.clear()
            policy_iteration += 1
            return {"value_function": 0.0, "surrogate": 0.0, "entropy": 0.0}
        result = original_update(*args, **kwargs)
        apply_schedule_if_due()
        policy_iteration += 1
        return result

    runner.alg.act = wrapped_act
    runner.alg.process_env_step = wrapped_process_env_step
    runner.alg.update = wrapped_update
    print(
        "[INFO]: Installed PPO late-phase schedule: "
        f"start_iter={start_iter}, ramp_iters={ramp_iters}, lr->{final_lr}, "
        f"entropy->{final_entropy}, desired_kl->{final_desired_kl}, clip->{final_clip}, "
        f"schedule->{final_schedule or base['schedule']}, freeze_normalization={freeze_normalization}, "
        f"freeze_actor={freeze_actor}, freeze_actor_start_iter={freeze_actor_start_iter}"
    )


def reset_policy_action_std(runner, init_noise_std: float) -> bool:
    """Reset actor action exploration std after loading a checkpoint."""

    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    if policy is None:
        return False

    with torch.no_grad():
        if hasattr(policy, "log_std"):
            policy.log_std.fill_(torch.log(torch.tensor(init_noise_std, device=policy.log_std.device)))
            return True
        if hasattr(policy, "std"):
            policy.std.fill_(init_noise_std)
            return True
    return False


def load_bc_warm_start(runner, checkpoint: str, device: str) -> None:
    """Load actor-compatible tensors from a BC checkpoint into the PPO policy."""

    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    if policy is None:
        raise RuntimeError("Cannot load BC checkpoint: runner has no policy module")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source_state = payload.get("model_state_dict", payload)
    target_state = policy.state_dict()
    compatible_state = {}
    skipped = []
    skip_prefixes = ("critic.", "critic_obs_normalizer.", "std", "log_std")
    for key, value in source_state.items():
        if key.startswith(skip_prefixes):
            skipped.append(key)
            continue
        target_value = target_state.get(key)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value
    policy.load_state_dict(compatible_state, strict=False)
    print(
        "[INFO]: BC warm-start loaded "
        f"{len(compatible_state)} tensors, skipped {len(skipped)} tensors from {checkpoint}"
    )


def load_full_model_warm_start(runner, checkpoint: str, device: str) -> None:
    """Load compatible policy tensors from an RL checkpoint without optimizer state."""

    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    if policy is None:
        raise RuntimeError("Cannot warm-start checkpoint: runner has no policy module")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source_state = payload.get("model_state_dict", payload)
    target_state = policy.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value
    policy.load_state_dict(compatible_state, strict=False)
    print(
        "[INFO]: Full model warm-start loaded "
        f"{len(compatible_state)} tensors, skipped {len(skipped)} tensors from {checkpoint}"
    )


def _camera_far_from_cfg(env_cfg) -> float | None:
    if hasattr(env_cfg, "dual_camera_far_clip_m"):
        return float(env_cfg.dual_camera_far_clip_m)
    camera_cfg = getattr(env_cfg, "tiled_camera_left", None)
    if camera_cfg is None:
        return None
    try:
        return float(camera_cfg.spawn.clipping_range[1])
    except Exception:
        return None


def _assert_visual_acceptance_matches(env_cfg, task_name: str | None, num_envs: int, summary_path: str) -> None:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("pass") is not True:
        raise RuntimeError(f"Visual acceptance summary did not pass: {summary_path}")
    if summary.get("foreign_leakage_pass") is not True:
        raise RuntimeError(f"Visual acceptance summary did not pass the foreign-leakage gate: {summary_path}")
    if summary.get("camera_learnability_pass") is not True:
        raise RuntimeError(f"Visual acceptance summary did not pass the camera-learnability gate: {summary_path}")

    mismatches = []
    if str(summary.get("task", "")) != str(task_name or ""):
        mismatches.append(f"task summary={summary.get('task')} requested={task_name}")
    if int(summary.get("num_envs", -1)) != int(num_envs):
        mismatches.append(f"num_envs summary={summary.get('num_envs')} requested={num_envs}")
    if abs(float(summary.get("env_spacing", -999.0)) - float(env_cfg.scene.env_spacing)) > 1e-6:
        mismatches.append(
            f"env_spacing summary={summary.get('env_spacing')} requested={float(env_cfg.scene.env_spacing)}"
        )
    if bool(summary.get("vision_room_enable", False)) != bool(getattr(env_cfg, "vision_room_enable", False)):
        mismatches.append(
            "vision_room_enable summary="
            f"{summary.get('vision_room_enable')} requested={bool(getattr(env_cfg, 'vision_room_enable', False))}"
        )

    summary_dual = summary.get("dual_camera_config") or {}
    if abs(float(summary_dual.get("hfov_deg", -999.0)) - float(getattr(env_cfg, "dual_camera_hfov_deg", -1))) > 1e-6:
        mismatches.append(
            f"hfov summary={summary_dual.get('hfov_deg')} requested={float(getattr(env_cfg, 'dual_camera_hfov_deg', -1))}"
        )
    summary_far = summary.get("camera_far", None)
    cfg_far = _camera_far_from_cfg(env_cfg)
    if summary_far is not None and cfg_far is not None and abs(float(summary_far) - float(cfg_far)) > 1e-6:
        mismatches.append(f"camera_far summary={summary_far} requested={cfg_far}")

    if mismatches:
        raise RuntimeError(
            "Visual acceptance summary does not match this multi-env RGB training config: "
            + "; ".join(mismatches)
        )

    env_id_coverage = summary.get("env_id_coverage") or {}
    mosaic_coverage = summary.get("mosaic_env_coverage") or {}
    if int(num_envs) > 1:
        if not env_id_coverage:
            raise RuntimeError(
                "Visual acceptance summary is from the old fixed-sample schema and does not prove scalable "
                f"multi-env isolation for num_envs={num_envs}: {summary_path}"
            )
        checked_count = int(env_id_coverage.get("checked_count", 0))
        coverage_fraction = float(env_id_coverage.get("coverage_fraction", 0.0))
        if checked_count < 2 or coverage_fraction <= 0.0 or env_id_coverage.get("pass") is not True:
            raise RuntimeError(
                "Visual acceptance target-env coverage is insufficient: "
                f"checked={checked_count}, fraction={coverage_fraction:.3f}, summary={summary_path}"
            )
        if bool(mosaic_coverage.get("require_full_mosaic_coverage", False)) and mosaic_coverage.get("pass") is not True:
            raise RuntimeError(f"Visual acceptance requested full mosaic coverage but did not pass: {summary_path}")

    print(
        "[INFO]: Multi-env visual isolation accepted by "
        f"{summary_path} (checked_envs={int((summary.get('env_id_coverage') or {}).get('checked_count', 0))}, "
        f"coverage={float((summary.get('env_id_coverage') or {}).get('coverage_fraction', 0.0)):.3f})"
    )


def guard_multi_env_visual_training(env_cfg, task_name: str | None, num_envs: int) -> None:
    """Fail closed for known-contaminated multi-env RGB training setups."""

    if int(num_envs) <= 1:
        return
    use_camera = bool(getattr(env_cfg, "use_camera", False))
    use_dual_cameras = bool(getattr(env_cfg, "use_dual_cameras", False))
    task_name = task_name or ""
    toyota_visual_task = "ToyotaDualCamera" in task_name or (use_camera and use_dual_cameras)
    if not toyota_visual_task:
        return
    if args_cli.vision_acceptance_summary:
        _assert_visual_acceptance_matches(env_cfg, task_name, num_envs, args_cli.vision_acceptance_summary)
        return

    if args_cli.allow_multi_env_vision or os.environ.get("ALLOW_MULTI_ENV_VISION", "") == "1":
        print(
            "[WARN]: Multi-env visual training explicitly allowed. "
            "Make sure camera isolation was verified for this exact spacing/clipping configuration."
        )
        return

    raise RuntimeError(
        "Refusing to train a visual dual-camera task with num_envs > 1 because multi-env RGB contamination "
        "has been observed in this project. Run validate_room60_visual_isolation.py for this exact config and "
        "pass --vision_acceptance_summary /path/to/visual_isolation_summary.json."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    guard_multi_env_visual_training(env_cfg, args_cli.task, int(env_cfg.scene.num_envs))

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if os.path.isabs(agent_cfg.load_checkpoint):
            resume_path = agent_cfg.load_checkpoint
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    install_policy_iteration_hook(runner)
    install_ppo_late_phase_schedule(runner)
    if args_cli.bc_checkpoint:
        load_bc_warm_start(runner, args_cli.bc_checkpoint, agent_cfg.device)
    if args_cli.warm_start_checkpoint:
        load_full_model_warm_start(runner, args_cli.warm_start_checkpoint, agent_cfg.device)
        reset_std_value = os.environ.get("RSL_RL_RESET_ACTION_STD_AFTER_LOAD", "")
        if reset_std_value:
            reset_ok = reset_policy_action_std(runner, float(reset_std_value))
            print(
                "[INFO]: Reset action std after warm-start "
                f"to {float(reset_std_value):.6f}: {'ok' if reset_ok else 'no compatible policy field'}"
            )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        reset_std_value = os.environ.get("RSL_RL_RESET_ACTION_STD_AFTER_LOAD", "")
        if reset_std_value:
            reset_ok = reset_policy_action_std(runner, float(reset_std_value))
            print(
                "[INFO]: Reset action std after checkpoint load "
                f"to {float(reset_std_value):.6f}: {'ok' if reset_ok else 'no compatible policy field'}"
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
