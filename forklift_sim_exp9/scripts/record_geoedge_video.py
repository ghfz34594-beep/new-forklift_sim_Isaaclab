"""Record global-view videos for a GeoEdge checkpoint.

The policy still consumes the 21D geometry observation. Cameras are enabled only
so Gym's RecordVideo wrapper can capture rendered frames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Record videos for GeoEdge checkpoints")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--video_folder", type=str, default=None)
parser.add_argument("--seed", type=int, default=20260427)
parser.add_argument(
    "--target_env_id",
    type=int,
    default=None,
    help="Focus the global render camera on this vectorized environment id.",
)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
parser.add_argument("--camera_eye", type=float, nargs=3, default=(-4.0, -6.0, 4.0))
parser.add_argument("--camera_lookat", type=float, nargs=3, default=(-1.5, 0.0, 0.2))
parser.add_argument(
    "--fixed_stage1_init",
    type=float,
    nargs=3,
    metavar=("X_M", "Y_M", "YAW_DEG"),
    default=None,
    help="Use one fixed Stage A reset pose instead of sampling x/y/yaw.",
)
parser.add_argument(
    "--actor_only_load",
    "--actor-only-load",
    action="store_true",
    help="Load only actor-compatible checkpoint tensors for legacy checkpoints with critic shape mismatch.",
)
parser.add_argument(
    "--stage1_eval",
    "--stage1-eval",
    action="store_true",
    help="Record with insert-only Stage A settings: lift-free success and Stage A reset distribution.",
)
parser.add_argument(
    "--reset_profile",
    choices=("default", "near", "mid", "full"),
    default="default",
    help="Optional Stage A reset profile override used only with --stage1_eval.",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


STAGE1_RESET_PROFILES = {
    "near": {
        "stage1_init_x_min_m": -3.2,
        "stage1_init_x_max_m": -2.4,
        "stage1_init_y_min_m": -0.25,
        "stage1_init_y_max_m": 0.25,
        "stage1_init_yaw_deg_min": -8.0,
        "stage1_init_yaw_deg_max": 8.0,
    },
    "mid": {
        "stage1_init_x_min_m": -3.6,
        "stage1_init_x_max_m": -2.6,
        "stage1_init_y_min_m": -0.4,
        "stage1_init_y_max_m": 0.4,
        "stage1_init_yaw_deg_min": -10.0,
        "stage1_init_yaw_deg_max": 10.0,
    },
    "full": {
        "stage1_init_x_min_m": -4.0,
        "stage1_init_x_max_m": -3.0,
        "stage1_init_y_min_m": -0.6,
        "stage1_init_y_max_m": 0.6,
        "stage1_init_yaw_deg_min": -14.32394487827058,
        "stage1_init_yaw_deg_max": 14.32394487827058,
    },
}


def _apply_stage1_reset_profile(env_cfg, profile: str) -> None:
    if profile == "default":
        return
    for key, value in STAGE1_RESET_PROFILES[profile].items():
        setattr(env_cfg, key, value)


def _as_bool_tensor(dones, device: str) -> torch.Tensor:
    if isinstance(dones, torch.Tensor):
        return dones.bool()
    return torch.tensor(dones, dtype=torch.bool, device=device)


def _load_checkpoint(runner: OnPolicyRunner, checkpoint: str, actor_only: bool, device: str) -> None:
    if not actor_only:
        runner.load(checkpoint)
        return

    loaded = torch.load(checkpoint, weights_only=False, map_location=device)
    source_state = loaded["model_state_dict"]
    target_state = runner.alg.policy.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        if key.startswith("critic.") or key.startswith("critic_obs_normalizer."):
            skipped.append(key)
            continue
        target_value = target_state.get(key)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value

    runner.alg.policy.load_state_dict(compatible_state, strict=False)
    print(
        "[INFO] actor-only checkpoint load: "
        f"loaded={len(compatible_state)} skipped={len(skipped)} checkpoint={checkpoint}"
    )


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env_cfg.seed = args_cli.seed
env_cfg.use_camera = False
env_cfg.use_asymmetric_critic = False
env_cfg.enable_geo_edge_obs = True
env_cfg.stage_1_mode = bool(args_cli.stage1_eval)
env_cfg.stage1_success_without_lift = bool(args_cli.stage1_eval)
env_cfg.hold_gate_curriculum_enable = False
env_cfg.tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
env_cfg.strict_tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
env_cfg.viewer.resolution = (int(args_cli.video_width), int(args_cli.video_height))
if args_cli.stage1_eval:
    _apply_stage1_reset_profile(env_cfg, args_cli.reset_profile)
if args_cli.fixed_stage1_init is not None:
    fixed_x, fixed_y, fixed_yaw_deg = (float(v) for v in args_cli.fixed_stage1_init)
    env_cfg.stage1_init_x_min_m = fixed_x
    env_cfg.stage1_init_x_max_m = fixed_x
    env_cfg.stage1_init_y_min_m = fixed_y
    env_cfg.stage1_init_y_max_m = fixed_y
    env_cfg.stage1_init_yaw_deg_min = fixed_yaw_deg
    env_cfg.stage1_init_yaw_deg_max = fixed_yaw_deg
if args_cli.device is not None:
    agent_cfg.device = args_cli.device

video_folder = args_cli.video_folder
if video_folder is None:
    video_folder = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "geoedge_strict_best")
Path(video_folder).mkdir(parents=True, exist_ok=True)

env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
if args_cli.target_env_id is not None:
    raw_focus_env = env.unwrapped
    if args_cli.target_env_id < 0 or args_cli.target_env_id >= raw_focus_env.num_envs:
        raise ValueError(f"target_env_id={args_cli.target_env_id} outside [0, {raw_focus_env.num_envs})")
    origin = raw_focus_env.scene.env_origins[args_cli.target_env_id].detach().cpu().numpy()
    eye = origin + np.asarray(args_cli.camera_eye, dtype=np.float32)
    lookat = origin + np.asarray(args_cli.camera_lookat, dtype=np.float32)
    raw_focus_env.sim.set_camera_view(eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in lookat))
    print(
        "[INFO] Focused global render camera: "
        f"target_env_id={args_cli.target_env_id} origin={origin.tolist()} "
        f"eye={eye.tolist()} lookat={lookat.tolist()}"
    )
env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=args_cli.video_length,
    disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
raw_env = env.unwrapped

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
_load_checkpoint(runner, args_cli.checkpoint, args_cli.actor_only_load, raw_env.device)
policy = runner.get_inference_policy(device=raw_env.device)

obs = env.get_observations()
recorded = 0
steps = 0
max_steps = args_cli.episodes * (args_cli.video_length + int(raw_env.max_episode_length) + 5)
while recorded < args_cli.episodes and steps < max_steps:
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, dones, _ = env.step(actions.detach().clone())
    steps += 1
    if bool(_as_bool_tensor(dones, raw_env.device).any().item()):
        recorded += 1

if recorded < args_cli.episodes:
    print(
        f"[WARN] Requested {args_cli.episodes} completed episodes, observed {recorded}. "
        "The video wrapper still recorded from step 0 up to video_length."
    )

env.close()
simulation_app.close()
print(f"[INFO] Wrote GeoEdge videos to: {video_folder}")
