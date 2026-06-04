"""Deterministic strict eval for GeometryEdgeObs forklift checkpoints.

Outputs a summary JSON plus a per-episode CSV. The policy observes the 21D
GeoEdge tensor; rendering/cameras are not used for policy input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Deterministic strict eval for GeoEdge checkpoints")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollouts", type=int, default=4)
parser.add_argument("--seed", type=int, default=20260427)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument(
    "--actor_only_load",
    "--actor-only-load",
    action="store_true",
    help=(
        "Load only actor-compatible checkpoint tensors. Use this for legacy checkpoints "
        "whose critic observation shape differs from the current GeoEdge config."
    ),
)
parser.add_argument(
    "--stage1_eval",
    "--stage1-eval",
    action="store_true",
    help="Evaluate with Stage A insert-only success logic: stage_1_mode=true and lift not required.",
)
parser.add_argument(
    "--reset_profile",
    "--reset-profile",
    type=str,
    choices=("default", "near", "mid", "full"),
    default="default",
    help="Optional Stage A reset profile override used only when --stage1_eval is set.",
)
parser.add_argument(
    "--fixed_stage1_init",
    "--fixed-stage1-init",
    type=float,
    nargs=3,
    metavar=("X_M", "Y_M", "YAW_DEG"),
    default=None,
    help="Use one fixed Stage A reset pose instead of sampling x/y/yaw.",
)
parser.add_argument(
    "--preserve_stage1_near_hard_curriculum",
    "--preserve-stage1-near-hard-curriculum",
    action="store_true",
    help=(
        "Keep the task's near-hard reset curriculum during eval. By default eval disables "
        "training-only reset curricula so bucket metrics reflect the requested reset profile."
    ),
)
parser.add_argument(
    "--hybrid_rescue_enable",
    "--hybrid-rescue-enable",
    action="store_true",
    help=(
        "Enable a privileged near-field recovery override around the loaded policy. "
        "This is off by default and is recorded in the output summary/CSV."
    ),
)
parser.add_argument("--rescue_trigger_dist_m", "--rescue-trigger-dist-m", type=float, default=0.70)
parser.add_argument("--rescue_release_dist_m", "--rescue-release-dist-m", type=float, default=0.92)
parser.add_argument("--rescue_center_m", "--rescue-center-m", type=float, default=0.18)
parser.add_argument("--rescue_tip_m", "--rescue-tip-m", type=float, default=0.12)
parser.add_argument("--rescue_yaw_deg", "--rescue-yaw-deg", type=float, default=6.0)
parser.add_argument("--rescue_insert_frac_max", "--rescue-insert-frac-max", type=float, default=0.25)
parser.add_argument("--rescue_reverse_drive", "--rescue-reverse-drive", type=float, default=0.45)
parser.add_argument("--rescue_steer", "--rescue-steer", type=float, default=0.65)
parser.add_argument("--rescue_max_steps", "--rescue-max-steps", type=int, default=64)
parser.add_argument(
    "--rescue_once_per_episode",
    "--rescue-once-per-episode",
    action="store_true",
    help="Do not re-enter hybrid rescue after the first recovery window in an episode.",
)
parser.add_argument(
    "--rescue_reverse_steer_flip",
    "--rescue-reverse-steer-flip",
    action="store_true",
    help="Flip the steer sign when the rescue command drives backward.",
)
parser.add_argument(
    "--rescue_opposite_yaw_only",
    "--rescue-opposite-yaw-only",
    action="store_true",
    help="Only enter rescue when lateral offset and yaw error have opposite signs.",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import (
    HoldLogicConfig,
    compute_hold_logic,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


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


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _compute_step_state(raw_env) -> dict[str, torch.Tensor]:
    pallet_pos = raw_env.pallet.data.root_pos_w
    tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()

    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
    yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)

    rel_fc = fork_center[:, :2] - pallet_pos[:, :2]
    center_y_signed = torch.sum(rel_fc * v_lat, dim=-1)
    tip_y_signed = torch.sum(rel_tip * v_lat, dim=-1)
    center_y_err = torch.abs(center_y_signed)
    tip_y_err = torch.abs(tip_y_signed)
    lift_height = tip[:, 2] - raw_env._fork_tip_z0
    pallet_lift_height = pallet_pos[:, 2] - raw_env.cfg.pallet_cfg.init_state.pos[2]
    z_err = torch.abs(lift_height - pallet_lift_height)
    valid_insert_z = z_err < raw_env.cfg.max_insert_z_err

    strict_tip_gate_m = float(getattr(raw_env.cfg, "strict_tip_align_entry_m", 0.12))
    strict_cfg = HoldLogicConfig(
        insert_thresh=raw_env._insert_thresh,
        max_lateral_err_m=raw_env.cfg.max_lateral_err_m,
        max_yaw_err_deg=raw_env.cfg.max_yaw_err_deg,
        hysteresis_ratio=raw_env.cfg.hysteresis_ratio,
        insert_exit_epsilon=raw_env.cfg.insert_exit_epsilon,
        lift_delta_m=raw_env.cfg.lift_delta_m,
        lift_exit_epsilon=raw_env.cfg.lift_exit_epsilon,
        hold_counter_decay=raw_env.cfg.hold_counter_decay,
        tip_align_entry_m=strict_tip_gate_m,
        tip_align_exit_m=max(float(raw_env.cfg.tip_align_exit_m), strict_tip_gate_m),
        tip_align_near_dist=raw_env.cfg.tip_align_near_dist,
        require_lift=not (raw_env._stage_1_mode and raw_env.cfg.stage1_success_without_lift),
    )
    hold_state = compute_hold_logic(
        center_y_err=center_y_err,
        yaw_err_deg=yaw_err_deg,
        insert_depth=insert_depth,
        lift_height=lift_height,
        tip_y_err=tip_y_err,
        dist_front=dist_front,
        hold_counter=raw_env._hold_counter,
        cfg=strict_cfg,
    )

    if hasattr(raw_env, "_pallet_disp_xy"):
        pallet_disp_xy = raw_env._pallet_disp_xy()
    else:
        pallet_init_pos_xy = torch.tensor(
            raw_env.cfg.pallet_cfg.init_state.pos[:2], device=raw_env.device
        )
        origin_xy = raw_env.scene.env_origins[:, :2]
        pallet_disp_xy = torch.norm(
            pallet_pos[:, :2] - (origin_xy + pallet_init_pos_xy), dim=-1
        )
    push_free = pallet_disp_xy < raw_env.cfg.push_free_disp_thresh_m
    strict_geom = hold_state.insert_entry & valid_insert_z & hold_state.align_entry & hold_state.tip_entry
    lifted_enough = lift_height >= raw_env.cfg.lift_delta_m
    strict_success = raw_env._hold_counter >= raw_env._hold_steps

    return {
        "inserted": hold_state.insert_entry,
        "strict_geom": strict_geom,
        "lifted": lifted_enough,
        "hold_entry": hold_state.hold_entry,
        "strict_success": strict_success,
        "push_free": push_free,
        "push_free_success": strict_success & push_free,
        "pallet_disp_xy": pallet_disp_xy,
        "center_lateral": center_y_err,
        "center_lateral_signed": center_y_signed,
        "tip_lateral": tip_y_err,
        "tip_lateral_signed": tip_y_signed,
        "yaw_deg": yaw_err_deg,
        "yaw_deg_signed": yaw_err * (180.0 / math.pi),
        "lift_height": lift_height,
        "dist_front": dist_front,
        "insert_norm": insert_depth / max(float(raw_env._insert_thresh), 1e-6),
    }


def _apply_hybrid_rescue(
    actions: torch.Tensor,
    state: dict[str, torch.Tensor],
    rescue_latched: torch.Tensor,
    rescue_steps: torch.Tensor,
    rescue_used: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not args.hybrid_rescue_enable:
        return actions, torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)

    center_signed = state["center_lateral_signed"]
    tip_signed = state["tip_lateral_signed"]
    yaw_signed_deg = state["yaw_deg_signed"]
    center_abs = torch.abs(center_signed)
    tip_abs = torch.abs(tip_signed)
    yaw_abs = torch.abs(yaw_signed_deg)
    insert_norm = state["insert_norm"]
    dist_front = state["dist_front"]

    misaligned = (
        (center_abs >= float(args.rescue_center_m))
        | (tip_abs >= float(args.rescue_tip_m))
        | (yaw_abs >= float(args.rescue_yaw_deg))
    )
    can_enter = (
        misaligned
        & (dist_front < float(args.rescue_trigger_dist_m))
        & (insert_norm < float(args.rescue_insert_frac_max))
    )
    if args.rescue_opposite_yaw_only:
        can_enter = can_enter & ((center_signed * yaw_signed_deg) < 0.0)
    if args.rescue_once_per_episode:
        can_enter = can_enter & (~rescue_used)

    aligned = (
        (center_abs < 0.55 * float(args.rescue_center_m))
        & (tip_abs < 0.60 * float(args.rescue_tip_m))
        & (yaw_abs < 0.85 * float(args.rescue_yaw_deg))
    )
    release = (
        (dist_front >= float(args.rescue_release_dist_m))
        | (insert_norm >= float(args.rescue_insert_frac_max))
        | aligned
    )
    max_steps = int(args.rescue_max_steps)
    if max_steps > 0:
        release = release | (rescue_steps >= max_steps)

    active = (rescue_latched | can_enter) & (~release)
    rescue_latched[:] = active.detach()
    rescue_used[:] |= can_enter.detach()
    if max_steps > 0:
        rescue_steps[:] = torch.where(
            active,
            rescue_steps + 1,
            torch.zeros_like(rescue_steps),
        )

    signal = (
        yaw_signed_deg / max(float(args.rescue_yaw_deg), 1e-6)
        + 0.35 * center_signed / max(float(args.rescue_center_m), 1e-6)
        + 0.20 * tip_signed / max(float(args.rescue_tip_m), 1e-6)
    )
    steer_sign = -torch.sign(signal)
    if args.rescue_reverse_steer_flip:
        steer_sign = -steer_sign
    steer_sign = torch.where(torch.abs(signal) > 1e-6, steer_sign, torch.zeros_like(steer_sign))
    rescue_actions = actions.clone()
    rescue_actions[:, 0] = -abs(float(args.rescue_reverse_drive))
    rescue_actions[:, 1] = steer_sign * abs(float(args.rescue_steer))
    return torch.where(active.unsqueeze(-1), rescue_actions, actions), active


def _tensor_to_bool_tensor(dones, device: str) -> torch.Tensor:
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


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
    env_cfg.use_camera = False
    env_cfg.use_asymmetric_critic = False
    env_cfg.enable_geo_edge_obs = True
    env_cfg.stage_1_mode = bool(args.stage1_eval)
    env_cfg.stage1_success_without_lift = bool(args.stage1_eval)
    env_cfg.hold_gate_curriculum_enable = False
    env_cfg.tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
    env_cfg.strict_tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
    if args.stage1_eval:
        _apply_stage1_reset_profile(env_cfg, args.reset_profile)
    if args.fixed_stage1_init is not None:
        fixed_x, fixed_y, fixed_yaw_deg = (float(v) for v in args.fixed_stage1_init)
        env_cfg.stage1_init_x_min_m = fixed_x
        env_cfg.stage1_init_x_max_m = fixed_x
        env_cfg.stage1_init_y_min_m = fixed_y
        env_cfg.stage1_init_y_max_m = fixed_y
        env_cfg.stage1_init_yaw_deg_min = fixed_yaw_deg
        env_cfg.stage1_init_yaw_deg_max = fixed_yaw_deg

    if not args.preserve_stage1_near_hard_curriculum:
        if hasattr(env_cfg, "stage1_near_hard_curriculum_enable"):
            env_cfg.stage1_near_hard_curriculum_enable = False
        if hasattr(env_cfg, "stage1_near_hard_curriculum_frac"):
            env_cfg.stage1_near_hard_curriculum_frac = 0.0
    if hasattr(env_cfg, "teacher_reference_reset_enable"):
        env_cfg.teacher_reference_reset_enable = False

    if args.device is not None:
        agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_checkpoint(runner, args.checkpoint, args.actor_only_load, raw_env.device)
    policy = runner.get_inference_policy(device=raw_env.device)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "outputs/geoedge_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.label}_summary.json"
    episode_path = output_dir / f"{args.label}_episodes.csv"

    episode_rows: list[dict[str, float | int | str]] = []
    global_counts = {
        "active_env_steps": 0.0,
        "step_inserted_sum": 0.0,
        "step_strict_geom_sum": 0.0,
        "step_lifted_sum": 0.0,
        "step_hold_entry_sum": 0.0,
        "step_success_sum": 0.0,
        "step_push_free_success_sum": 0.0,
        "step_pallet_disp_xy_sum": 0.0,
    }

    max_steps = int(raw_env.max_episode_length) + 5

    obs = env_wrapped.get_observations()
    for rollout in range(args.rollouts):
        if rollout > 0:
            obs, _ = env_wrapped.reset()
        init_robot_x = raw_env._debug_reset_x.clone()
        init_robot_y = raw_env._debug_reset_y.clone()
        init_robot_yaw_deg = raw_env._debug_reset_yaw_deg.clone()
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_inserted = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_strict_geom = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_lifted = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_hold_entry = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_success = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_push_free_success = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_max_hold_counter = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
        ep_max_pallet_disp = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
        ep_min_center_lateral = torch.full((args.num_envs,), float("inf"), dtype=torch.float32, device=raw_env.device)
        ep_min_tip_lateral = torch.full((args.num_envs,), float("inf"), dtype=torch.float32, device=raw_env.device)
        ep_min_yaw_deg = torch.full((args.num_envs,), float("inf"), dtype=torch.float32, device=raw_env.device)
        ep_max_lift_height = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
        ep_rescue_used = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_rescue_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
        ep_len = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
        ep_done_len = torch.full((args.num_envs,), -1, dtype=torch.int32, device=raw_env.device)
        rescue_latched = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        rescue_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)

        step_count = 0
        while not done_mask.all() and step_count < max_steps:
            state = _compute_step_state(raw_env)
            active = ~done_mask
            active_count = int(active.sum().item())

            if active_count > 0:
                global_counts["active_env_steps"] += active_count
                global_counts["step_inserted_sum"] += float(state["inserted"][active].sum().item())
                global_counts["step_strict_geom_sum"] += float(state["strict_geom"][active].sum().item())
                global_counts["step_lifted_sum"] += float(state["lifted"][active].sum().item())
                global_counts["step_hold_entry_sum"] += float(state["hold_entry"][active].sum().item())
                global_counts["step_success_sum"] += float(state["strict_success"][active].sum().item())
                global_counts["step_push_free_success_sum"] += float(state["push_free_success"][active].sum().item())
                global_counts["step_pallet_disp_xy_sum"] += float(state["pallet_disp_xy"][active].sum().item())

                ep_inserted[active] |= state["inserted"][active]
                ep_strict_geom[active] |= state["strict_geom"][active]
                ep_lifted[active] |= state["lifted"][active]
                ep_hold_entry[active] |= state["hold_entry"][active]
                ep_success[active] |= state["strict_success"][active]
                ep_push_free_success[active] |= state["push_free_success"][active]
                ep_max_hold_counter[active] = torch.maximum(ep_max_hold_counter[active], raw_env._hold_counter[active])
                ep_max_pallet_disp[active] = torch.maximum(ep_max_pallet_disp[active], state["pallet_disp_xy"][active])
                ep_min_center_lateral[active] = torch.minimum(ep_min_center_lateral[active], state["center_lateral"][active])
                ep_min_tip_lateral[active] = torch.minimum(ep_min_tip_lateral[active], state["tip_lateral"][active])
                ep_min_yaw_deg[active] = torch.minimum(ep_min_yaw_deg[active], state["yaw_deg"][active])
                ep_max_lift_height[active] = torch.maximum(ep_max_lift_height[active], state["lift_height"][active])
                ep_len[active] += 1

            with torch.inference_mode():
                actions = policy(obs)
                actions, rescue_active = _apply_hybrid_rescue(
                    actions.detach().clone(),
                    state,
                    rescue_latched,
                    rescue_steps,
                    ep_rescue_used,
                )
                ep_rescue_used[active] |= rescue_active[active]
                ep_rescue_steps[active] += rescue_active[active].to(torch.int32)
            obs, _, dones, _ = env_wrapped.step(actions.detach().clone())

            step_count += 1
            newly_done = _tensor_to_bool_tensor(dones, raw_env.device) & ~done_mask
            ep_done_len[newly_done] = ep_len[newly_done]
            done_mask |= newly_done

        for env_id in range(args.num_envs):
            ep_length = int(ep_done_len[env_id].item())
            if ep_length < 0:
                ep_length = int(ep_len[env_id].item())
            timeout = int(ep_length >= int(raw_env.max_episode_length) - 1 and not bool(ep_success[env_id].item()))
            dirty_insert = int(
                bool(ep_inserted[env_id].item())
                and float(ep_max_pallet_disp[env_id].item()) > float(raw_env.cfg.push_free_disp_thresh_m)
            )
            episode_rows.append(
                {
                    "label": args.label,
                    "rollout": rollout,
                    "env_id": env_id,
                    "init_x_m": float(init_robot_x[env_id].item()),
                    "init_y_m": float(init_robot_y[env_id].item()),
                    "init_yaw_deg": float(init_robot_yaw_deg[env_id].item()),
                    "strict_success": int(ep_success[env_id].item()),
                    "ever_inserted": int(ep_inserted[env_id].item()),
                    "ever_strict_geom": int(ep_strict_geom[env_id].item()),
                    "ever_lifted": int(ep_lifted[env_id].item()),
                    "ever_hold_entry": int(ep_hold_entry[env_id].item()),
                    "push_free_success": int(ep_push_free_success[env_id].item()),
                    "dirty_insert": dirty_insert,
                    "max_hold_counter": float(ep_max_hold_counter[env_id].item()),
                    "max_hold_counter_frac": float(ep_max_hold_counter[env_id].item()) / float(raw_env._hold_steps),
                    "max_pallet_disp_xy": float(ep_max_pallet_disp[env_id].item()),
                    "min_center_lateral": float(ep_min_center_lateral[env_id].item()),
                    "min_tip_lateral": float(ep_min_tip_lateral[env_id].item()),
                    "min_yaw_deg": float(ep_min_yaw_deg[env_id].item()),
                    "max_lift_height": float(ep_max_lift_height[env_id].item()),
                    "hybrid_rescue_used": int(ep_rescue_used[env_id].item()),
                    "hybrid_rescue_steps": int(ep_rescue_steps[env_id].item()),
                    "episode_length": ep_length,
                    "timeout": timeout,
                }
            )

    total_episodes = len(episode_rows)
    active_env_steps = max(global_counts["active_env_steps"], 1.0)

    def mean_bool(key: str) -> float:
        return sum(int(row[key]) for row in episode_rows) / max(total_episodes, 1)

    def mean_float(key: str) -> float:
        return sum(float(row[key]) for row in episode_rows) / max(total_episodes, 1)

    def p90_float(key: str) -> float:
        vals = sorted(float(row[key]) for row in episode_rows)
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, max(0, math.ceil(0.9 * len(vals)) - 1))
        return vals[idx]

    push_free_thresh = float(raw_env.cfg.push_free_disp_thresh_m)

    def category_rate(predicate) -> float:
        return sum(1 for row in episode_rows if predicate(row)) / max(total_episodes, 1)

    summary = {
        "label": args.label,
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_envs": args.num_envs,
        "rollouts": args.rollouts,
        "total_episodes": total_episodes,
        "eval_seed": args.seed,
        "stage1_eval": bool(args.stage1_eval),
        "reset_profile": args.reset_profile,
        "stage_1_mode": bool(raw_env._stage_1_mode),
        "stage1_success_without_lift": bool(raw_env.cfg.stage1_success_without_lift),
        "stage1_init_x_min_m": float(raw_env.cfg.stage1_init_x_min_m),
        "stage1_init_x_max_m": float(raw_env.cfg.stage1_init_x_max_m),
        "stage1_init_y_min_m": float(raw_env.cfg.stage1_init_y_min_m),
        "stage1_init_y_max_m": float(raw_env.cfg.stage1_init_y_max_m),
        "stage1_init_yaw_deg_min": float(raw_env.cfg.stage1_init_yaw_deg_min),
        "stage1_init_yaw_deg_max": float(raw_env.cfg.stage1_init_yaw_deg_max),
        "stage1_near_hard_curriculum_enable": bool(
            getattr(raw_env.cfg, "stage1_near_hard_curriculum_enable", False)
        ),
        "stage1_near_hard_curriculum_frac": float(
            getattr(raw_env.cfg, "stage1_near_hard_curriculum_frac", 0.0)
        ),
        "teacher_reference_reset_enable": bool(
            getattr(raw_env.cfg, "teacher_reference_reset_enable", False)
        ),
        "preinsert_action_guard_enable": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_enable", False)
        ),
        "preinsert_action_guard_stateful_enable": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_stateful_enable", False)
        ),
        "preinsert_action_guard_opposite_yaw_only": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_opposite_yaw_only", False)
        ),
        "preinsert_action_guard_trigger_dist_m": float(
            getattr(raw_env.cfg, "preinsert_action_guard_trigger_dist_m", 0.0)
        ),
        "preinsert_action_guard_release_dist_m": float(
            getattr(raw_env.cfg, "preinsert_action_guard_release_dist_m", 0.0)
        ),
        "preinsert_action_guard_center_m": float(
            getattr(raw_env.cfg, "preinsert_action_guard_center_m", 0.0)
        ),
        "preinsert_action_guard_tip_m": float(
            getattr(raw_env.cfg, "preinsert_action_guard_tip_m", 0.0)
        ),
        "preinsert_action_guard_yaw_deg": float(
            getattr(raw_env.cfg, "preinsert_action_guard_yaw_deg", 0.0)
        ),
        "preinsert_action_guard_insert_frac_max": float(
            getattr(raw_env.cfg, "preinsert_action_guard_insert_frac_max", 0.0)
        ),
        "preinsert_action_guard_force_reverse": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_force_reverse", False)
        ),
        "preinsert_action_guard_reverse_action": float(
            getattr(raw_env.cfg, "preinsert_action_guard_reverse_action", 0.0)
        ),
        "preinsert_action_guard_steer_action": float(
            getattr(raw_env.cfg, "preinsert_action_guard_steer_action", 0.0)
        ),
        "preinsert_action_guard_reverse_steer_flip": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_reverse_steer_flip", False)
        ),
        "preinsert_action_guard_center_steer_weight": float(
            getattr(raw_env.cfg, "preinsert_action_guard_center_steer_weight", 0.0)
        ),
        "preinsert_action_guard_tip_steer_weight": float(
            getattr(raw_env.cfg, "preinsert_action_guard_tip_steer_weight", 0.0)
        ),
        "preinsert_action_guard_yaw_steer_weight": float(
            getattr(raw_env.cfg, "preinsert_action_guard_yaw_steer_weight", 0.0)
        ),
        "preinsert_action_guard_once_per_episode": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_once_per_episode", False)
        ),
        "preinsert_action_guard_max_steps": int(
            getattr(raw_env.cfg, "preinsert_action_guard_max_steps", 0)
        ),
        "preinsert_action_guard_release_on_not_eligible": bool(
            getattr(raw_env.cfg, "preinsert_action_guard_release_on_not_eligible", True)
        ),
        "hybrid_rescue_enable": bool(args.hybrid_rescue_enable),
        "rescue_trigger_dist_m": float(args.rescue_trigger_dist_m),
        "rescue_release_dist_m": float(args.rescue_release_dist_m),
        "rescue_center_m": float(args.rescue_center_m),
        "rescue_tip_m": float(args.rescue_tip_m),
        "rescue_yaw_deg": float(args.rescue_yaw_deg),
        "rescue_insert_frac_max": float(args.rescue_insert_frac_max),
        "rescue_reverse_drive": float(args.rescue_reverse_drive),
        "rescue_steer": float(args.rescue_steer),
        "rescue_max_steps": int(args.rescue_max_steps),
        "rescue_once_per_episode": bool(args.rescue_once_per_episode),
        "rescue_reverse_steer_flip": bool(args.rescue_reverse_steer_flip),
        "rescue_opposite_yaw_only": bool(args.rescue_opposite_yaw_only),
        "strict_tip_align_entry_m": float(getattr(raw_env.cfg, "strict_tip_align_entry_m", 0.12)),
        "lift_delta_m": float(raw_env.cfg.lift_delta_m),
        "hold_time_s": float(raw_env.cfg.hold_time_s),
        "hold_steps": int(raw_env._hold_steps),
        "strict_success_rate": mean_bool("strict_success"),
        "ever_inserted_rate": mean_bool("ever_inserted"),
        "ever_strict_geom_rate": mean_bool("ever_strict_geom"),
        "ever_lifted_rate": mean_bool("ever_lifted"),
        "ever_hold_entry_rate": mean_bool("ever_hold_entry"),
        "push_free_success_rate": mean_bool("push_free_success"),
        "clean_insert_rate": category_rate(
            lambda row: int(row["ever_inserted"]) == 1 and float(row["max_pallet_disp_xy"]) <= push_free_thresh
        ),
        "dirty_insert_rate": category_rate(
            lambda row: int(row["ever_inserted"]) == 1 and float(row["max_pallet_disp_xy"]) > push_free_thresh
        ),
        "stall_no_push_rate": category_rate(
            lambda row: int(row["ever_inserted"]) == 0 and float(row["max_pallet_disp_xy"]) <= push_free_thresh
        ),
        "push_no_insert_rate": category_rate(
            lambda row: int(row["ever_inserted"]) == 0 and float(row["max_pallet_disp_xy"]) > push_free_thresh
        ),
        "push_no_insert_big_rate": category_rate(
            lambda row: int(row["ever_inserted"]) == 0 and float(row["max_pallet_disp_xy"]) > 0.5
        ),
        "timeout_frac": mean_bool("timeout"),
        "hybrid_rescue_used_rate": mean_bool("hybrid_rescue_used"),
        "mean_hybrid_rescue_steps": mean_float("hybrid_rescue_steps"),
        "mean_episode_length": mean_float("episode_length"),
        "mean_max_hold_counter": mean_float("max_hold_counter"),
        "p90_max_hold_counter": p90_float("max_hold_counter"),
        "mean_max_hold_counter_frac": mean_float("max_hold_counter_frac"),
        "mean_max_pallet_disp_xy": mean_float("max_pallet_disp_xy"),
        "p90_max_pallet_disp_xy": p90_float("max_pallet_disp_xy"),
        "mean_min_center_lateral": mean_float("min_center_lateral"),
        "mean_min_tip_lateral": mean_float("min_tip_lateral"),
        "mean_min_yaw_deg": mean_float("min_yaw_deg"),
        "mean_max_lift_height": mean_float("max_lift_height"),
        "step_mean_frac_inserted": global_counts["step_inserted_sum"] / active_env_steps,
        "step_mean_frac_strict_geom": global_counts["step_strict_geom_sum"] / active_env_steps,
        "step_mean_frac_lifted": global_counts["step_lifted_sum"] / active_env_steps,
        "step_mean_frac_hold_entry": global_counts["step_hold_entry_sum"] / active_env_steps,
        "step_mean_frac_strict_success": global_counts["step_success_sum"] / active_env_steps,
        "step_mean_frac_push_free_success": global_counts["step_push_free_success_sum"] / active_env_steps,
        "step_mean_pallet_disp_xy": global_counts["step_pallet_disp_xy_sum"] / active_env_steps,
    }

    with episode_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary to: {summary_path}")
    print(f"[INFO] Wrote episodes to: {episode_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
