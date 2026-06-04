"""Record one GeoEdge Stage-A episode only after a successful insertion.

This helper keeps the policy input as the 21D GeometryEdge observation. Rendering
is used only for the saved global-view mp4.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Record a successful GeoEdge insertion video")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_attempts", type=int, default=64)
parser.add_argument("--video_seconds", type=float, default=10.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--camera_eye", type=float, nargs=3, default=(-4.0, -6.0, 4.0))
parser.add_argument("--camera_lookat", type=float, nargs=3, default=(-1.5, 0.0, 0.2))
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
    help="Use insert-only Stage A settings: lift-free success and Stage A reset distribution.",
)
parser.add_argument(
    "--reset_profile",
    "--reset-profile",
    choices=("default", "near", "mid", "full"),
    default="near",
    help="Optional Stage A reset profile override used only with --stage1_eval.",
)
parser.add_argument(
    "--require_clean",
    "--require-clean",
    action="store_true",
    help="Save only if success happens while max pallet displacement stays within push-free threshold.",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import (
    HoldLogicConfig,
    compute_hold_logic,
)
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
    center_y_err = torch.abs(torch.sum(rel_fc * v_lat, dim=-1))
    tip_y_err = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
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
    strict_geom = hold_state.insert_entry & valid_insert_z & hold_state.align_entry & hold_state.tip_entry
    strict_success = raw_env._hold_counter >= raw_env._hold_steps

    return {
        "inserted": hold_state.insert_entry,
        "strict_geom": strict_geom,
        "strict_success": strict_success,
        "pallet_disp_xy": pallet_disp_xy,
        "center_lateral": center_y_err,
        "tip_lateral": tip_y_err,
        "yaw_deg": yaw_err_deg,
    }


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


def _to_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] > 3:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)


def main() -> None:
    target_frames = max(1, int(round(args_cli.video_seconds * args_cli.fps)))
    video_path = Path(args_cli.video_path).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    env_cfg.seed = args_cli.seed
    env_cfg.use_camera = False
    env_cfg.use_asymmetric_critic = False
    env_cfg.enable_geo_edge_obs = True
    env_cfg.stage_1_mode = bool(args_cli.stage1_eval)
    env_cfg.stage1_success_without_lift = bool(args_cli.stage1_eval)
    env_cfg.hold_gate_curriculum_enable = False
    env_cfg.tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
    env_cfg.strict_tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
    env_cfg.viewer.resolution = (int(args_cli.width), int(args_cli.height))
    env_cfg.viewer.eye = tuple(float(v) for v in args_cli.camera_eye)
    env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.camera_lookat)
    if args_cli.stage1_eval:
        _apply_stage1_reset_profile(env_cfg, args_cli.reset_profile)
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env_wrapped.unwrapped
    raw_env.sim.set_camera_view(eye=args_cli.camera_eye, target=args_cli.camera_lookat)

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_checkpoint(runner, args_cli.checkpoint, args_cli.actor_only_load, raw_env.device)
    policy = runner.get_inference_policy(device=raw_env.device)

    max_steps = min(int(raw_env.max_episode_length), target_frames)
    push_free_thresh = float(raw_env.cfg.push_free_disp_thresh_m)

    for attempt in range(1, args_cli.max_attempts + 1):
        obs, _ = env_wrapped.reset()
        frames: list[np.ndarray] = []
        ever_inserted = False
        ever_strict_geom = False
        max_hold_counter = 0.0
        max_pallet_disp = 0.0
        min_center_lateral = float("inf")
        min_tip_lateral = float("inf")
        min_yaw_deg = float("inf")

        for step in range(max_steps):
            frame = env.render(recompute=True)
            if frame is not None:
                frames.append(_to_frame(frame))

            state = _compute_step_state(raw_env)
            ever_inserted |= bool(state["inserted"][0].item())
            ever_strict_geom |= bool(state["strict_geom"][0].item())
            max_hold_counter = max(max_hold_counter, float(raw_env._hold_counter[0].item()))
            max_pallet_disp = max(max_pallet_disp, float(state["pallet_disp_xy"][0].item()))
            min_center_lateral = min(min_center_lateral, float(state["center_lateral"][0].item()))
            min_tip_lateral = min(min_tip_lateral, float(state["tip_lateral"][0].item()))
            min_yaw_deg = min(min_yaw_deg, float(state["yaw_deg"][0].item()))

            strict_success = bool(state["strict_success"][0].item())
            clean_enough = max_pallet_disp <= push_free_thresh
            target_met = strict_success and (clean_enough or not args_cli.require_clean)
            if target_met:
                while len(frames) < target_frames and frames:
                    frames.append(frames[-1].copy())
                imageio.mimsave(video_path, frames[:target_frames], fps=args_cli.fps, macro_block_size=1)
                print(
                    "[INFO] saved_success_video "
                    f"path={video_path} attempt={attempt} steps={step + 1} "
                    f"strict_success=1 clean={int(clean_enough)} ever_inserted={int(ever_inserted)} "
                    f"ever_strict_geom={int(ever_strict_geom)} max_hold_counter={max_hold_counter:.1f} "
                    f"max_pallet_disp_xy={max_pallet_disp:.6f} "
                    f"min_center_lateral={min_center_lateral:.6f} "
                    f"min_tip_lateral={min_tip_lateral:.6f} min_yaw_deg={min_yaw_deg:.3f}"
                )
                env_wrapped.close()
                return

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env_wrapped.step(actions.detach().clone())

            if bool(dones[0].item()):
                break

        print(
            "[INFO] discard_attempt "
            f"attempt={attempt} steps={len(frames)} ever_inserted={int(ever_inserted)} "
            f"ever_strict_geom={int(ever_strict_geom)} max_hold_counter={max_hold_counter:.1f} "
            f"max_pallet_disp_xy={max_pallet_disp:.6f} "
            f"min_center_lateral={min_center_lateral:.6f} "
            f"min_tip_lateral={min_tip_lateral:.6f} min_yaw_deg={min_yaw_deg:.3f}"
        )

    env_wrapped.close()
    raise RuntimeError(
        f"No successful insertion found in {args_cli.max_attempts} attempts "
        f"(require_clean={args_cli.require_clean}, reset_profile={args_cli.reset_profile})."
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
