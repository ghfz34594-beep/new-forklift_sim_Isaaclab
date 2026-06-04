#!/usr/bin/env python3
"""Collect CleanView RGB samples with signed geometry labels.

This is a diagnostic dataset for direct visual RL. It records dual CleanView
RGB plus signed lateral/yaw geometry from the simulator; it does not record
teacher actions and is not a BC/student dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect CleanView signed geometry probe data")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV31-v0")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--samples", type=int, default=2048)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--record_every", type=int, default=1)
parser.add_argument("--reset_every", type=int, default=4)
parser.add_argument("--seed", type=int, default=20260527)
parser.add_argument("--env_spacing", type=float, default=20.0)
parser.add_argument("--camera_far", type=float, default=8.0)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=100.0)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(150.0, 75.0, 140.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(150.0, -75.0, 140.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, -20.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, 20.0))
parser.add_argument("--vision_room", action="store_true", default=False)
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room")
parser.add_argument("--random_drive_abs", type=float, default=0.20)
parser.add_argument("--random_steer_abs", type=float, default=0.45)
parser.add_argument("--min_abs_signed_lateral_m", type=float, default=0.03)
parser.add_argument("--min_abs_yaw_deg", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import _quat_to_yaw
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401


def _to_uint8_hwc(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        arr = image.numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _set_camera_far(env_cfg: Any, far: float) -> None:
    if hasattr(env_cfg, "dual_camera_far_clip_m"):
        env_cfg.dual_camera_far_clip_m = float(far)
    for name in ("tiled_camera_left", "tiled_camera_right"):
        camera_cfg = getattr(env_cfg, name, None)
        if camera_cfg is None:
            continue
        near = 0.1
        try:
            near = float(camera_cfg.spawn.clipping_range[0])
        except Exception:
            pass
        camera_cfg.spawn.clipping_range = (near, float(far))


def _apply_camera_overrides(env_cfg: Any) -> None:
    env_cfg.dual_camera_hfov_deg = float(args_cli.dual_camera_hfov_deg)
    env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args_cli.dual_camera_left_pos)
    env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args_cli.dual_camera_right_pos)
    env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)
    env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_right_rpy_deg)


def _geometry_rows(raw_env) -> dict[str, torch.Tensor]:
    root_pos = raw_env.robot.data.root_pos_w
    pallet_pos = raw_env.pallet.data.root_pos_w
    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    tip = raw_env._compute_fork_tip()
    center = raw_env._compute_fork_center()
    rel_root = root_pos[:, :2] - pallet_pos[:, :2]
    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    rel_center = center[:, :2] - pallet_pos[:, :2]
    s_front = -0.5 * float(raw_env.cfg.pallet_depth_m)
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    signed_lateral = torch.sum(rel_root * v_lat, dim=-1)
    tip_signed = torch.sum(rel_tip * v_lat, dim=-1)
    center_signed = torch.sum(rel_center * v_lat, dim=-1)
    yaw_signed_deg = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
    yaw_signed_deg = yaw_signed_deg * 180.0 / math.pi
    pallet_disp_xy = raw_env._pallet_disp_xy() if hasattr(raw_env, "_pallet_disp_xy") else torch.zeros_like(dist_front)
    return {
        "signed_lateral_err_m": signed_lateral,
        "tip_lateral_signed_m": tip_signed,
        "center_lateral_signed_m": center_signed,
        "yaw_err_signed_deg": yaw_signed_deg,
        "tip_lateral_err_m": torch.abs(tip_signed),
        "center_lateral_err_m": torch.abs(center_signed),
        "yaw_err_deg": torch.abs(yaw_signed_deg),
        "dist_front_m": dist_front,
        "insert_depth_m": insert_depth,
        "pallet_disp_xy_m": pallet_disp_xy,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=int(args_cli.num_envs))
    env_cfg.seed = int(args_cli.seed)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    env_cfg.scene.filter_collisions = True
    env_cfg.vision_room_enable = bool(args_cli.vision_room)
    _set_camera_far(env_cfg, float(args_cli.camera_far))
    _apply_camera_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    obs, _ = env.reset()
    del obs

    rows: list[dict[str, Any]] = []
    lateral_pos = lateral_neg = yaw_pos = yaw_neg = 0
    sample_id = 0
    record_every = max(1, int(args_cli.record_every))
    reset_every = max(1, int(args_cli.reset_every))
    action_dim = int(getattr(raw_env.cfg, "action_space", 2))

    for step in range(int(args_cli.steps)):
        if step > 0 and step % reset_every == 0:
            env.reset()
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()
        raw_obs = raw_env._get_observations()
        if step % record_every == 0:
            left_batch = raw_obs["image_left"]
            right_batch = raw_obs["image_right"]
            geom = _geometry_rows(raw_env)
            for env_id in range(raw_env.num_envs):
                if sample_id >= int(args_cli.samples):
                    break
                signed_lateral = float(geom["signed_lateral_err_m"][env_id].detach().cpu().item())
                yaw_signed = float(geom["yaw_err_signed_deg"][env_id].detach().cpu().item())
                if abs(signed_lateral) < float(args_cli.min_abs_signed_lateral_m):
                    continue
                if abs(yaw_signed) < float(args_cli.min_abs_yaw_deg):
                    continue
                left_rel = f"left/{sample_id:06d}.png"
                right_rel = f"right/{sample_id:06d}.png"
                _save_png(left_dir / f"{sample_id:06d}.png", _to_uint8_hwc(left_batch[env_id]))
                _save_png(right_dir / f"{sample_id:06d}.png", _to_uint8_hwc(right_batch[env_id]))
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "step": step,
                    "env_id": int(env_id),
                    "image_left": left_rel,
                    "image_right": right_rel,
                }
                for key, values in geom.items():
                    row[key] = float(values[env_id].detach().cpu().item())
                rows.append(row)
                lateral_pos += int(signed_lateral >= 0.0)
                lateral_neg += int(signed_lateral < 0.0)
                yaw_pos += int(yaw_signed >= 0.0)
                yaw_neg += int(yaw_signed < 0.0)
                sample_id += 1
            if sample_id >= int(args_cli.samples):
                break

        actions = torch.zeros((raw_env.num_envs, action_dim), dtype=torch.float32, device=raw_env.device)
        actions[:, 0] = torch.rand((raw_env.num_envs,), device=raw_env.device) * float(args_cli.random_drive_abs)
        if action_dim >= 2:
            actions[:, 1] = (
                torch.rand((raw_env.num_envs,), device=raw_env.device) * 2.0 - 1.0
            ) * float(args_cli.random_steer_abs)
        _, _, terminated, truncated, _ = env.step(actions)
        if bool((terminated | truncated).any().detach().cpu().item()):
            env.reset()

    _write_csv(output_dir / "metadata.csv", rows)
    summary = {
        "task": args_cli.task,
        "output_dir": str(output_dir),
        "num_samples": len(rows),
        "num_envs": int(args_cli.num_envs),
        "env_spacing": float(args_cli.env_spacing),
        "camera_far": float(args_cli.camera_far),
        "hfov": float(args_cli.dual_camera_hfov_deg),
        "vision_room_enable": bool(args_cli.vision_room),
        "lateral_pos": lateral_pos,
        "lateral_neg": lateral_neg,
        "yaw_pos": yaw_pos,
        "yaw_neg": yaw_neg,
        "metadata_path": str(output_dir / "metadata.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
