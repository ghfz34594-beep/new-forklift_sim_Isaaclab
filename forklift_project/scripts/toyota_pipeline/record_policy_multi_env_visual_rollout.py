"""Record a multi-env mosaic rollout for a visual RSL-RL forklift policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record multi-env dual-camera mosaic rollout for a checkpoint")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV39-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--record_every", type=int, default=1)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--seed", type=int, default=20260528)
parser.add_argument("--mosaic_max_envs", type=int, default=16)
parser.add_argument("--mosaic_cols", type=int, default=4)
parser.add_argument("--mosaic_env_ids", type=int, nargs="*", default=None)
parser.add_argument("--env_spacing", type=float, default=None)
parser.add_argument("--camera_far", type=float, default=None)
parser.add_argument("--vision_room", action="store_true", default=None, help="Force per-env room occlusion on.")
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room", help="Force per-env room occlusion off.")
parser.add_argument(
    "--disable_teacher_reference_reset",
    action="store_true",
    help="Use the task's normal reset distribution instead of teacher-reference starts.",
)
parser.add_argument(
    "--teacher_reference_reset_mix",
    type=float,
    default=None,
    help="Override teacher-reference reset probability by setting both mix_start and mix_end.",
)
parser.add_argument("--save_tiles", action="store_true", help="Save per-env dual-camera tiles for every recorded frame.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import _quat_to_yaw
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
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


def _font(size: int = 12):
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _tile_with_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    label_h = 48
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + label_h), (16, 16, 16))
    canvas.paste(Image.fromarray(image), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    font = _font(11)
    y = 3
    for line in lines[:3]:
        draw.text((5, y), line, fill=(235, 235, 235), font=font)
        y += 14
    return np.asarray(canvas)


def _make_grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    if not tiles:
        raise ValueError("Cannot make a mosaic without tiles")
    cols = max(1, int(cols))
    rows = int(math.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return canvas


def _concat_dual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 4, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _make_video(frame_dir: Path, output_path: Path, fps: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(output_path),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


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


def _selected_env_ids(num_envs: int) -> list[int]:
    if args_cli.mosaic_env_ids:
        env_ids = sorted({int(env_id) for env_id in args_cli.mosaic_env_ids})
    else:
        env_ids = list(range(min(int(args_cli.mosaic_max_envs), int(num_envs))))
    bad = [env_id for env_id in env_ids if env_id < 0 or env_id >= int(num_envs)]
    if bad:
        raise ValueError(f"mosaic env ids outside [0, {int(num_envs)}): {bad}")
    return env_ids


def _metric_rows(raw_env: Any, action: torch.Tensor, effective_action: torch.Tensor, step: int) -> list[dict[str, Any]]:
    root_pos = raw_env.robot.data.root_pos_w
    root_quat = raw_env.robot.data.root_quat_w
    pallet_pos = raw_env.pallet.data.root_pos_w
    pallet_quat = raw_env.pallet.data.root_quat_w
    yaw = _quat_to_yaw(root_quat)
    pallet_yaw = _quat_to_yaw(pallet_quat)
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)
    tip = raw_env._compute_fork_tip()
    center = raw_env._compute_fork_center()
    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    rel_center = center[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * float(raw_env.cfg.pallet_depth_m)
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / (float(raw_env.cfg.pallet_depth_m) + 1e-6), 0.0, 1.0)
    tip_lateral = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
    center_lateral = torch.abs(torch.sum(rel_center * v_lat, dim=-1))
    yaw_err_deg = torch.abs(torch.atan2(torch.sin(yaw - pallet_yaw), torch.cos(yaw - pallet_yaw))) * 180.0 / math.pi
    if hasattr(raw_env, "_pallet_disp_xy"):
        pallet_disp_xy = raw_env._pallet_disp_xy()
    else:
        pallet_init_xy = torch.tensor(raw_env.cfg.pallet_cfg.init_state.pos[:2], device=raw_env.device)
        pallet_disp_xy = torch.norm(pallet_pos[:, :2] - (raw_env.scene.env_origins[:, :2] + pallet_init_xy), dim=-1)
    success = getattr(raw_env, "_success_termination", torch.zeros(raw_env.num_envs, device=raw_env.device, dtype=torch.bool))
    hold_counter = getattr(raw_env, "_hold_counter", torch.zeros(raw_env.num_envs, device=raw_env.device))
    rows: list[dict[str, Any]] = []
    for env_id in range(int(raw_env.num_envs)):
        rows.append(
            {
                "step": int(step),
                "env_id": int(env_id),
                "raw_drive": float(action[env_id, 0].detach().cpu().item()),
                "raw_steer": float(action[env_id, 1].detach().cpu().item()) if action.shape[1] > 1 else 0.0,
                "drive": float(effective_action[env_id, 0].detach().cpu().item()),
                "steer": float(effective_action[env_id, 1].detach().cpu().item()) if effective_action.shape[1] > 1 else 0.0,
                "stage_dist_front_m": float(dist_front[env_id].detach().cpu().item()),
                "tip_lateral_m": float(tip_lateral[env_id].detach().cpu().item()),
                "center_lateral_m": float(center_lateral[env_id].detach().cpu().item()),
                "yaw_err_deg": float(yaw_err_deg[env_id].detach().cpu().item()),
                "insert_depth_m": float(insert_depth[env_id].detach().cpu().item()),
                "insert_norm": float(insert_norm[env_id].detach().cpu().item()),
                "pallet_disp_xy_m": float(pallet_disp_xy[env_id].detach().cpu().item()),
                "hold_counter": float(hold_counter[env_id].detach().cpu().item()),
                "success": bool(success[env_id].detach().cpu().item()),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    frame_dir = output_dir / "mosaic_dual_frames"
    tile_dir = output_dir / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = int(args_cli.seed)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.scene.filter_collisions = True
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    if args_cli.camera_far is not None:
        _set_camera_far(env_cfg, float(args_cli.camera_far))
    if args_cli.vision_room is not None:
        env_cfg.vision_room_enable = bool(args_cli.vision_room)
    if bool(args_cli.disable_teacher_reference_reset):
        env_cfg.teacher_reference_reset_enable = False
    if args_cli.teacher_reference_reset_mix is not None:
        mix = max(0.0, min(1.0, float(args_cli.teacher_reference_reset_mix)))
        env_cfg.teacher_reference_reset_enable = mix > 0.0
        env_cfg.teacher_reference_reset_mix_start = mix
        env_cfg.teacher_reference_reset_mix_end = mix
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    selected_env_ids = _selected_env_ids(int(raw_env.num_envs))

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)

    obs, _ = wrapped.reset()
    rows: list[dict[str, Any]] = []
    saved_frames = 0
    for step in range(int(args_cli.steps)):
        with torch.inference_mode():
            action = policy(obs)
        if agent_cfg.clip_actions is not None:
            effective_action = torch.clamp(action.detach().clone(), -float(agent_cfg.clip_actions), float(agent_cfg.clip_actions))
        else:
            effective_action = action.detach().clone()
        obs, _, dones, _ = wrapped.step(action.detach().clone())

        if step % max(1, int(args_cli.record_every)) != 0:
            continue
        if hasattr(raw_env, "_sync_camera_poses_to_robot"):
            raw_env._sync_camera_poses_to_robot()
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()
            raw_env.scene.update(dt=0.0)
        raw_obs = raw_env._get_observations()
        left_batch = raw_obs["image_left"]
        right_batch = raw_obs["image_right"]
        metric_rows = _metric_rows(raw_env, action, effective_action, step)
        for row in metric_rows:
            row["frame"] = int(saved_frames)
            row["done"] = bool(torch.as_tensor(dones)[row["env_id"]].detach().cpu().item())
        rows.extend(metric_rows)
        rows_by_env = {int(row["env_id"]): row for row in metric_rows}

        tiles: list[np.ndarray] = []
        for env_id in selected_env_ids:
            left = _to_uint8_hwc(left_batch[env_id])
            right = _to_uint8_hwc(right_batch[env_id])
            dual = _concat_dual(left, right)
            row = rows_by_env[env_id]
            label = [
                f"env={env_id:03d} step={step:03d} act=({row['drive']:+.2f},{row['steer']:+.2f})",
                f"dist={row['stage_dist_front_m']:.3f} y={row['tip_lateral_m']:.3f} yaw={row['yaw_err_deg']:.1f}",
                f"ins={row['insert_norm']:.2f} hold={row['hold_counter']:.0f} succ={int(row['success'])}",
            ]
            tile = _tile_with_label(dual, label)
            tiles.append(tile)
            if bool(args_cli.save_tiles):
                _save_png(tile_dir / f"env_{env_id:03d}" / f"frame_{saved_frames:06d}.png", tile)
        _save_png(frame_dir / f"frame_{saved_frames:06d}.png", _make_grid(tiles, int(args_cli.mosaic_cols)))
        saved_frames += 1

    metrics_csv = output_dir / "metrics.csv"
    _write_csv(metrics_csv, rows)
    video_path = output_dir / "policy_multi_env_dual_mosaic.mp4"
    videos_ok = {"dual_mosaic": _make_video(frame_dir, video_path, int(args_cli.fps))}
    summary = {
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_envs": int(raw_env.num_envs),
        "selected_env_ids": selected_env_ids,
        "steps": int(args_cli.steps),
        "record_every": int(args_cli.record_every),
        "fps": int(args_cli.fps),
        "saved_frames": int(saved_frames),
        "duration_s": float(saved_frames / max(1, int(args_cli.fps))),
        "teacher_reference_reset_enable": bool(getattr(env_cfg, "teacher_reference_reset_enable", False)),
        "teacher_reference_reset_mix_start": float(getattr(env_cfg, "teacher_reference_reset_mix_start", 0.0)),
        "teacher_reference_reset_mix_end": float(getattr(env_cfg, "teacher_reference_reset_mix_end", 0.0)),
        "vision_room_enable": bool(getattr(env_cfg, "vision_room_enable", False)),
        "metrics_csv": str(metrics_csv),
        "video": str(video_path),
        "videos_ok": videos_ok,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
