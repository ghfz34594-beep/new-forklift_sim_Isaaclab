"""Replay an exact rollout pose and render dual-camera frames for diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay exact dual-camera pose metadata for visual diagnostics")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV34-v0")
parser.add_argument("--frame_meta", type=str, required=True, help="Path to frame_meta.jsonl from visual eval.")
parser.add_argument("--frame_index", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=20260528)
parser.add_argument("--env_spacing", type=float, default=None)
parser.add_argument("--camera_far", type=float, default=None)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=None)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=None)
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=None)
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=None)
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=None)
parser.add_argument("--vision_room", action="store_true", default=None)
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room")
parser.add_argument("--renders", type=int, default=3, help="Render the static pose this many times to check stability.")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument(
    "--absolute_world_pose",
    action="store_true",
    help="Replay raw world coordinates exactly. Default shifts the pose from the source env origin to the target env origin.",
)
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

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401


def _load_frame_meta(path: Path, frame_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("frame_index", -1)) == int(frame_index):
                return item
    raise RuntimeError(f"frame_index={frame_index} not found in {path}")


def _to_tensor(values: list[float], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _to_uint8_hwc(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
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


def _sha256_image(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _concat_dual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 4, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _draw_overlay(image: np.ndarray, lines: list[str]) -> np.ndarray:
    panel_h = 54
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + panel_h), (18, 18, 18))
    canvas.paste(Image.fromarray(image), (0, panel_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    y = 5
    for line in lines[:3]:
        draw.text((8, y), line, fill=(235, 235, 235), font=font)
        y += 17
    return np.asarray(canvas)


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


def _red_component_stats(image: np.ndarray, min_area: int = 250) -> dict[str, Any]:
    rgb = image[..., :3].astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (red >= 55) & (red >= green + 20) & (red >= blue + 20) & (
        red * 4 >= np.maximum(green, blue) * 5
    )
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        area = 0
        xmin = xmax = x0
        ymin = ymax = y0
        while stack:
            y, x = stack.pop()
            area += 1
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if area >= int(min_area):
            components.append(
                {
                    "area_px": int(area),
                    "bbox_xyxy": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "width_px": int(xmax - xmin + 1),
                    "height_px": int(ymax - ymin + 1),
                }
            )
    components.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return {
        "large_red_components": len(components),
        "largest_area_px": int(components[0]["area_px"]) if components else 0,
        "second_largest_area_px": int(components[1]["area_px"]) if len(components) > 1 else 0,
        "components": components[:8],
    }


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


def _apply_dual_camera_overrides(env_cfg: Any) -> None:
    if args_cli.dual_camera_hfov_deg is not None:
        env_cfg.dual_camera_hfov_deg = float(args_cli.dual_camera_hfov_deg)
    if args_cli.dual_camera_left_pos is not None:
        env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args_cli.dual_camera_left_pos)
    if args_cli.dual_camera_right_pos is not None:
        env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args_cli.dual_camera_right_pos)
    if args_cli.dual_camera_left_rpy_deg is not None:
        env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)
    if args_cli.dual_camera_right_rpy_deg is not None:
        env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_right_rpy_deg)


def _tensor_list(value: Any, env_id: int | None = None) -> list[float]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if env_id is not None:
            tensor = tensor[env_id]
        return [float(v) for v in tensor.cpu().reshape(-1).tolist()]
    arr = np.asarray(value, dtype=np.float64)
    if env_id is not None:
        arr = arr[env_id]
    return [float(v) for v in arr.reshape(-1).tolist()]


def _camera_pose(raw_env: Any, side: str, env_id: int) -> dict[str, Any]:
    camera = getattr(raw_env, f"_camera_{side}", None)
    if camera is None:
        return {"available": False}
    pose_update_error = None
    try:
        camera._update_poses([int(env_id)])
    except Exception as exc:
        pose_update_error = repr(exc)
    data = camera.data
    result: dict[str, Any] = {
        "available": True,
        "pos_w": _tensor_list(data.pos_w, env_id) if data.pos_w is not None else None,
        "quat_w_world": _tensor_list(data.quat_w_world, env_id) if data.quat_w_world is not None else None,
        "frame_counter": int(camera.frame[env_id].detach().cpu().item()) if hasattr(camera, "frame") else None,
        "pose_update_error": pose_update_error,
    }
    try:
        result["prim_path"] = str(camera._view.prim_paths[env_id])
    except Exception:
        pass
    return result


def _apply_replay_pose(raw_env: Any, meta: dict[str, Any], env_id: int) -> dict[str, Any]:
    env_ids = torch.tensor([int(env_id)], device=raw_env.device, dtype=torch.long)
    robot_pos = _to_tensor(meta["robot"]["root_pos_w"], raw_env.device).reshape(1, 3)
    pallet_pos = _to_tensor(meta["pallet"]["root_pos_w"], raw_env.device).reshape(1, 3)
    source_origin = None
    target_origin = raw_env.scene.env_origins[env_ids].reshape(1, 3)
    pose_shift = torch.zeros((1, 3), dtype=torch.float32, device=raw_env.device)
    if not bool(args_cli.absolute_world_pose) and meta.get("env_origin") is not None:
        source_origin = _to_tensor(meta["env_origin"], raw_env.device).reshape(1, 3)
        pose_shift = target_origin - source_origin
        robot_pos = robot_pos + pose_shift
        pallet_pos = pallet_pos + pose_shift
    robot_pose = torch.cat([robot_pos, _to_tensor(meta["robot"]["root_quat_w"], raw_env.device).reshape(1, 4)], dim=1)
    pallet_pose = torch.cat([pallet_pos, _to_tensor(meta["pallet"]["root_quat_w"], raw_env.device).reshape(1, 4)], dim=1)
    raw_env.robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
    raw_env.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw_env.device), env_ids=env_ids)
    raw_env.pallet.write_root_pose_to_sim(pallet_pose, env_ids=env_ids)
    raw_env.pallet.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw_env.device), env_ids=env_ids)
    joint_pos = raw_env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    raw_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    if hasattr(raw_env, "actions"):
        raw_env.actions[env_ids] = 0.0
    if raw_env.sim.has_rtx_sensors():
        raw_env.sim.render()
    raw_env.scene.update(dt=0.0)
    return {
        "absolute_world_pose": bool(args_cli.absolute_world_pose),
        "source_env_origin": _tensor_list(source_origin) if source_origin is not None else None,
        "target_env_origin": _tensor_list(target_origin),
        "pose_shift_w": _tensor_list(pose_shift),
        "replayed_robot_pos_w": _tensor_list(robot_pos),
        "replayed_pallet_pos_w": _tensor_list(pallet_pos),
    }


def _read_dual(raw_env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(raw_env, "_sync_camera_poses_to_robot"):
        raw_env._sync_camera_poses_to_robot()
    if raw_env.sim.has_rtx_sensors():
        raw_env.sim.render()
        raw_env.scene.update(dt=0.0)
    obs = raw_env._get_observations()
    return obs["image_left"], obs["image_right"]


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_frame_meta(Path(args_cli.frame_meta), int(args_cli.frame_index))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.scene.filter_collisions = True
    _apply_dual_camera_overrides(env_cfg)
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    if args_cli.camera_far is not None:
        _set_camera_far(env_cfg, float(args_cli.camera_far))
    if args_cli.vision_room is not None:
        env_cfg.vision_room_enable = bool(args_cli.vision_room)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()
    replay_pose_info = _apply_replay_pose(raw_env, meta, int(args_cli.env_id))

    left_dir = output_dir / "left_frames"
    right_dir = output_dir / "right_frames"
    dual_dir = output_dir / "dual_frames"
    frame_rows: list[dict[str, Any]] = []
    for render_idx in range(max(1, int(args_cli.renders))):
        left_batch, right_batch = _read_dual(raw_env)
        left = _to_uint8_hwc(left_batch[int(args_cli.env_id)])
        right = _to_uint8_hwc(right_batch[int(args_cli.env_id)])
        lines = [
            f"replay frame={args_cli.frame_index} render={render_idx} env={args_cli.env_id}/{args_cli.num_envs}",
            f"source ep={meta.get('episode')} step={meta.get('episode_step')} done_any={int(bool(meta.get('done_any')))}",
            f"robot=({meta['robot']['root_pos_w'][0]:+.3f},{meta['robot']['root_pos_w'][1]:+.3f}) yaw={meta['robot']['yaw_deg']:+.2f}",
        ]
        _save_png(left_dir / f"frame_{render_idx:06d}.png", _draw_overlay(left, lines))
        _save_png(right_dir / f"frame_{render_idx:06d}.png", _draw_overlay(right, lines))
        _save_png(dual_dir / f"frame_{render_idx:06d}.png", _draw_overlay(_concat_dual(left, right), lines))
        frame_rows.append(
            {
                "render_index": int(render_idx),
                "left_sha256": _sha256_image(left),
                "right_sha256": _sha256_image(right),
                "left_red": _red_component_stats(left),
                "right_red": _red_component_stats(right),
                "camera_left": _camera_pose(raw_env, "left", int(args_cli.env_id)),
                "camera_right": _camera_pose(raw_env, "right", int(args_cli.env_id)),
            }
        )

    videos_ok = {
        "left": _make_video(left_dir, output_dir / "left.mp4", int(args_cli.fps)),
        "right": _make_video(right_dir, output_dir / "right.mp4", int(args_cli.fps)),
        "dual": _make_video(dual_dir, output_dir / "dual_camera.mp4", int(args_cli.fps)),
    }
    hashes_left = {row["left_sha256"] for row in frame_rows}
    hashes_right = {row["right_sha256"] for row in frame_rows}
    summary = {
        "task": str(args_cli.task),
        "frame_meta": str(args_cli.frame_meta),
        "frame_index": int(args_cli.frame_index),
        "source_meta": meta,
        "replay_pose_info": replay_pose_info,
        "num_envs": int(raw_env.num_envs),
        "env_id": int(args_cli.env_id),
        "renders": int(len(frame_rows)),
        "static_hash_stable": bool(len(hashes_left) == 1 and len(hashes_right) == 1),
        "frames": frame_rows,
        "videos_ok": videos_ok,
        "left_video": str(output_dir / "left.mp4"),
        "right_video": str(output_dir / "right.mp4"),
        "dual_camera_video": str(output_dir / "dual_camera.mp4"),
        "dual_camera_config": {
            "hfov_deg": float(raw_env.cfg.dual_camera_hfov_deg),
            "left_pos_local": [float(v) for v in raw_env.cfg.dual_camera_left_pos_local],
            "right_pos_local": [float(v) for v in raw_env.cfg.dual_camera_right_pos_local],
            "left_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_left_rpy_local_deg],
            "right_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_right_rpy_local_deg],
            "camera_far": float(getattr(raw_env.cfg, "dual_camera_far_clip_m", -1.0)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
