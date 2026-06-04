"""Replay a recorded visual-eval trajectory with a different dual-camera rig."""

from __future__ import annotations

import argparse
from collections import deque
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--frame_meta_jsonl", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--skip_initial_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=0)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
parser.add_argument("--env_spacing", type=float, default=20.0)
parser.add_argument("--camera_far", type=float, default=8.0)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=100.0)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(150.0, 75.0, 140.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(150.0, -75.0, 140.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, -20.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, 20.0))
parser.add_argument("--topdown_camera_eye", type=float, nargs=3, default=(-2.7, 0.0, 7.0))
parser.add_argument("--topdown_camera_lookat", type=float, nargs=3, default=(-2.7, 0.0, 0.0))
parser.add_argument("--topdown_render_retries", type=int, default=3)
parser.add_argument(
    "--highlight_pallet",
    action="store_true",
    help="Bind a temporary green material to the pallet in this diagnostic replay for visibility metrics.",
)
parser.add_argument("--pallet_visible_min_area_px", type=int, default=250)
parser.add_argument("--fork_visible_min_area_px", type=int, default=50)
parser.add_argument(
    "--metrics_only",
    action="store_true",
    help="Only render dual cameras and compute visibility metrics; skip frame/video output.",
)
parser.add_argument("--vision_room", action="store_true", default=None)
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room")
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
from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401


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


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_panel(image: np.ndarray, lines: list[str], panel_h: int = 64) -> np.ndarray:
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + panel_h), (18, 18, 18))
    canvas.paste(Image.fromarray(image), (0, panel_h))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    y = 6
    for line in lines[:3]:
        draw.text((8, y), line, fill=(235, 235, 235), font=font)
        y += 19
    return np.asarray(canvas)


def _concat_dual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 4, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    scale = min(width / pil.width, height / pil.height)
    size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
    pil = pil.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (238, 240, 242))
    canvas.paste(pil, ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(canvas)


def _make_row(topdown: np.ndarray, dual: np.ndarray) -> np.ndarray:
    panel_w = 720
    panel_h = 456
    label_h = 34
    gutter = 10
    width = panel_w * 2 + gutter
    height = panel_h + label_h
    canvas = Image.new("RGB", (width, height), (20, 22, 24))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    draw.rectangle((0, 0, panel_w, label_h), fill=(18, 18, 18))
    draw.rectangle((panel_w + gutter, 0, width, label_h), fill=(18, 18, 18))
    draw.text((10, 7), "third-person topdown", fill=(232, 235, 238), font=title_font)
    draw.text((panel_w + gutter + 10, 7), "forklift left/right cameras", fill=(232, 235, 238), font=title_font)
    canvas.paste(Image.fromarray(_fit(topdown, panel_w, panel_h)), (0, label_h))
    canvas.paste(Image.fromarray(_fit(dual, panel_w, panel_h)), (panel_w + gutter, label_h))
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
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(output_path),
    ]
    return subprocess.run(cmd, check=False).returncode == 0 and output_path.is_file()


def _is_blank_frame(image: np.ndarray) -> bool:
    rgb = image[..., :3].astype(np.float32)
    return bool(rgb.mean() < 5.0 and rgb.std() < 2.0)


def _set_camera_far(env_cfg: Any, far: float) -> None:
    if hasattr(env_cfg, "dual_camera_far_clip_m"):
        env_cfg.dual_camera_far_clip_m = float(far)
    for name in ("tiled_camera_left", "tiled_camera_right"):
        cfg = getattr(env_cfg, name, None)
        if cfg is None:
            continue
        near = float(cfg.spawn.clipping_range[0]) if cfg.spawn.clipping_range else 0.1
        cfg.spawn.clipping_range = (near, float(far))


def _apply_dual_camera_overrides(env_cfg: Any) -> None:
    env_cfg.dual_camera_hfov_deg = float(args_cli.dual_camera_hfov_deg)
    env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args_cli.dual_camera_left_pos)
    env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args_cli.dual_camera_right_pos)
    env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)
    env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_right_rpy_deg)


def _set_topdown_camera(raw_env: Any) -> None:
    origin = raw_env.scene.env_origins[0].detach().cpu().numpy()
    eye = origin + np.asarray(args_cli.topdown_camera_eye, dtype=np.float32)
    lookat = origin + np.asarray(args_cli.topdown_camera_lookat, dtype=np.float32)
    raw_env.sim.set_camera_view(eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in lookat))


def _render_topdown(env: Any, raw_env: Any) -> np.ndarray:
    last: np.ndarray | None = None
    retries = max(1, int(args_cli.topdown_render_retries))
    for _ in range(retries):
        _set_topdown_camera(raw_env)
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()
            if hasattr(raw_env, "_force_dual_camera_update"):
                raw_env._force_dual_camera_update()
            raw_env.scene.update(dt=0.0)
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; use render_mode='rgb_array'.")
        last = _to_uint8_hwc(frame)
        if not _is_blank_frame(last):
            return last
    if last is None:
        raise RuntimeError("topdown render produced no frames")
    return last


def _make_preview_material(
    stage: Usd.Stage,
    path: str,
    diffuse_color: tuple[float, float, float],
    roughness: float = 0.6,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material_recursive(root_prim: Usd.Prim, material: UsdShade.Material) -> int:
    count = 0
    if root_prim and root_prim.IsValid():
        UsdShade.MaterialBindingAPI(root_prim).Bind(material)
        count += 1
        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim:
                continue
            if prim.IsA(UsdGeom.Gprim) or prim.GetTypeName() in ("Mesh", "Cube"):
                UsdShade.MaterialBindingAPI(prim).Bind(material)
                count += 1
    return count


def _highlight_pallets(raw_env: Any) -> dict[str, Any]:
    stage = raw_env.sim.stage
    material = _make_preview_material(stage, "/World/ReplayPalletVisibilityMaterial", (0.0, 1.0, 0.0), 0.45)
    bound_envs: list[int] = []
    bound_prims = 0
    for env_id in range(int(raw_env.num_envs)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Pallet")
        count = _bind_material_recursive(prim, material)
        if count > 0:
            bound_envs.append(int(env_id))
            bound_prims += int(count)
    return {
        "enabled": True,
        "bound_env_count": int(len(bound_envs)),
        "bound_prim_count": int(bound_prims),
        "sample_env_ids": bound_envs[:16],
        "pass": len(bound_envs) == int(raw_env.num_envs),
    }


def _component_stats(mask: np.ndarray, min_area: int, component_key: str) -> dict[str, Any]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    total_area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        visited[y0, x0] = True
        area = 0
        xmin = xmax = x0
        ymin = ymax = y0
        while queue:
            y, x = queue.popleft()
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
                    queue.append((ny, nx))
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
        "total_area_px": int(total_area),
        component_key: int(len(components)),
        "largest_area_px": int(components[0]["area_px"]) if components else 0,
        "components": components[:8],
    }


def _red_component_stats(image: np.ndarray, min_area: int) -> dict[str, Any]:
    rgb = image[..., :3].astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (red >= 55) & (red >= green + 20) & (red >= blue + 20) & (
        red * 4 >= np.maximum(green, blue) * 5
    )
    return _component_stats(mask, min_area, "large_red_components")


def _green_component_stats(image: np.ndarray, min_area: int) -> dict[str, Any]:
    rgb = image[..., :3].astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (
        (green >= 70)
        & (green >= red + 28)
        & (green >= blue + 28)
        & (green * 4 >= np.maximum(red, blue) * 5)
    )
    return _component_stats(mask, min_area, "large_green_components")


def _load_meta(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip = max(0, int(args_cli.skip_initial_frames))
    if skip > 0:
        records = records[skip:]
    stride = max(1, int(args_cli.stride))
    records = records[::stride]
    if int(args_cli.max_frames) > 0:
        records = records[: int(args_cli.max_frames)]
    return records


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_meta(Path(args_cli.frame_meta_jsonl))
    if not records:
        raise RuntimeError("No metadata records to replay.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = 20260604
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.viewer.resolution = (int(args_cli.video_width), int(args_cli.video_height))
    env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    _set_camera_far(env_cfg, float(args_cli.camera_far))
    _apply_dual_camera_overrides(env_cfg)
    if args_cli.vision_room is not None:
        env_cfg.vision_room_enable = bool(args_cli.vision_room)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = env.unwrapped
    env.reset()
    pallet_highlight = {"enabled": False, "pass": True}
    if bool(args_cli.highlight_pallet):
        pallet_highlight = _highlight_pallets(raw_env)
    env_ids = torch.tensor([0], dtype=torch.long, device=raw_env.device)
    topdown_dir = output_dir / "topdown_frames"
    dual_dir = output_dir / "dual_camera_frames"
    row_dir = output_dir / "side_by_side_frames"
    stats: list[dict[str, Any]] = []

    for frame_index, meta in enumerate(records):
        robot_pose = torch.tensor(
            [meta["robot"]["root_pos_w"] + meta["robot"]["root_quat_w"]],
            dtype=torch.float32,
            device=raw_env.device,
        )
        pallet_pose = torch.tensor(
            [meta["pallet"]["root_pos_w"] + meta["pallet"]["root_quat_w"]],
            dtype=torch.float32,
            device=raw_env.device,
        )
        raw_env.robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        raw_env.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw_env.device), env_ids=env_ids)
        raw_env.pallet.write_root_pose_to_sim(pallet_pose, env_ids=env_ids)
        raw_env.pallet.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw_env.device), env_ids=env_ids)
        joint_pos = raw_env.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        raw_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        raw_env.scene.write_data_to_sim()
        raw_env.scene.update(dt=0.0)
        if hasattr(raw_env, "_sync_dual_camera_poses"):
            raw_env._sync_dual_camera_poses(
                env_ids,
                root_pos=robot_pose[:, :3],
                root_quat=robot_pose[:, 3:7],
            )
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()
            raw_env.scene.update(dt=0.0)

        left_batch, right_batch = raw_env._get_dual_camera_images()
        left = _to_uint8_hwc(left_batch[0])
        right = _to_uint8_hwc(right_batch[0])
        row = meta.get("row", {})
        labels = [
            (
                f"replay ep={meta.get('episode')} step={meta.get('episode_step')} "
                f"drive={float(row.get('drive', 0.0)):+.2f} steer={float(row.get('steer', 0.0)):+.2f}"
            ),
            (
                f"insert={float(row.get('insert_depth_m', 0.0)):.3f}m "
                f"pallet_disp={float(row.get('pallet_disp_xy_m', 0.0)):.3f}m "
                f"yaw_err={float(row.get('yaw_err_deg', 0.0)):.1f}deg"
            ),
            (
                f"rig HFOV={args_cli.dual_camera_hfov_deg:.0f} "
                f"Lpos={tuple(float(v) for v in args_cli.dual_camera_left_pos)} "
                f"Lrpy={tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)}"
            ),
        ]
        if not bool(args_cli.metrics_only):
            dual = _concat_dual(left, right)
            dual_labeled = _draw_text_panel(dual, labels)
            topdown = _draw_text_panel(_render_topdown(env, raw_env), labels)
            side_by_side = _make_row(topdown, dual_labeled)

            _save_png(topdown_dir / f"frame_{frame_index:06d}.png", topdown)
            _save_png(dual_dir / f"frame_{frame_index:06d}.png", dual_labeled)
            _save_png(row_dir / f"frame_{frame_index:06d}.png", side_by_side)
        left_red = _red_component_stats(left, int(args_cli.fork_visible_min_area_px))
        right_red = _red_component_stats(right, int(args_cli.fork_visible_min_area_px))
        left_green = _green_component_stats(left, int(args_cli.pallet_visible_min_area_px))
        right_green = _green_component_stats(right, int(args_cli.pallet_visible_min_area_px))
        stats.append(
            {
                "frame_index": int(frame_index),
                "episode_step": int(meta.get("episode_step", -1)),
                "left_red_area_px": int(left_red["total_area_px"]),
                "right_red_area_px": int(right_red["total_area_px"]),
                "left_fork_red": left_red,
                "right_fork_red": right_red,
                "left_pallet_green": left_green,
                "right_pallet_green": right_green,
                "fork_visible_pass": bool(
                    max(int(left_red["largest_area_px"]), int(right_red["largest_area_px"]))
                    >= int(args_cli.fork_visible_min_area_px)
                ),
                "pallet_visible_pass": bool(
                    max(int(left_green["largest_area_px"]), int(right_green["largest_area_px"]))
                    >= int(args_cli.pallet_visible_min_area_px)
                ),
            }
        )

    if bool(args_cli.metrics_only):
        videos_ok = {"topdown": False, "dual_camera": False, "side_by_side": False}
    else:
        videos_ok = {
            "topdown": _make_video(topdown_dir, output_dir / "topdown.mp4", int(args_cli.fps)),
            "dual_camera": _make_video(dual_dir, output_dir / "dual_camera.mp4", int(args_cli.fps)),
            "side_by_side": _make_video(
                row_dir,
                output_dir / "side_by_side_topdown_dual_camera.mp4",
                int(args_cli.fps),
            ),
        }
    summary = {
        "task": str(args_cli.task),
        "source_frame_meta_jsonl": str(args_cli.frame_meta_jsonl),
        "frames": len(records),
        "videos_ok": videos_ok,
        "metrics_only": bool(args_cli.metrics_only),
        "dual_camera": {
            "hfov_deg": float(args_cli.dual_camera_hfov_deg),
            "far_clip_m": float(args_cli.camera_far),
            "left_pos_local": [float(v) for v in args_cli.dual_camera_left_pos],
            "right_pos_local": [float(v) for v in args_cli.dual_camera_right_pos],
            "left_rpy_local_deg": [float(v) for v in args_cli.dual_camera_left_rpy_deg],
            "right_rpy_local_deg": [float(v) for v in args_cli.dual_camera_right_rpy_deg],
        },
        "pallet_highlight": pallet_highlight,
        "pallet_visible_min_area_px": int(args_cli.pallet_visible_min_area_px),
        "fork_visible_min_area_px": int(args_cli.fork_visible_min_area_px),
        "fork_visible_all_frames_pass": bool(all(bool(item["fork_visible_pass"]) for item in stats)),
        "pallet_visible_all_frames_pass": bool(all(bool(item["pallet_visible_pass"]) for item in stats)),
        "first_low_fork": next((item for item in stats if not bool(item["fork_visible_pass"])), None),
        "first_low_pallet": next((item for item in stats if not bool(item["pallet_visible_pass"])), None),
        "red_visibility": {
            "left_low_red_frames_lt50": int(
                sum(1 for item in stats if int(item["left_fork_red"]["largest_area_px"]) < 50)
            ),
            "right_low_red_frames_lt50": int(
                sum(1 for item in stats if int(item["right_fork_red"]["largest_area_px"]) < 50)
            ),
            "first_left_low_red": next(
                (item for item in stats if int(item["left_fork_red"]["largest_area_px"]) < 50), None
            ),
            "first_right_low_red": next(
                (item for item in stats if int(item["right_fork_red"]["largest_area_px"]) < 50), None
            ),
        },
        "frame_stats": stats,
        "videos": {
            "topdown": "topdown.mp4",
            "dual_camera": "dual_camera.mp4",
            "side_by_side": "side_by_side_topdown_dual_camera.mp4",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
