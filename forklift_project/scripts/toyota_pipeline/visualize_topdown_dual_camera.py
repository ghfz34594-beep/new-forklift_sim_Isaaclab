"""Render top-down third-person and forklift-mounted dual camera views.

This is a static visual audit for the Toyota dual-camera task.  It places each
parallel env into a pre-insert pose with a different yaw error, then saves
third-person top-down frames and the corresponding left/right policy-camera
frames for manual inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize top-down and dual-camera forklift views")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=20260604)
parser.add_argument("--env_spacing", type=float, default=20.0)
parser.add_argument("--camera_far", type=float, default=8.0)
parser.add_argument("--gap_m", type=float, default=0.35)
parser.add_argument("--lateral_m", type=float, default=-0.525)
parser.add_argument("--yaw_deg", type=float, nargs="*", default=None)
parser.add_argument("--yaw_min_deg", type=float, default=-14.0)
parser.add_argument("--yaw_max_deg", type=float, default=14.0)
parser.add_argument("--fps", type=int, default=2)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
parser.add_argument("--topdown_camera_eye", type=float, nargs=3, default=(-2.7, 0.0, 7.0))
parser.add_argument("--topdown_camera_lookat", type=float, nargs=3, default=(-2.7, 0.0, 0.0))
parser.add_argument("--topdown_warmup_renders", type=int, default=2)
parser.add_argument("--topdown_render_retries", type=int, default=4)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=100.0)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(150.0, 75.0, 140.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(150.0, -75.0, 140.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, -20.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, 20.0))
parser.add_argument("--vision_room", action="store_true", default=None)
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room")
parser.add_argument(
    "--highlight_pallet",
    action="store_true",
    help="Color pallet green in this audit render so camera visibility is obvious.",
)
parser.add_argument("--pallet_visible_min_area_px", type=int, default=250)
parser.add_argument("--fork_visible_min_area_px", type=int, default=250)
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
            raise ValueError("Pass one env image, not a batch.")
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


def _image_hash(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


def _is_blank_frame(image: np.ndarray) -> bool:
    rgb = image[..., :3].astype(np.float32)
    return bool(float(rgb.mean()) < 5.0 and float(rgb.std()) < 5.0)


def _load_font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label_tile(image: np.ndarray, lines: list[str], width: int | None = None, height: int | None = None) -> np.ndarray:
    pil = Image.fromarray(image)
    if width is not None or height is not None:
        if width is None:
            width = int(round(pil.width * float(height) / float(pil.height)))
        if height is None:
            height = int(round(pil.height * float(width) / float(pil.width)))
        pil = pil.resize((int(width), int(height)), Image.Resampling.LANCZOS)

    panel_h = 48
    canvas = Image.new("RGB", (pil.width, pil.height + panel_h), (20, 20, 20))
    canvas.paste(pil, (0, panel_h))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(13)
    y = 5
    for line in lines[:2]:
        draw.text((8, y), line, fill=(238, 238, 238), font=font)
        y += 18
    return np.asarray(canvas)


def _concat_dual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 4, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _make_row(topdown: np.ndarray, dual: np.ndarray, lines: list[str]) -> np.ndarray:
    top_tile = _label_tile(topdown, ["third-person topdown", lines[0]], width=448)
    dual_tile = _label_tile(dual, ["forklift left/right cameras", lines[1]], width=452)
    h = max(top_tile.shape[0], dual_tile.shape[0])
    w = top_tile.shape[1] + 16 + dual_tile.shape[1]
    row = np.full((h, w, 3), 26, dtype=np.uint8)
    row[: top_tile.shape[0], : top_tile.shape[1]] = top_tile
    x = top_tile.shape[1] + 16
    row[: dual_tile.shape[0], x : x + dual_tile.shape[1]] = dual_tile
    return row


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise RuntimeError("No rows to stack")
    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + 12 * (len(rows) - 1)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    y = 0
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + 12
    return canvas


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


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw.reshape(-1, 1) * 0.5
    return torch.cat([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)


def _yaw_values(num_envs: int) -> list[float]:
    if args_cli.yaw_deg is not None and len(args_cli.yaw_deg) > 0:
        return [float(v) for v in args_cli.yaw_deg]
    if num_envs <= 1:
        return [float(args_cli.yaw_min_deg)]
    values = np.linspace(float(args_cli.yaw_min_deg), float(args_cli.yaw_max_deg), num_envs)
    return [float(v) for v in values.tolist()]


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
    env_cfg.dual_camera_hfov_deg = float(args_cli.dual_camera_hfov_deg)
    env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args_cli.dual_camera_left_pos)
    env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args_cli.dual_camera_right_pos)
    env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)
    env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_right_rpy_deg)


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
    material = _make_preview_material(stage, "/World/AuditPalletVisibilityMaterial", (0.0, 1.0, 0.0), 0.45)
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
        "large_green_components": int(len(components)),
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
        "large_red_components": int(len(components)),
        "largest_area_px": int(components[0]["area_px"]) if components else 0,
        "components": components[:8],
    }


def _apply_preinsert_poses(raw_env: Any, yaws_deg: list[float]) -> dict[str, Any]:
    env_ids = torch.arange(int(raw_env.num_envs), device=raw_env.device, dtype=torch.long)
    n_envs = int(raw_env.num_envs)
    yaw_tensor = torch.tensor(yaws_deg, dtype=torch.float32, device=raw_env.device) * math.pi / 180.0

    pallet_pos_local = torch.tensor(raw_env.cfg.pallet_cfg.init_state.pos, device=raw_env.device).repeat(n_envs, 1)
    pallet_pos = pallet_pos_local + raw_env.scene.env_origins[env_ids]
    pallet_quat = torch.tensor(raw_env.cfg.pallet_cfg.init_state.rot, device=raw_env.device).repeat(n_envs, 1)
    raw_env.pallet.write_root_pose_to_sim(torch.cat([pallet_pos, pallet_quat], dim=1), env_ids=env_ids)
    raw_env.pallet.write_root_velocity_to_sim(torch.zeros((n_envs, 6), device=raw_env.device), env_ids=env_ids)

    pallet_yaw = torch.zeros((n_envs,), device=raw_env.device)
    u_in = torch.stack([torch.cos(pallet_yaw), torch.sin(pallet_yaw)], dim=-1)
    v_lat = torch.stack([-torch.sin(pallet_yaw), torch.cos(pallet_yaw)], dim=-1)
    s_front = -0.5 * float(raw_env.cfg.pallet_depth_m)
    fork_forward = float(getattr(raw_env, "_fork_forward_offset", 1.87))
    gap = torch.full((n_envs,), float(args_cli.gap_m), device=raw_env.device)
    lateral = torch.full((n_envs,), float(args_cli.lateral_m), device=raw_env.device)
    root_axis = s_front - gap - fork_forward
    root_xy = pallet_pos[:, :2] + root_axis.unsqueeze(-1) * u_in + lateral.unsqueeze(-1) * v_lat
    root_z = torch.full((n_envs, 1), 0.03, device=raw_env.device)
    robot_pos = torch.cat([root_xy, root_z], dim=1)
    robot_quat = _quat_from_yaw(pallet_yaw + yaw_tensor)
    raw_env.robot.write_root_pose_to_sim(torch.cat([robot_pos, robot_quat], dim=1), env_ids=env_ids)
    raw_env.robot.write_root_velocity_to_sim(torch.zeros((n_envs, 6), device=raw_env.device), env_ids=env_ids)

    joint_pos = raw_env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    raw_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    if hasattr(raw_env, "_fork_tip_z0"):
        raw_env._fork_tip_z0[env_ids] = root_z.squeeze(-1)
    if hasattr(raw_env, "actions"):
        raw_env.actions[env_ids] = 0.0

    raw_env.scene.write_data_to_sim()
    raw_env.scene.update(dt=0.0)
    if hasattr(raw_env, "_sync_dual_camera_poses"):
        raw_env._sync_dual_camera_poses(env_ids, root_pos=robot_pos, root_quat=robot_quat)
    if raw_env.sim.has_rtx_sensors():
        raw_env.sim.render()
        raw_env.scene.update(dt=0.0)

    return {
        "gap_m": float(args_cli.gap_m),
        "lateral_m": float(args_cli.lateral_m),
        "yaw_deg": [float(v) for v in yaws_deg],
        "root_axis_m": float(root_axis[0].detach().cpu().item()),
        "fork_forward_offset_m": float(fork_forward),
        "pallet_depth_m": float(raw_env.cfg.pallet_depth_m),
    }


def _set_topdown_camera(raw_env: Any, env_id: int) -> None:
    origin = raw_env.scene.env_origins[int(env_id)].detach().cpu().numpy()
    eye = origin + np.asarray(args_cli.topdown_camera_eye, dtype=np.float32)
    lookat = origin + np.asarray(args_cli.topdown_camera_lookat, dtype=np.float32)
    raw_env.sim.set_camera_view(eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in lookat))


def _warmup_topdown_renderer(env: Any, raw_env: Any) -> None:
    warmups = max(0, int(args_cli.topdown_warmup_renders))
    if warmups <= 0:
        return
    _set_topdown_camera(raw_env, 0)
    for _ in range(warmups):
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()
            raw_env.scene.update(dt=0.0)
        env.render()


def _render_topdown(env: Any, raw_env: Any, env_id: int) -> np.ndarray:
    last: np.ndarray | None = None
    retries = max(1, int(args_cli.topdown_render_retries))
    for _ in range(retries):
        _set_topdown_camera(raw_env, env_id)
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()
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


def _write_html(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for frame in summary["frames"]:
        rows.append(
            "<tr>"
            f"<td>{frame['env_id']}</td>"
            f"<td>{frame['yaw_deg']:+.1f}</td>"
            f"<td><a href='{html.escape(frame['topdown_image'])}'>topdown</a></td>"
            f"<td><a href='{html.escape(frame['dual_image'])}'>dual</a></td>"
            f"<td><a href='{html.escape(frame['row_image'])}'>side-by-side</a></td>"
            f"<td>{frame['left_pallet_green']['largest_area_px']}</td>"
            f"<td>{frame['right_pallet_green']['largest_area_px']}</td>"
            f"<td>{frame['left_fork_red']['largest_area_px']}</td>"
            f"<td>{frame['right_fork_red']['largest_area_px']}</td>"
            "</tr>"
        )
    videos = summary["videos"]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Topdown + Dual Camera Visual Check</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, sans-serif; background: #111; color: #eee; }}
    a {{ color: #88c7ff; }}
    img, video {{ max-width: 100%; border: 1px solid #444; background: #000; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    table {{ border-collapse: collapse; margin-top: 16px; width: 100%; }}
    th, td {{ border-bottom: 1px solid #333; padding: 6px 8px; text-align: left; }}
    code {{ background: #222; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Topdown + Dual Camera Visual Check</h1>
  <p>vision_room=<code>{summary['vision_room_enable']}</code>,
     env_spacing=<code>{summary['env_spacing']}</code>,
     camera_far=<code>{summary['dual_camera_config']['camera_far']}</code>,
     pallet_highlight=<code>{summary['pallet_highlight']['enabled']}</code></p>
  <h2>Overview</h2>
  <a href="overview.png"><img src="overview.png" alt="overview"></a>
  <h2>Videos</h2>
  <div class="grid">
    <div><h3>Side by Side</h3><video src="{html.escape(videos['side_by_side'])}" controls loop></video></div>
    <div><h3>Third Person Topdown</h3><video src="{html.escape(videos['topdown'])}" controls loop></video></div>
    <div><h3>Dual Camera</h3><video src="{html.escape(videos['dual'])}" controls loop></video></div>
  </div>
  <h2>Frames</h2>
  <table>
    <thead><tr><th>env</th><th>yaw deg</th><th>topdown</th><th>dual</th><th>row</th><th>left green px</th><th>right green px</th><th>left red px</th><th>right red px</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaws = _yaw_values(int(args_cli.num_envs))
    num_envs = len(yaws)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.viewer.resolution = (int(args_cli.video_width), int(args_cli.video_height))
    env_cfg.scene.filter_collisions = True
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

    pose_summary = _apply_preinsert_poses(raw_env, yaws)
    _warmup_topdown_renderer(env, raw_env)
    left_batch, right_batch = raw_env._get_dual_camera_images()

    topdown_frame_dir = output_dir / "topdown_frames"
    dual_frame_dir = output_dir / "dual_frames"
    row_frame_dir = output_dir / "side_by_side_frames"
    frame_rows: list[dict[str, Any]] = []
    overview_rows: list[np.ndarray] = []
    min_green = int(args_cli.pallet_visible_min_area_px)
    min_red = int(args_cli.fork_visible_min_area_px)

    for env_id, yaw_deg in enumerate(yaws):
        env_dir = output_dir / f"env_{env_id:03d}"
        topdown = _render_topdown(env, raw_env, env_id)
        left = _to_uint8_hwc(left_batch[env_id])
        right = _to_uint8_hwc(right_batch[env_id])
        dual = _concat_dual(left, right)
        lines = [
            f"env={env_id:03d} yaw={yaw_deg:+.1f} deg gap={args_cli.gap_m:.2f} m lat={args_cli.lateral_m:+.2f} m",
            f"HFOV={args_cli.dual_camera_hfov_deg:.0f} far={args_cli.camera_far:.1f}m room={bool(getattr(raw_env.cfg, 'vision_room_enable', False))}",
        ]
        top_labeled = _label_tile(topdown, ["third-person topdown", lines[0]], width=448)
        dual_labeled = _label_tile(dual, ["forklift left/right cameras", lines[1]], width=452)
        row = _make_row(topdown, dual, lines)

        _save_png(env_dir / "topdown.png", topdown)
        _save_png(env_dir / "left.png", left)
        _save_png(env_dir / "right.png", right)
        _save_png(env_dir / "dual_camera.png", dual)
        _save_png(env_dir / "side_by_side.png", row)
        _save_png(topdown_frame_dir / f"frame_{env_id:06d}.png", top_labeled)
        _save_png(dual_frame_dir / f"frame_{env_id:06d}.png", dual_labeled)
        _save_png(row_frame_dir / f"frame_{env_id:06d}.png", row)
        overview_rows.append(row)

        left_green = _green_component_stats(left, min_green)
        right_green = _green_component_stats(right, min_green)
        left_red = _red_component_stats(left, min_red)
        right_red = _red_component_stats(right, min_red)
        frame_rows.append(
            {
                "env_id": int(env_id),
                "yaw_deg": float(yaw_deg),
                "topdown_image": str(Path(f"env_{env_id:03d}") / "topdown.png"),
                "left_image": str(Path(f"env_{env_id:03d}") / "left.png"),
                "right_image": str(Path(f"env_{env_id:03d}") / "right.png"),
                "dual_image": str(Path(f"env_{env_id:03d}") / "dual_camera.png"),
                "row_image": str(Path(f"env_{env_id:03d}") / "side_by_side.png"),
                "topdown_sha1": _image_hash(topdown),
                "topdown_blank": _is_blank_frame(topdown),
                "left_sha1": _image_hash(left),
                "right_sha1": _image_hash(right),
                "left_pallet_green": left_green,
                "right_pallet_green": right_green,
                "left_fork_red": left_red,
                "right_fork_red": right_red,
                "pallet_visible_pass": bool(
                    max(int(left_green["largest_area_px"]), int(right_green["largest_area_px"])) >= min_green
                ),
                "fork_visible_pass": bool(
                    max(int(left_red["largest_area_px"]), int(right_red["largest_area_px"])) >= min_red
                ),
            }
        )

    overview = _stack_rows(overview_rows)
    _save_png(output_dir / "overview.png", overview)
    videos_ok = {
        "topdown": _make_video(topdown_frame_dir, output_dir / "topdown.mp4", int(args_cli.fps)),
        "dual": _make_video(dual_frame_dir, output_dir / "dual_camera.mp4", int(args_cli.fps)),
        "side_by_side": _make_video(row_frame_dir, output_dir / "side_by_side.mp4", int(args_cli.fps)),
    }

    far_ratio = float(getattr(raw_env.cfg, "visual_isolation_far_clip_env_spacing_ratio", 0.45))
    summary = {
        "task": str(args_cli.task),
        "num_envs": int(raw_env.num_envs),
        "env_spacing": float(raw_env.cfg.scene.env_spacing),
        "seed": int(args_cli.seed),
        "vision_room_enable": bool(getattr(raw_env.cfg, "vision_room_enable", False)),
        "visual_isolation": {
            "mode": "room" if bool(getattr(raw_env.cfg, "vision_room_enable", False)) else "far_clip_spacing",
            "far_clip_env_spacing_ratio": float(far_ratio),
            "far_clip_pass": bool(float(args_cli.camera_far) <= float(raw_env.cfg.scene.env_spacing) * far_ratio),
        },
        "topdown_camera": {
            "eye_local": [float(v) for v in args_cli.topdown_camera_eye],
            "lookat_local": [float(v) for v in args_cli.topdown_camera_lookat],
            "resolution": [int(args_cli.video_width), int(args_cli.video_height)],
        },
        "dual_camera_config": {
            "hfov_deg": float(raw_env.cfg.dual_camera_hfov_deg),
            "left_pos_local": [float(v) for v in raw_env.cfg.dual_camera_left_pos_local],
            "right_pos_local": [float(v) for v in raw_env.cfg.dual_camera_right_pos_local],
            "left_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_left_rpy_local_deg],
            "right_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_right_rpy_local_deg],
            "camera_far": float(getattr(raw_env.cfg, "dual_camera_far_clip_m", -1.0)),
        },
        "pose_summary": pose_summary,
        "pallet_highlight": pallet_highlight,
        "pallet_visible_min_area_px": int(min_green),
        "pallet_visible_all_frames_pass": bool(all(bool(row["pallet_visible_pass"]) for row in frame_rows)),
        "fork_visible_min_area_px": int(min_red),
        "fork_visible_all_frames_pass": bool(all(bool(row["fork_visible_pass"]) for row in frame_rows)),
        "topdown_all_frames_nonblank": bool(not any(bool(row["topdown_blank"]) for row in frame_rows)),
        "frames": frame_rows,
        "videos_ok": videos_ok,
        "videos": {
            "topdown": "topdown.mp4",
            "dual": "dual_camera.mp4",
            "side_by_side": "side_by_side.mp4",
        },
        "overview_image": "overview.png",
        "index_html": "index.html",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_html(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
