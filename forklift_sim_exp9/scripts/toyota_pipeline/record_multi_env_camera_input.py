"""Record one env's dual-camera policy input while the task runs with many envs.

This is a visual isolation check: launch the normal IsaacLab task with
``num_envs > 1`` and save the left/right camera observations from one selected
environment.  The videos are meant for manual inspection of cross-env visual
contamination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record multi-env dual-camera input for one IsaacLab env")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--warmup_steps", type=int, default=4)
parser.add_argument("--record_every", type=int, default=2)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--drive", type=float, default=0.0, help="Constant drive command applied to all envs.")
parser.add_argument("--steer", type=float, default=0.0, help="Constant steer command applied to all envs.")
parser.add_argument("--lift", type=float, default=0.0, help="Constant lift command for 3D-action tasks.")
parser.add_argument("--seed", type=int, default=20260521)
parser.add_argument("--env_spacing", type=float, default=None, help="Optional override for scene.env_spacing.")
parser.add_argument("--camera_far", type=float, default=None, help="Optional override for dual-camera far clipping range.")
parser.add_argument("--dual_camera_hfov_deg", type=float, default=None, help="Override dual-camera horizontal FoV.")
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"))
parser.add_argument("--red_component_min_area_px", type=int, default=250)
parser.add_argument("--red_component_gate", type=int, default=1)
parser.add_argument("--max_second_red_area_px", type=int, default=250)
parser.add_argument("--min_fork_red_area_px", type=int, default=250)
parser.add_argument("--max_red_area_fraction", type=float, default=0.20)
parser.add_argument(
    "--pallet_visibility_audit",
    action="store_true",
    help="Temporarily color pallets green and require pallet pixels in the rendered policy cameras.",
)
parser.add_argument("--pallet_visible_min_area_px", type=int, default=250)
parser.add_argument("--pallet_confident_min_fraction", type=float, default=0.015)
parser.add_argument("--sentinel_audit", action="store_true", help="Spawn magenta sentinels in non-target envs.")
parser.add_argument("--sentinel_foreign_envs", action="store_true", default=True)
parser.add_argument("--no_sentinel_foreign_envs", action="store_false", dest="sentinel_foreign_envs")
parser.add_argument(
    "--sentinel_exclude_env_ids",
    type=int,
    nargs="*",
    default=None,
    help=(
        "Env ids that must not receive in-room foreign sentinels. Defaults to "
        "the target env, or the mosaic env ids when recording a mosaic."
    ),
)
parser.add_argument(
    "--sentinel_room_probes_all_envs",
    action="store_true",
    help="Spawn magenta room-boundary probes outside every env room for mosaic leakage checks.",
)
parser.add_argument("--sentinel_min_area_px", type=int, default=64)
parser.add_argument("--vision_room", action="store_true", default=None, help="Force per-env room occlusion on.")
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room", help="Force per-env room occlusion off.")
parser.add_argument("--audit_pose", choices=("reset", "preinsert"), default="reset")
parser.add_argument("--audit_preinsert_gap_m", type=float, default=0.35)
parser.add_argument(
    "--preinsert_pose_sweep",
    action="store_true",
    help="For preinsert audits, vary yaw/lateral/gap per env so the mosaic proves per-env independence.",
)
parser.add_argument(
    "--preinsert_sweep_yaw_min_deg",
    type=float,
    default=-14.0,
    help="Minimum yaw error used by --preinsert_pose_sweep. Defaults to the V5 reset yaw range.",
)
parser.add_argument(
    "--preinsert_sweep_yaw_max_deg",
    type=float,
    default=14.0,
    help="Maximum yaw error used by --preinsert_pose_sweep. Defaults to the V5 reset yaw range.",
)
parser.add_argument("--record_mosaic", action="store_true", help="Record a tiled all-env batch view.")
parser.add_argument("--mosaic_max_envs", type=int, default=16)
parser.add_argument("--mosaic_cols", type=int, default=4)
parser.add_argument(
    "--mosaic_start_env",
    type=int,
    default=0,
    help="First env id included in mosaic stats when --mosaic_env_ids is not provided.",
)
parser.add_argument(
    "--mosaic_env_stride",
    type=int,
    default=1,
    help="Stride between env ids included in mosaic stats when --mosaic_env_ids is not provided.",
)
parser.add_argument(
    "--mosaic_env_ids",
    type=int,
    nargs="*",
    default=None,
    help="Explicit env ids included in mosaic stats/video. Overrides --mosaic_start_env/--mosaic_env_stride.",
)
parser.add_argument(
    "--mosaic_save_frames",
    action="store_true",
    default=True,
    help="Save tiled mosaic PNG/MP4 outputs. Disable for large N stats-only isolation checks.",
)
parser.add_argument("--no_mosaic_save_frames", action="store_false", dest="mosaic_save_frames")
parser.add_argument("--overlay", action="store_true", help="Add a small non-occluding label panel above frames.")
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401


def _to_uint8_hwc(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            raise ValueError("Pass a single env image, not a batched image.")
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


def _draw_overlay(image: np.ndarray, lines: list[str]) -> np.ndarray:
    panel_h = 46
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + panel_h), (18, 18, 18))
    canvas.paste(Image.fromarray(image), (0, panel_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    y = 5
    for line in lines[:2]:
        draw.text((8, y), line, fill=(235, 235, 235), font=font)
        y += 18
    return np.asarray(canvas)


def _tile_with_label(image: np.ndarray, label: str) -> np.ndarray:
    label_h = 18
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + label_h), (15, 15, 15))
    canvas.paste(Image.fromarray(image), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 2), label, fill=(235, 235, 235), font=font)
    return np.asarray(canvas)


def _make_mosaic(images: list[np.ndarray], labels: list[str], cols: int) -> np.ndarray:
    if not images:
        raise ValueError("mosaic requires at least one image")
    cols = max(1, int(cols))
    tiles = [_tile_with_label(image, label) for image, label in zip(images, labels)]
    rows = int(np.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return canvas


def _concat_dual_with_divider(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider_w = 4
    divider = np.full((left.shape[0], divider_w, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _image_hash(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


def _green_component_stats(image: np.ndarray, min_area: int) -> dict[str, Any]:
    """Detect audit-only green pallet pixels."""
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
        "total_area_px": total_area,
        "large_green_components": len(components),
        "largest_area_px": int(components[0]["area_px"]) if components else 0,
        "components": components[:8],
    }


def _red_component_stats(image: np.ndarray, min_area: int) -> dict[str, Any]:
    """Return simple red connected-component stats for fork ambiguity checks.

    This deliberately avoids cv2/scipy dependencies because the script runs in
    the IsaacLab environment.  It is only an auxiliary signal; manual video
    review remains the acceptance gate.
    """
    rgb = image[..., :3].astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (
        (red >= 55)
        & (red >= green + 20)
        & (red >= blue + 20)
        & (red * 4 >= np.maximum(green, blue) * 5)
    )
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []

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
        "large_red_components": len(components),
        "largest_area_px": int(components[0]["area_px"]) if components else 0,
        "second_largest_area_px": int(components[1]["area_px"]) if len(components) > 1 else 0,
        "components": components[:8],
    }


def _magenta_component_stats(image: np.ndarray, min_area: int) -> dict[str, Any]:
    """Detect audit-only magenta sentinels from foreign envs."""
    rgb = image[..., :3].astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (red >= 150) & (blue >= 150) & (green <= 100) & (red >= green + 60) & (blue >= green + 60)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []

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
        "large_magenta_components": len(components),
        "largest_area_px": int(components[0]["area_px"]) if components else 0,
        "components": components[:8],
    }


def _make_preview_material(stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float = 0.65):
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(path).AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
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


def _apply_pallet_visibility_material(raw_env) -> dict[str, Any]:
    stage = raw_env.sim.stage
    material = _make_preview_material(stage, "/World/AuditPalletVisibilityMaterial", (0.0, 1.0, 0.0), 0.45)
    bound_envs: list[int] = []
    bound_prims = 0
    for env_id in range(int(raw_env.num_envs)):
        prim_path = f"/World/envs/env_{env_id}/Pallet"
        prim = stage.GetPrimAtPath(prim_path)
        count = _bind_material_recursive(prim, material)
        if count > 0:
            bound_envs.append(int(env_id))
            bound_prims += int(count)
    if raw_env.sim.has_rtx_sensors():
        raw_env.sim.render()
    return {
        "enabled": True,
        "color": [0.0, 1.0, 0.0],
        "bound_env_count": int(len(bound_envs)),
        "bound_prim_count": int(bound_prims),
        "sample_env_ids": bound_envs[:16],
        "pass": len(bound_envs) == int(raw_env.num_envs),
    }


def _spawn_foreign_env_sentinels(
    raw_env,
    target_env_id: int,
    exclude_env_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Place high-saturation markers outside the target room for leakage checks.

    The markers live under a global /World path, not under /World/envs/env_N.
    With inherited IsaacLab clones, adding audit prims below env_0 after cloning
    can mirror them into cloned envs and create false positives.
    """
    stage = raw_env.sim.stage
    material = _make_preview_material(stage, "/World/AuditForeignSentinelMaterial", (1.0, 0.0, 1.0), 0.6)

    parent_path = "/World/AuditForeignSentinels"
    UsdGeom.Xform.Define(stage, parent_path)
    records: list[dict[str, Any]] = []

    def spawn_cube(name: str, translation_w: tuple[float, float, float], size: float, kind: str, env_id: int) -> None:
        prim_path = f"{parent_path}/{name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            cube = UsdGeom.Cube.Define(stage, prim_path)
        else:
            cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(float(size))
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation_w))
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
        records.append(
            {
                "kind": kind,
                "env_id": int(env_id),
                "prim_path": prim_path,
                "translation_w": [float(v) for v in translation_w],
                "size_m": float(size),
            }
        )

    origins = raw_env.scene.env_origins.detach().cpu().numpy()
    exclude_ids = {int(target_env_id)}
    if exclude_env_ids is not None:
        exclude_ids.update(int(env_id) for env_id in exclude_env_ids)
    if bool(args_cli.sentinel_foreign_envs):
        for env_id in range(int(raw_env.num_envs)):
            if env_id in exclude_ids:
                continue
            origin = origins[env_id]
            # Place it near the pallet/front work volume of each foreign env. If
            # walls/FOV/clip are wrong, this color is intentionally easy to detect.
            translation = (float(origin[0] - 1.8), float(origin[1]), 0.7)
            spawn_cube(f"ForeignEnv{env_id:03d}", translation, 0.9, "foreign_env", env_id)

    if bool(getattr(raw_env.cfg, "vision_room_enable", False)):
        length = float(getattr(raw_env.cfg, "vision_room_length_m", 10.0))
        width = float(getattr(raw_env.cfg, "vision_room_width_m", 8.0))
        thickness = float(getattr(raw_env.cfg, "vision_room_wall_thickness_m", 0.15))
        cx = float(getattr(raw_env.cfg, "vision_room_center_x_m", -1.5))
        cy = float(getattr(raw_env.cfg, "vision_room_center_y_m", 0.0))
        offset = thickness + 0.35
        probes = {
            "RoomProbePosX": (cx + 0.5 * length + offset, cy, 0.7),
            "RoomProbeNegX": (cx - 0.5 * length - offset, cy, 0.7),
            "RoomProbePosY": (cx, cy + 0.5 * width + offset, 0.7),
            "RoomProbeNegY": (cx, cy - 0.5 * width - offset, 0.7),
        }
        room_probe_env_ids = range(int(raw_env.num_envs)) if bool(args_cli.sentinel_room_probes_all_envs) else [int(target_env_id)]
        for probe_env_id in room_probe_env_ids:
            origin = origins[int(probe_env_id)]
            kind = "room_probe_all_envs" if bool(args_cli.sentinel_room_probes_all_envs) else "target_room_probe"
            for name, local in probes.items():
                translation = (
                    float(origin[0] + local[0]),
                    float(origin[1] + local[1]),
                    float(origin[2] + local[2]),
                )
                spawn_cube(f"Env{int(probe_env_id):03d}_{name}", translation, 0.6, kind, probe_env_id)

    return records


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw.reshape(-1, 1) * 0.5
    return torch.cat([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)


def _apply_preinsert_audit_pose(raw_env, gap_m: float) -> dict[str, Any]:
    """Put envs into non-inserted poses for visual auditing."""
    env_ids = torch.arange(int(raw_env.num_envs), device=raw_env.device, dtype=torch.long)
    n_envs = int(raw_env.num_envs)

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
    gap = torch.full((n_envs,), float(gap_m), device=raw_env.device)
    lateral = torch.zeros((n_envs,), device=raw_env.device)
    yaw_error = torch.zeros((n_envs,), device=raw_env.device)
    if bool(args_cli.preinsert_pose_sweep):
        idx = torch.arange(n_envs, device=raw_env.device, dtype=torch.float32)
        col = torch.remainder(idx, 8.0)
        row = torch.remainder(torch.floor(idx / 8.0), 8.0)
        yaw_min_deg = float(args_cli.preinsert_sweep_yaw_min_deg)
        yaw_max_deg = float(args_cli.preinsert_sweep_yaw_max_deg)
        yaw_step_deg = (yaw_max_deg - yaw_min_deg) / 7.0
        yaw_error = (yaw_min_deg + col * yaw_step_deg) * torch.pi / 180.0
        gap = torch.full_like(row, float(gap_m)) + row * 0.20
        lateral = (torch.remainder(row, 4.0) - 1.5) * 0.35
    root_axis = s_front - gap - fork_forward
    root_xy = pallet_pos[:, :2] + root_axis.unsqueeze(-1) * u_in + lateral.unsqueeze(-1) * v_lat
    root_z = torch.full((n_envs, 1), 0.03, device=raw_env.device)
    robot_pos = torch.cat([root_xy, root_z], dim=1)
    robot_quat = _quat_from_yaw(pallet_yaw + yaw_error)
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

    sample_count = min(n_envs, 16)
    return {
        "mode": "preinsert",
        "pose_sweep": bool(args_cli.preinsert_pose_sweep),
        "gap_m": float(gap[0].detach().cpu().item()),
        "gap_range_m": [
            float(gap.min().detach().cpu().item()),
            float(gap.max().detach().cpu().item()),
        ],
        "lateral_range_m": [
            float(lateral.min().detach().cpu().item()),
            float(lateral.max().detach().cpu().item()),
        ],
        "yaw_range_deg": [
            float((yaw_error.min() * 180.0 / torch.pi).detach().cpu().item()),
            float((yaw_error.max() * 180.0 / torch.pi).detach().cpu().item()),
        ],
        "yaw_error_range_deg": [
            float((yaw_error.min() * 180.0 / torch.pi).detach().cpu().item()),
            float((yaw_error.max() * 180.0 / torch.pi).detach().cpu().item()),
        ],
        "root_axis_m": float(root_axis[0].detach().cpu().item()),
        "root_axis_range_m": [
            float(root_axis.min().detach().cpu().item()),
            float(root_axis.max().detach().cpu().item()),
        ],
        "fork_forward_offset_m": fork_forward,
        "pallet_depth_m": float(raw_env.cfg.pallet_depth_m),
        "pallet_quat_wxyz_sample": [float(v) for v in pallet_quat[0].detach().cpu().tolist()],
        "pose_samples": [
            {
                "env_id": int(i),
                "gap_m": float(gap[i].detach().cpu().item()),
                "lateral_m": float(lateral[i].detach().cpu().item()),
                "yaw_error_deg": float((yaw_error[i] * 180.0 / torch.pi).detach().cpu().item()),
            }
            for i in range(sample_count)
        ],
    }


def _geometry_audit(raw_env) -> dict[str, Any]:
    pallet_pos = raw_env.pallet.data.root_pos_w
    q = raw_env.pallet.data.root_quat_w
    pallet_yaw = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )
    tip = raw_env._compute_fork_tip()
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)
    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    tip_y_err = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
    s_front = -0.5 * float(raw_env.cfg.pallet_depth_m)
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    lift_height = tip[:, 2] - raw_env._fork_tip_z0
    pallet_lift_height = pallet_pos[:, 2] - float(raw_env.cfg.pallet_cfg.init_state.pos[2])
    z_err = torch.abs(lift_height - pallet_lift_height)
    return {
        "dist_front_min_m": float(dist_front.min().detach().cpu().item()),
        "dist_front_max_m": float(dist_front.max().detach().cpu().item()),
        "insert_depth_max_m": float(insert_depth.max().detach().cpu().item()),
        "tip_lateral_err_max_m": float(tip_y_err.max().detach().cpu().item()),
        "tip_z_err_max_m": float(z_err.max().detach().cpu().item()),
        "preinsert_pass": bool(float(insert_depth.max().detach().cpu().item()) <= 0.02),
    }


def _room_prim_check(raw_env) -> dict[str, Any]:
    """Check that cloned per-env room walls exist in the live USD stage."""
    stage = getattr(raw_env.sim, "stage", None)
    if stage is None:
        return {"pass": False, "error": "raw_env.sim.stage is unavailable", "missing": []}

    wall_names = ["WallPosX", "WallNegX", "WallPosY", "WallNegY"]
    if bool(getattr(raw_env.cfg, "vision_room_ceiling_enable", True)):
        wall_names.append("Ceiling")
    missing: list[str] = []
    for env_id in range(int(raw_env.num_envs)):
        for wall_name in wall_names:
            prim_path = f"/World/envs/env_{env_id}/Room/{wall_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                missing.append(prim_path)
    return {
        "pass": len(missing) == 0,
        "checked_envs": int(raw_env.num_envs),
        "checked_walls_per_env": len(wall_names),
        "missing": missing[:32],
        "missing_count": len(missing),
    }


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


def _selected_mosaic_env_ids(num_envs: int) -> list[int]:
    if not bool(args_cli.record_mosaic):
        return []
    if args_cli.mosaic_env_ids is not None and len(args_cli.mosaic_env_ids) > 0:
        env_ids = sorted({int(env_id) for env_id in args_cli.mosaic_env_ids})
    else:
        start = max(0, int(args_cli.mosaic_start_env))
        stride = max(1, int(args_cli.mosaic_env_stride))
        max_count = max(0, int(args_cli.mosaic_max_envs))
        env_ids = []
        env_id = start
        while env_id < int(num_envs) and len(env_ids) < max_count:
            env_ids.append(env_id)
            env_id += stride
    bad = [env_id for env_id in env_ids if env_id < 0 or env_id >= int(num_envs)]
    if bad:
        raise ValueError(f"mosaic env ids outside [0, {int(num_envs)}): {bad[:8]}")
    return env_ids


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


def _camera_signature(raw_env: Any, camera_far: float | None) -> dict[str, Any]:
    cfg = raw_env.cfg
    signature = {
        "camera_version": str(getattr(cfg, "camera_version", "unspecified")),
        "hfov_deg": float(getattr(cfg, "dual_camera_hfov_deg", 0.0)),
        "far_clip_m": float(camera_far) if camera_far is not None else None,
        "left_pos_local": [float(v) for v in getattr(cfg, "dual_camera_left_pos_local", ())],
        "right_pos_local": [float(v) for v in getattr(cfg, "dual_camera_right_pos_local", ())],
        "left_rpy_local_deg": [float(v) for v in getattr(cfg, "dual_camera_left_rpy_local_deg", ())],
        "right_rpy_local_deg": [float(v) for v in getattr(cfg, "dual_camera_right_rpy_local_deg", ())],
        "vision_room_enable": bool(getattr(cfg, "vision_room_enable", False)),
    }
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    signature["config_hash_sha1"] = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return signature


def main() -> None:
    if int(args_cli.num_envs) < 1:
        raise ValueError("--num_envs must be >= 1")
    if int(args_cli.env_id) < 0 or int(args_cli.env_id) >= int(args_cli.num_envs):
        raise ValueError("--env_id must be in [0, num_envs)")

    output_dir = Path(args_cli.output_dir)
    left_dir = output_dir / f"env_{int(args_cli.env_id):03d}" / "left_frames"
    right_dir = output_dir / f"env_{int(args_cli.env_id):03d}" / "right_frames"
    dual_dir = output_dir / f"env_{int(args_cli.env_id):03d}" / "dual_frames"
    mosaic_left_dir = output_dir / "mosaic_left_frames"
    mosaic_right_dir = output_dir / "mosaic_right_frames"
    mosaic_dual_dir = output_dir / "mosaic_dual_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    obs, _ = env.reset()
    room_prim_check = _room_prim_check(raw_env)
    selected_mosaic_env_ids = _selected_mosaic_env_ids(int(raw_env.num_envs))
    pallet_visibility_material: dict[str, Any] = {"enabled": False, "pass": True}
    if bool(args_cli.pallet_visibility_audit):
        pallet_visibility_material = _apply_pallet_visibility_material(raw_env)
    audit_pose_summary: dict[str, Any] = {"mode": str(args_cli.audit_pose)}
    if args_cli.audit_pose == "preinsert":
        audit_pose_summary = _apply_preinsert_audit_pose(raw_env, float(args_cli.audit_preinsert_gap_m))
    sentinel_records: list[dict[str, Any]] = []
    if bool(args_cli.sentinel_audit):
        sentinel_exclude_env_ids = args_cli.sentinel_exclude_env_ids
        if sentinel_exclude_env_ids is None:
            sentinel_exclude_env_ids = selected_mosaic_env_ids if selected_mosaic_env_ids else [int(args_cli.env_id)]
        sentinel_records = _spawn_foreign_env_sentinels(
            raw_env,
            int(args_cli.env_id),
            [int(env_id) for env_id in sentinel_exclude_env_ids],
        )
        if raw_env.sim.has_rtx_sensors():
            raw_env.sim.render()

    action_dim = int(getattr(raw_env.cfg, "action_space", 2))
    base_action = torch.zeros((raw_env.num_envs, action_dim), dtype=torch.float32, device=raw_env.device)
    base_action[:, 0] = float(args_cli.drive)
    if action_dim >= 2:
        base_action[:, 1] = float(args_cli.steer)
    if action_dim >= 3:
        base_action[:, 2] = float(args_cli.lift)
    base_action = torch.clamp(base_action, -1.0, 1.0)

    if args_cli.audit_pose == "reset":
        for _ in range(max(0, int(args_cli.warmup_steps))):
            obs, _, _, _, _ = env.step(base_action)
    elif raw_env.sim.has_rtx_sensors():
        raw_env.sim.render()

    saved_frames = 0
    env_id = int(args_cli.env_id)
    red_rows: list[dict[str, Any]] = []
    red_max: dict[str, dict[str, Any]] = {
        "left": {"large_red_components": 0, "largest_area_px": 0, "second_largest_area_px": 0},
        "right": {"large_red_components": 0, "largest_area_px": 0, "second_largest_area_px": 0},
    }
    sentinel_max: dict[str, dict[str, Any]] = {
        "left": {"large_magenta_components": 0, "largest_area_px": 0},
        "right": {"large_magenta_components": 0, "largest_area_px": 0},
    }
    mosaic_sentinel_max: dict[str, dict[str, Any]] = {
        "left": {"envs_with_magenta": 0, "max_components": 0, "largest_area_px": 0},
        "right": {"envs_with_magenta": 0, "max_components": 0, "largest_area_px": 0},
    }
    mosaic_red_visibility: dict[str, Any] = {
        "checked_envs": 0,
        "min_visible_envs": 0,
        "max_oversized_envs": 0,
        "max_largest_red_area_px": 0,
        "max_red_area_fraction": 0.0,
    }
    mosaic_pallet_visibility: dict[str, Any] = {
        "enabled": bool(args_cli.pallet_visibility_audit),
        "checked_envs": 0,
        "min_visible_envs": 0,
        "min_confident_visible_envs": 0,
        "min_pallet_area_px": None,
        "max_pallet_area_px": 0,
        "min_pallet_fraction": None,
        "max_pallet_fraction": 0.0,
        "hard_corner_envs": [],
        "env_records": [],
        "hard_corner_min_area_px": None,
        "hard_corner_min_fraction": None,
        "hard_corner_pass": True,
    }
    mosaic_hash_counts: Counter[str] = Counter()
    mosaic_hash_sample: dict[str, dict[str, Any]] = {}
    mosaic_hash_min_unique_per_frame: int | None = None
    mosaic_hash_max_duplicate_per_frame = 0
    for step in range(int(args_cli.steps)):
        if args_cli.audit_pose == "reset":
            obs, _, terminated, truncated, _ = env.step(base_action)
        else:
            terminated = torch.zeros((raw_env.num_envs,), dtype=torch.bool, device=raw_env.device)
            truncated = torch.zeros((raw_env.num_envs,), dtype=torch.bool, device=raw_env.device)
            if raw_env.sim.has_rtx_sensors():
                raw_env.sim.render()
            raw_env.scene.update(dt=0.0)
        if step % max(1, int(args_cli.record_every)) != 0:
            continue

        raw_obs = raw_env._get_observations()
        left_batch = raw_obs["image_left"]
        right_batch = raw_obs["image_right"]
        if args_cli.record_mosaic:
            mosaic_left_images: list[np.ndarray] = []
            mosaic_right_images: list[np.ndarray] = []
            mosaic_dual_images: list[np.ndarray] = []
            mosaic_labels_left: list[str] = []
            mosaic_labels_right: list[str] = []
            mosaic_labels_dual: list[str] = []
            frame_left_envs_with_magenta = 0
            frame_right_envs_with_magenta = 0
            frame_left_max_components = 0
            frame_right_max_components = 0
            frame_left_largest_area = 0
            frame_right_largest_area = 0
            frame_visible_envs = 0
            frame_oversized_envs = 0
            frame_max_red_area = 0
            frame_max_red_fraction = 0.0
            frame_pallet_visible_envs = 0
            frame_pallet_confident_envs = 0
            frame_pallet_min_area: int | None = None
            frame_pallet_max_area = 0
            frame_pallet_min_fraction: float | None = None
            frame_pallet_max_fraction = 0.0
            frame_hard_corner_records: list[dict[str, Any]] = []
            frame_pallet_env_records: list[dict[str, Any]] = []
            frame_area = max(1, int(raw_env.cfg.dual_camera_width) * int(raw_env.cfg.dual_camera_height))
            frame_hash_counts: Counter[str] = Counter()
            for batch_env_id in selected_mosaic_env_ids:
                batch_left = _to_uint8_hwc(left_batch[batch_env_id])
                batch_right = _to_uint8_hwc(right_batch[batch_env_id])
                batch_left_red = _red_component_stats(batch_left, int(args_cli.red_component_min_area_px))
                batch_right_red = _red_component_stats(batch_right, int(args_cli.red_component_min_area_px))
                batch_left_pallet = _green_component_stats(batch_left, int(args_cli.pallet_visible_min_area_px))
                batch_right_pallet = _green_component_stats(batch_right, int(args_cli.pallet_visible_min_area_px))
                batch_left_sentinel = _magenta_component_stats(batch_left, int(args_cli.sentinel_min_area_px))
                batch_right_sentinel = _magenta_component_stats(batch_right, int(args_cli.sentinel_min_area_px))
                batch_dual = _concat_dual_with_divider(batch_left, batch_right)
                dual_hash = _image_hash(batch_dual)
                mosaic_hash_counts[dual_hash] += 1
                frame_hash_counts[dual_hash] += 1
                mosaic_hash_sample.setdefault(
                    dual_hash,
                    {
                        "env_id": int(batch_env_id),
                        "frame": int(saved_frames),
                    },
                )
                env_largest_red = max(int(batch_left_red["largest_area_px"]), int(batch_right_red["largest_area_px"]))
                env_red_fraction = env_largest_red / frame_area
                frame_max_red_area = max(frame_max_red_area, env_largest_red)
                frame_max_red_fraction = max(frame_max_red_fraction, env_red_fraction)
                if env_largest_red >= int(args_cli.min_fork_red_area_px):
                    frame_visible_envs += 1
                if env_red_fraction > float(args_cli.max_red_area_fraction):
                    frame_oversized_envs += 1
                if bool(args_cli.pallet_visibility_audit):
                    env_pallet_area = max(
                        int(batch_left_pallet["total_area_px"]),
                        int(batch_right_pallet["total_area_px"]),
                    )
                    env_pallet_fraction = env_pallet_area / frame_area
                    frame_pallet_max_area = max(frame_pallet_max_area, env_pallet_area)
                    frame_pallet_max_fraction = max(frame_pallet_max_fraction, env_pallet_fraction)
                    frame_pallet_min_area = (
                        env_pallet_area
                        if frame_pallet_min_area is None
                        else min(frame_pallet_min_area, env_pallet_area)
                    )
                    frame_pallet_min_fraction = (
                        env_pallet_fraction
                        if frame_pallet_min_fraction is None
                        else min(frame_pallet_min_fraction, env_pallet_fraction)
                    )
                    if env_pallet_area >= int(args_cli.pallet_visible_min_area_px):
                        frame_pallet_visible_envs += 1
                    if env_pallet_fraction >= float(args_cli.pallet_confident_min_fraction):
                        frame_pallet_confident_envs += 1
                    pallet_record = {
                        "env_id": int(batch_env_id),
                        "pallet_area_px": int(env_pallet_area),
                        "pallet_fraction": float(env_pallet_fraction),
                        "visible_pass": bool(env_pallet_area >= int(args_cli.pallet_visible_min_area_px)),
                        "confident_pass": bool(
                            env_pallet_fraction >= float(args_cli.pallet_confident_min_fraction)
                        ),
                    }
                    if bool(args_cli.preinsert_pose_sweep):
                        row = int(batch_env_id) // 8
                        col = int(batch_env_id) % 8
                        yaw_min_deg = float(args_cli.preinsert_sweep_yaw_min_deg)
                        yaw_max_deg = float(args_cli.preinsert_sweep_yaw_max_deg)
                        yaw_step_deg = (yaw_max_deg - yaw_min_deg) / 7.0
                        pallet_record.update(
                            {
                                "row": int(row),
                                "col": int(col),
                                "yaw_error_deg": float(yaw_min_deg + col * yaw_step_deg),
                                "gap_m": float(args_cli.audit_preinsert_gap_m + row * 0.20),
                                "lateral_m": float((row % 4 - 1.5) * 0.35),
                            }
                        )
                        if row == 7 and col in (0, 7):
                            frame_hard_corner_records.append(
                                {
                                    "env_id": int(batch_env_id),
                                    "yaw_error_deg": float(yaw_min_deg + col * yaw_step_deg),
                                    "gap_m": float(args_cli.audit_preinsert_gap_m + row * 0.20),
                                    "pallet_area_px": int(env_pallet_area),
                                    "pallet_fraction": float(env_pallet_fraction),
                                    "visible_pass": bool(env_pallet_area >= int(args_cli.pallet_visible_min_area_px)),
                                    "confident_pass": bool(
                                        env_pallet_fraction >= float(args_cli.pallet_confident_min_fraction)
                                    ),
                                }
                            )
                    frame_pallet_env_records.append(pallet_record)
                if int(batch_left_sentinel["large_magenta_components"]) > 0:
                    frame_left_envs_with_magenta += 1
                if int(batch_right_sentinel["large_magenta_components"]) > 0:
                    frame_right_envs_with_magenta += 1
                frame_left_max_components = max(frame_left_max_components, int(batch_left_sentinel["large_magenta_components"]))
                frame_right_max_components = max(frame_right_max_components, int(batch_right_sentinel["large_magenta_components"]))
                frame_left_largest_area = max(frame_left_largest_area, int(batch_left_sentinel["largest_area_px"]))
                frame_right_largest_area = max(frame_right_largest_area, int(batch_right_sentinel["largest_area_px"]))
                if bool(args_cli.mosaic_save_frames):
                    mosaic_left_images.append(batch_left)
                    mosaic_right_images.append(batch_right)
                    mosaic_dual_images.append(batch_dual)
                    mosaic_labels_left.append(f"env {batch_env_id:03d} left")
                    mosaic_labels_right.append(f"env {batch_env_id:03d} right")
                    mosaic_labels_dual.append(f"env {batch_env_id:03d} dual")
            mosaic_sentinel_max["left"]["envs_with_magenta"] = max(
                int(mosaic_sentinel_max["left"]["envs_with_magenta"]), frame_left_envs_with_magenta
            )
            mosaic_sentinel_max["right"]["envs_with_magenta"] = max(
                int(mosaic_sentinel_max["right"]["envs_with_magenta"]), frame_right_envs_with_magenta
            )
            mosaic_sentinel_max["left"]["max_components"] = max(
                int(mosaic_sentinel_max["left"]["max_components"]), frame_left_max_components
            )
            mosaic_sentinel_max["right"]["max_components"] = max(
                int(mosaic_sentinel_max["right"]["max_components"]), frame_right_max_components
            )
            mosaic_sentinel_max["left"]["largest_area_px"] = max(
                int(mosaic_sentinel_max["left"]["largest_area_px"]), frame_left_largest_area
            )
            mosaic_sentinel_max["right"]["largest_area_px"] = max(
                int(mosaic_sentinel_max["right"]["largest_area_px"]), frame_right_largest_area
            )
            mosaic_red_visibility["checked_envs"] = max(
                int(mosaic_red_visibility["checked_envs"]), len(selected_mosaic_env_ids)
            )
            existing_min_visible = int(mosaic_red_visibility["min_visible_envs"])
            mosaic_red_visibility["min_visible_envs"] = (
                frame_visible_envs if existing_min_visible == 0 else min(existing_min_visible, frame_visible_envs)
            )
            mosaic_red_visibility["max_oversized_envs"] = max(
                int(mosaic_red_visibility["max_oversized_envs"]), frame_oversized_envs
            )
            mosaic_red_visibility["max_largest_red_area_px"] = max(
                int(mosaic_red_visibility["max_largest_red_area_px"]), frame_max_red_area
            )
            mosaic_red_visibility["max_red_area_fraction"] = max(
                float(mosaic_red_visibility["max_red_area_fraction"]), float(frame_max_red_fraction)
            )
            if bool(args_cli.pallet_visibility_audit):
                mosaic_pallet_visibility["checked_envs"] = max(
                    int(mosaic_pallet_visibility["checked_envs"]), len(selected_mosaic_env_ids)
                )
                existing_visible = int(mosaic_pallet_visibility["min_visible_envs"])
                mosaic_pallet_visibility["min_visible_envs"] = (
                    frame_pallet_visible_envs
                    if existing_visible == 0
                    else min(existing_visible, frame_pallet_visible_envs)
                )
                existing_confident = int(mosaic_pallet_visibility["min_confident_visible_envs"])
                mosaic_pallet_visibility["min_confident_visible_envs"] = (
                    frame_pallet_confident_envs
                    if existing_confident == 0
                    else min(existing_confident, frame_pallet_confident_envs)
                )
                existing_min_area = mosaic_pallet_visibility["min_pallet_area_px"]
                mosaic_pallet_visibility["min_pallet_area_px"] = (
                    int(frame_pallet_min_area or 0)
                    if existing_min_area is None
                    else min(int(existing_min_area), int(frame_pallet_min_area or 0))
                )
                mosaic_pallet_visibility["max_pallet_area_px"] = max(
                    int(mosaic_pallet_visibility["max_pallet_area_px"]), int(frame_pallet_max_area)
                )
                existing_min_fraction = mosaic_pallet_visibility["min_pallet_fraction"]
                mosaic_pallet_visibility["min_pallet_fraction"] = (
                    float(frame_pallet_min_fraction or 0.0)
                    if existing_min_fraction is None
                    else min(float(existing_min_fraction), float(frame_pallet_min_fraction or 0.0))
                )
                mosaic_pallet_visibility["max_pallet_fraction"] = max(
                    float(mosaic_pallet_visibility["max_pallet_fraction"]), float(frame_pallet_max_fraction)
                )
                if frame_hard_corner_records:
                    mosaic_pallet_visibility["hard_corner_envs"] = frame_hard_corner_records
                    hard_min_area = min(int(item["pallet_area_px"]) for item in frame_hard_corner_records)
                    hard_min_fraction = min(float(item["pallet_fraction"]) for item in frame_hard_corner_records)
                    mosaic_pallet_visibility["hard_corner_min_area_px"] = (
                        hard_min_area
                        if mosaic_pallet_visibility["hard_corner_min_area_px"] is None
                        else min(int(mosaic_pallet_visibility["hard_corner_min_area_px"]), hard_min_area)
                    )
                    mosaic_pallet_visibility["hard_corner_min_fraction"] = (
                        hard_min_fraction
                        if mosaic_pallet_visibility["hard_corner_min_fraction"] is None
                        else min(float(mosaic_pallet_visibility["hard_corner_min_fraction"]), hard_min_fraction)
                    )
                    mosaic_pallet_visibility["hard_corner_pass"] = bool(
                        int(mosaic_pallet_visibility["hard_corner_min_area_px"])
                        >= int(args_cli.pallet_visible_min_area_px)
                        and float(mosaic_pallet_visibility["hard_corner_min_fraction"])
                        >= float(args_cli.pallet_confident_min_fraction)
                    )
                mosaic_pallet_visibility["env_records"] = frame_pallet_env_records
            frame_unique_hashes = len(frame_hash_counts)
            frame_duplicate_count = sum(count - 1 for count in frame_hash_counts.values() if count > 1)
            mosaic_hash_min_unique_per_frame = (
                frame_unique_hashes
                if mosaic_hash_min_unique_per_frame is None
                else min(mosaic_hash_min_unique_per_frame, frame_unique_hashes)
            )
            mosaic_hash_max_duplicate_per_frame = max(
                int(mosaic_hash_max_duplicate_per_frame), int(frame_duplicate_count)
            )
            if bool(args_cli.mosaic_save_frames) and selected_mosaic_env_ids:
                _save_png(
                    mosaic_left_dir / f"frame_{saved_frames:06d}.png",
                    _make_mosaic(mosaic_left_images, mosaic_labels_left, int(args_cli.mosaic_cols)),
                )
                _save_png(
                    mosaic_right_dir / f"frame_{saved_frames:06d}.png",
                    _make_mosaic(mosaic_right_images, mosaic_labels_right, int(args_cli.mosaic_cols)),
                )
                _save_png(
                    mosaic_dual_dir / f"frame_{saved_frames:06d}.png",
                    _make_mosaic(mosaic_dual_images, mosaic_labels_dual, int(args_cli.mosaic_cols)),
                )
        left = _to_uint8_hwc(left_batch[env_id])
        right = _to_uint8_hwc(right_batch[env_id])
        left_red = _red_component_stats(left, int(args_cli.red_component_min_area_px))
        right_red = _red_component_stats(right, int(args_cli.red_component_min_area_px))
        left_sentinel = _magenta_component_stats(left, int(args_cli.sentinel_min_area_px))
        right_sentinel = _magenta_component_stats(right, int(args_cli.sentinel_min_area_px))
        for camera_name, stats in (("left", left_red), ("right", right_red)):
            red_rows.append(
                {
                    "frame": int(saved_frames),
                    "step": int(step),
                    "camera": camera_name,
                    "large_red_components": int(stats["large_red_components"]),
                    "largest_area_px": int(stats["largest_area_px"]),
                    "second_largest_area_px": int(stats["second_largest_area_px"]),
                    "components_json": json.dumps(stats["components"], sort_keys=True),
                }
            )
            red_max[camera_name]["large_red_components"] = max(
                int(red_max[camera_name]["large_red_components"]),
                int(stats["large_red_components"]),
            )
            red_max[camera_name]["largest_area_px"] = max(
                int(red_max[camera_name]["largest_area_px"]),
                int(stats["largest_area_px"]),
            )
            red_max[camera_name]["second_largest_area_px"] = max(
                int(red_max[camera_name]["second_largest_area_px"]),
                int(stats["second_largest_area_px"]),
            )
        for camera_name, stats in (("left", left_sentinel), ("right", right_sentinel)):
            sentinel_max[camera_name]["large_magenta_components"] = max(
                int(sentinel_max[camera_name]["large_magenta_components"]),
                int(stats["large_magenta_components"]),
            )
            sentinel_max[camera_name]["largest_area_px"] = max(
                int(sentinel_max[camera_name]["largest_area_px"]),
                int(stats["largest_area_px"]),
            )
        dual = _concat_dual_with_divider(left, right)

        if args_cli.overlay:
            done = bool(torch.as_tensor(terminated | truncated)[env_id].detach().cpu().item())
            origin = raw_env.scene.env_origins[env_id].detach().cpu().numpy().tolist()
            label = [
                f"task={args_cli.task} num_envs={raw_env.num_envs} env_id={env_id} step={step} done={int(done)}",
                f"action=({float(base_action[env_id,0]):+.2f},{float(base_action[env_id,1]):+.2f}) origin=({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})",
            ]
            left = _draw_overlay(left, label)
            right = _draw_overlay(right, label)
            dual = _draw_overlay(dual, label)

        _save_png(left_dir / f"frame_{saved_frames:06d}.png", left)
        _save_png(right_dir / f"frame_{saved_frames:06d}.png", right)
        _save_png(dual_dir / f"frame_{saved_frames:06d}.png", dual)
        saved_frames += 1

    env_dir = output_dir / f"env_{env_id:03d}"
    left_video = env_dir / "left.mp4"
    right_video = env_dir / "right.mp4"
    dual_video = env_dir / "dual_camera.mp4"
    videos_ok = {
        "left": _make_video(left_dir, left_video, int(args_cli.fps)),
        "right": _make_video(right_dir, right_video, int(args_cli.fps)),
        "dual": _make_video(dual_dir, dual_video, int(args_cli.fps)),
    }
    mosaic_videos_ok = {"left": False, "right": False, "dual": False}
    mosaic_left_video = output_dir / "mosaic_left.mp4"
    mosaic_right_video = output_dir / "mosaic_right.mp4"
    mosaic_dual_video = output_dir / "mosaic_dual.mp4"
    if args_cli.record_mosaic and bool(args_cli.mosaic_save_frames):
        mosaic_videos_ok = {
            "left": _make_video(mosaic_left_dir, mosaic_left_video, int(args_cli.fps)),
            "right": _make_video(mosaic_right_dir, mosaic_right_video, int(args_cli.fps)),
            "dual": _make_video(mosaic_dual_dir, mosaic_dual_video, int(args_cli.fps)),
        }
    red_stats_csv = output_dir / f"env_{env_id:03d}" / "red_component_stats.csv"
    if red_rows:
        with red_stats_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(red_rows[0].keys()))
            writer.writeheader()
            writer.writerows(red_rows)

    camera_far = None
    try:
        camera_far = float(getattr(raw_env.cfg, "dual_camera_far_clip_m"))
    except Exception:
        try:
            camera_far = float(raw_env.cfg.tiled_camera_left.spawn.clipping_range[1])
        except Exception:
            pass
    videos_pass = bool(videos_ok["left"] and videos_ok["right"] and videos_ok["dual"] and saved_frames > 0)
    red_gate_pass = all(
        int(red_max[camera]["large_red_components"]) <= int(args_cli.red_component_gate)
        and int(red_max[camera]["second_largest_area_px"]) < int(args_cli.max_second_red_area_px)
        for camera in ("left", "right")
    )
    frame_area = max(1, int(raw_env.cfg.dual_camera_width) * int(raw_env.cfg.dual_camera_height))
    fork_visibility_pass = any(
        int(red_max[camera]["largest_area_px"]) >= int(args_cli.min_fork_red_area_px)
        and (int(red_max[camera]["largest_area_px"]) / frame_area) <= float(args_cli.max_red_area_fraction)
        for camera in ("left", "right")
    )
    sentinel_gate_pass = all(
        int(sentinel_max[camera]["large_magenta_components"]) == 0 for camera in ("left", "right")
    )
    sentinel_probe_coverage_pass = True
    if bool(args_cli.sentinel_audit) and (bool(args_cli.sentinel_room_probes_all_envs) or bool(args_cli.sentinel_foreign_envs)):
        sentinel_probe_coverage_pass = bool(len(sentinel_records) > 0)
    full_mosaic_coverage = bool(
        bool(args_cli.record_mosaic)
        and selected_mosaic_env_ids == list(range(int(raw_env.num_envs)))
    )
    if full_mosaic_coverage and not bool(getattr(raw_env.cfg, "vision_room_enable", False)):
        # In the no-wall path, full-mosaic hash coverage plus far clipping is the
        # primary cross-env proof.  Foreign sentinels are target-camera probes and
        # would become own-env distractors for non-target mosaic tiles.
        sentinel_probe_coverage_pass = True
    mosaic_sentinel_gate_pass = True
    if args_cli.record_mosaic:
        mosaic_sentinel_gate_pass = all(
            int(mosaic_sentinel_max[camera]["envs_with_magenta"]) == 0 for camera in ("left", "right")
        )
    mosaic_visibility_pass = True
    if args_cli.record_mosaic:
        checked_envs = int(mosaic_red_visibility["checked_envs"])
        mosaic_visibility_pass = bool(
            checked_envs > 0
            and int(mosaic_red_visibility["min_visible_envs"]) >= checked_envs
            and int(mosaic_red_visibility["max_oversized_envs"]) == 0
        )
    mosaic_unique_hashes = len(mosaic_hash_counts)
    mosaic_hash_duplicate_count = sum(count - 1 for count in mosaic_hash_counts.values() if count > 1)
    mosaic_identity_pass = True
    if args_cli.record_mosaic and (bool(args_cli.preinsert_pose_sweep) or str(args_cli.audit_pose) == "reset"):
        checked_hash_envs = int(mosaic_red_visibility["checked_envs"])
        mosaic_identity_pass = bool(
            checked_hash_envs > 0
            and mosaic_hash_min_unique_per_frame is not None
            and int(mosaic_hash_min_unique_per_frame) >= checked_hash_envs
            and int(mosaic_hash_max_duplicate_per_frame) == 0
        )
    pallet_visibility_pass = True
    pallet_confident_visibility_pass = True
    pallet_hard_corner_pass = True
    if bool(args_cli.pallet_visibility_audit):
        checked_pallet_envs = int(mosaic_pallet_visibility["checked_envs"])
        pallet_visibility_pass = bool(
            bool(pallet_visibility_material.get("pass", False))
            and checked_pallet_envs > 0
            and int(mosaic_pallet_visibility["min_visible_envs"]) >= checked_pallet_envs
        )
        pallet_confident_visibility_pass = bool(
            checked_pallet_envs > 0
            and int(mosaic_pallet_visibility["min_confident_visible_envs"]) >= checked_pallet_envs
        )
        pallet_hard_corner_pass = bool(mosaic_pallet_visibility.get("hard_corner_pass", True))
    visual_target_visibility_pass = fork_visibility_pass
    mosaic_target_visibility_pass = mosaic_visibility_pass
    if bool(args_cli.pallet_visibility_audit):
        visual_target_visibility_pass = bool(pallet_visibility_pass and pallet_confident_visibility_pass)
        mosaic_target_visibility_pass = bool(pallet_visibility_pass and pallet_confident_visibility_pass)
    vision_room_enabled = bool(getattr(raw_env.cfg, "vision_room_enable", False))
    env_spacing = float(raw_env.cfg.scene.env_spacing)
    far_clip_ratio = float(getattr(raw_env.cfg, "visual_isolation_far_clip_env_spacing_ratio", 0.45))
    max_far_clip = env_spacing * far_clip_ratio
    far_clip_pass = bool(camera_far is not None and float(camera_far) <= max_far_clip)
    room_pass = bool(room_prim_check.get("pass")) if vision_room_enabled else True
    visual_isolation_pass = bool((vision_room_enabled and room_pass) or ((not vision_room_enabled) and far_clip_pass))
    visual_isolation = {
        "pass": visual_isolation_pass,
        "mode": "room" if vision_room_enabled else "far_clip_spacing",
        "vision_room_enable": vision_room_enabled,
        "vision_room_pass": room_pass,
        "camera_far": float(camera_far) if camera_far is not None else None,
        "env_spacing": env_spacing,
        "far_clip_env_spacing_ratio": far_clip_ratio,
        "max_far_clip_m": max_far_clip,
        "far_clip_pass": far_clip_pass,
    }
    geometry = _geometry_audit(raw_env)
    geometry_pass = bool(geometry.get("preinsert_pass", False)) if args_cli.audit_pose == "preinsert" else True
    mosaic_videos_pass = bool(
        not args_cli.record_mosaic
        or not bool(args_cli.mosaic_save_frames)
        or (mosaic_videos_ok["left"] and mosaic_videos_ok["right"] and mosaic_videos_ok["dual"])
    )
    foreign_leakage_pass = bool(
        videos_pass
        and mosaic_videos_pass
        and visual_isolation_pass
        and mosaic_identity_pass
        and sentinel_probe_coverage_pass
        and sentinel_gate_pass
        and mosaic_sentinel_gate_pass
    )
    camera_learnability_pass = bool(
        videos_pass
        and mosaic_videos_pass
        and visual_isolation_pass
        and geometry_pass
        and red_gate_pass
        and visual_target_visibility_pass
        and mosaic_target_visibility_pass
        and mosaic_identity_pass
        and pallet_visibility_pass
        and pallet_confident_visibility_pass
        and pallet_hard_corner_pass
    )
    acceptance_pass = bool(
        foreign_leakage_pass
        and camera_learnability_pass
    )

    camera_signature = _camera_signature(raw_env, camera_far)
    summary = {
        "task": args_cli.task,
        "num_envs": int(raw_env.num_envs),
        "env_id": env_id,
        "camera_signature": camera_signature,
        "pass": acceptance_pass,
        "foreign_leakage_pass": foreign_leakage_pass,
        "camera_learnability_pass": camera_learnability_pass,
        "acceptance": {
            "videos_pass": videos_pass,
            "mosaic_videos_pass": mosaic_videos_pass,
            "foreign_leakage_pass": foreign_leakage_pass,
            "camera_learnability_pass": camera_learnability_pass,
            "red_gate_pass": red_gate_pass,
            "fork_visibility_pass": fork_visibility_pass,
            "visual_target_visibility_pass": visual_target_visibility_pass,
            "mosaic_visibility_pass": mosaic_visibility_pass,
            "mosaic_target_visibility_pass": mosaic_target_visibility_pass,
            "mosaic_identity_pass": mosaic_identity_pass,
            "pallet_visibility_pass": pallet_visibility_pass,
            "pallet_confident_visibility_pass": pallet_confident_visibility_pass,
            "pallet_hard_corner_pass": pallet_hard_corner_pass,
            "sentinel_gate_pass": sentinel_gate_pass,
            "mosaic_sentinel_gate_pass": mosaic_sentinel_gate_pass,
            "sentinel_probe_coverage_pass": sentinel_probe_coverage_pass,
            "visual_isolation_pass": visual_isolation_pass,
            "vision_room_pass": room_pass,
            "far_clip_spacing_pass": far_clip_pass,
            "geometry_pass": geometry_pass,
            "manual_review_recommended": True,
        },
        "steps": int(args_cli.steps),
        "warmup_steps": int(args_cli.warmup_steps),
        "record_every": int(args_cli.record_every),
        "saved_frames": int(saved_frames),
        "action": {
            "drive": float(args_cli.drive),
            "steer": float(args_cli.steer),
            "lift": float(args_cli.lift),
            "action_dim": int(action_dim),
        },
        "env_spacing": env_spacing,
        "filter_collisions": bool(getattr(raw_env.cfg.scene, "filter_collisions", True)),
        "vision_room_enable": vision_room_enabled,
        "vision_room_prim_check": room_prim_check,
        "visual_isolation": visual_isolation,
        "camera_far": camera_far,
        "dual_camera_config": {
            "camera_version": camera_signature["camera_version"],
            "config_hash_sha1": camera_signature["config_hash_sha1"],
            "hfov_deg": float(raw_env.cfg.dual_camera_hfov_deg),
            "left_pos_local": [float(v) for v in raw_env.cfg.dual_camera_left_pos_local],
            "right_pos_local": [float(v) for v in raw_env.cfg.dual_camera_right_pos_local],
            "left_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_left_rpy_local_deg],
            "right_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_right_rpy_local_deg],
        },
        "red_component_min_area_px": int(args_cli.red_component_min_area_px),
        "red_component_gate": int(args_cli.red_component_gate),
        "max_second_red_area_px": int(args_cli.max_second_red_area_px),
        "min_fork_red_area_px": int(args_cli.min_fork_red_area_px),
        "max_red_area_fraction": float(args_cli.max_red_area_fraction),
        "red_component_stats_csv": str(red_stats_csv),
        "red_component_max": red_max,
        "sentinel_audit": bool(args_cli.sentinel_audit),
        "sentinel_room_probes_all_envs": bool(args_cli.sentinel_room_probes_all_envs),
        "sentinel_foreign_envs": bool(args_cli.sentinel_foreign_envs),
        "sentinel_count": int(len(sentinel_records)),
        "sentinel_records": sentinel_records,
        "sentinel_min_area_px": int(args_cli.sentinel_min_area_px),
        "sentinel_component_max": sentinel_max,
        "mosaic_sentinel_component_max": mosaic_sentinel_max,
        "mosaic_env_ids": selected_mosaic_env_ids,
        "mosaic_env_id_count": int(len(selected_mosaic_env_ids)),
        "mosaic_env_coverage_complete": bool(
            bool(args_cli.record_mosaic)
            and len(selected_mosaic_env_ids) == int(raw_env.num_envs)
            and selected_mosaic_env_ids == list(range(int(raw_env.num_envs)))
        ),
        "mosaic_save_frames": bool(args_cli.mosaic_save_frames),
        "mosaic_red_visibility": mosaic_red_visibility,
        "pallet_visibility_material": pallet_visibility_material,
        "pallet_visibility_thresholds": {
            "visible_min_area_px": int(args_cli.pallet_visible_min_area_px),
            "confident_min_fraction": float(args_cli.pallet_confident_min_fraction),
            "confident_min_area_px": int(
                max(1, round(frame_area * float(args_cli.pallet_confident_min_fraction)))
            ),
        },
        "mosaic_pallet_visibility": mosaic_pallet_visibility,
        "mosaic_dual_hashes": {
            "unique_hashes": int(mosaic_unique_hashes),
            "duplicate_count": int(mosaic_hash_duplicate_count),
            "min_unique_per_frame": (
                int(mosaic_hash_min_unique_per_frame) if mosaic_hash_min_unique_per_frame is not None else 0
            ),
            "max_duplicate_per_frame": int(mosaic_hash_max_duplicate_per_frame),
            "identity_pass": bool(mosaic_identity_pass),
            "top_counts": [
                {
                    "hash": str(hash_value),
                    "count": int(count),
                    "sample": mosaic_hash_sample.get(hash_value, {}),
                }
                for hash_value, count in mosaic_hash_counts.most_common(8)
            ],
        },
        "audit_pose": audit_pose_summary,
        "geometry": geometry,
        "red_component_gate_hint": {
            "diagnostic_only": True,
            "large_red_components_per_camera_should_be_lte": int(args_cli.red_component_gate),
            "second_largest_area_px_should_be_lt": int(args_cli.max_second_red_area_px),
            "manual_review_required": True,
        },
        "env_origin": raw_env.scene.env_origins[env_id].detach().cpu().tolist(),
        "left_video": str(left_video),
        "right_video": str(right_video),
        "dual_camera_video": str(dual_video),
        "mosaic_left_video": str(mosaic_left_video) if args_cli.record_mosaic else None,
        "mosaic_right_video": str(mosaic_right_video) if args_cli.record_mosaic else None,
        "mosaic_dual_video": str(mosaic_dual_video) if args_cli.record_mosaic else None,
        "videos_ok": videos_ok,
        "mosaic_videos_ok": mosaic_videos_ok,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
