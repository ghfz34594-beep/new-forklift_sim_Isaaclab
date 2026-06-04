"""Record the exact multi-env image tensors that the vision actor feeds to ResNet.

The current VisionActorCritic preprocessing path is:
  1. convert image observation to NCHW float,
  2. scale uint8 images by 255, with a float fallback if the batch max is above 1,
  3. clamp to [0, 1],
  4. apply ImageNet mean/std normalization when the actor uses ImageNet init,
  5. pass the result into image_encoder.

This script launches the task with many envs and saves visual grids of those
post-preprocess tensors.  It also writes the first recorded batch as a .pt file
for exact inspection.
"""

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


parser = argparse.ArgumentParser(description="Record multi-env ResNet input images from Toyota dual-camera obs")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--warmup_steps", type=int, default=4)
parser.add_argument("--record_every", type=int, default=2)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--drive", type=float, default=0.0)
parser.add_argument("--steer", type=float, default=0.0)
parser.add_argument("--lift", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=20260521)
parser.add_argument("--env_spacing", type=float, default=None)
parser.add_argument("--camera_far", type=float, default=None)
parser.add_argument("--vision_room", action="store_true", default=None, help="Force per-env room occlusion on.")
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room", help="Force per-env room occlusion off.")
parser.add_argument("--grid_cols", type=int, default=4)
parser.add_argument("--save_first_env_tiles", action="store_true", default=True)
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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _ensure_actor_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Mirror VisionActorCritic._ensure_image_tensor()."""
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims, got shape={tuple(image.shape)}")
    if image.shape[1] == 3:
        return image
    if image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2)
    raise ValueError(f"Unexpected image shape={tuple(image.shape)}")


def _scale_image_to_unit(image: torch.Tensor) -> torch.Tensor:
    """Mirror VisionActorCritic._scale_image_to_unit()."""
    image = _ensure_actor_image_tensor(image)
    if image.dtype == torch.uint8:
        return image.float() / 255.0
    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    return torch.clamp(image, 0.0, 1.0)


def _resnet_input(image: torch.Tensor) -> torch.Tensor:
    """Mirror VisionActorCritic._preprocess_image_for_encoder() for ImageNet actors."""
    image = _scale_image_to_unit(image)
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype).unsqueeze(0)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype).unsqueeze(0)
    return (image - mean) / std


def _to_uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0.0, 1.0)
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected CHW image, got shape={tuple(image.shape)}")
    image = image[:3].permute(1, 2, 0).numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _font():
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", 13)
    except Exception:
        return ImageFont.load_default()


def _make_labeled_grid(tiles: list[np.ndarray], labels: list[str], cols: int) -> np.ndarray:
    if not tiles:
        raise ValueError("Cannot make a grid without tiles")
    cols = max(1, int(cols))
    rows = int(math.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    label_h = 22
    pad = 6
    canvas_w = pad + cols * (tile_w + pad)
    canvas_h = pad + rows * (tile_h + label_h + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + label_h + pad)
        draw.text((x, y + 3), labels[idx], fill=(235, 235, 235), font=font)
        canvas.paste(Image.fromarray(tile), (x, y + label_h))
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


def _policy_obs(obs: Any) -> dict[str, torch.Tensor]:
    if hasattr(obs, "keys") and "policy" in obs:
        return obs["policy"]
    return obs


def _stats_rows(frame: int, step: int, camera: str, batch: torch.Tensor) -> list[dict[str, float | int | str]]:
    rows = []
    for env_id in range(batch.shape[0]):
        image = batch[env_id].detach().float().cpu()
        rows.append(
            {
                "frame": int(frame),
                "step": int(step),
                "camera": camera,
                "env_id": int(env_id),
                "mean": float(image.mean().item()),
                "std": float(image.std().item()),
                "min": float(image.min().item()),
                "max": float(image.max().item()),
                "active_ratio_gt_0p10": float((image > 0.10).float().mean().item()),
            }
        )
    return rows


def _write_stats(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if int(args_cli.num_envs) < 2:
        raise ValueError("Use --num_envs >= 2 for this multi-env check")

    output_dir = Path(args_cli.output_dir)
    left_grid_dir = output_dir / "left_resnet_input_grid_frames"
    right_grid_dir = output_dir / "right_resnet_input_grid_frames"
    dual_grid_dir = output_dir / "dual_resnet_input_grid_frames"
    first_tiles_dir = output_dir / "first_frame_tiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    obs, _ = env.reset()

    action_dim = int(getattr(raw_env.cfg, "action_space", 2))
    action = torch.zeros((raw_env.num_envs, action_dim), dtype=torch.float32, device=raw_env.device)
    action[:, 0] = float(args_cli.drive)
    if action_dim >= 2:
        action[:, 1] = float(args_cli.steer)
    if action_dim >= 3:
        action[:, 2] = float(args_cli.lift)
    action = action.clamp(-1.0, 1.0)

    for _ in range(max(0, int(args_cli.warmup_steps))):
        obs, _, _, _, _ = env.step(action)

    stats: list[dict[str, float | int | str]] = []
    saved_frames = 0
    first_tensor_written = False

    for step in range(int(args_cli.steps)):
        obs, _, _, _, _ = env.step(action)
        if step % max(1, int(args_cli.record_every)) != 0:
            continue

        policy_obs = _policy_obs(obs)
        left_input = _resnet_input(policy_obs["image_left"])
        right_input = _resnet_input(policy_obs["image_right"])
        proprio = policy_obs["proprio"].detach().cpu()

        if not first_tensor_written:
            torch.save(
                {
                    "image_left": left_input.detach().cpu(),
                    "image_right": right_input.detach().cpu(),
                    "proprio": proprio,
                    "notes": (
                        "These tensors mirror VisionActorCritic._encode_policy_obs(): "
                        "NCHW float, scaled to [0, 1], then ImageNet mean/std normalized."
                    ),
                },
                output_dir / "first_resnet_input_batch.pt",
            )
            if args_cli.save_first_env_tiles:
                for env_id in range(raw_env.num_envs):
                    _save_png(first_tiles_dir / f"env_{env_id:03d}_left.png", _to_uint8_hwc(left_input[env_id]))
                    _save_png(first_tiles_dir / f"env_{env_id:03d}_right.png", _to_uint8_hwc(right_input[env_id]))
                    pair = np.concatenate([_to_uint8_hwc(left_input[env_id]), _to_uint8_hwc(right_input[env_id])], axis=1)
                    _save_png(first_tiles_dir / f"env_{env_id:03d}_dual.png", pair)
            first_tensor_written = True

        stats.extend(_stats_rows(saved_frames, step, "left", left_input))
        stats.extend(_stats_rows(saved_frames, step, "right", right_input))

        left_tiles = [_to_uint8_hwc(left_input[i]) for i in range(raw_env.num_envs)]
        right_tiles = [_to_uint8_hwc(right_input[i]) for i in range(raw_env.num_envs)]
        dual_tiles = [np.concatenate([left_tiles[i], right_tiles[i]], axis=1) for i in range(raw_env.num_envs)]
        labels = [f"env {i:02d}" for i in range(raw_env.num_envs)]

        _save_png(
            left_grid_dir / f"frame_{saved_frames:06d}.png",
            _make_labeled_grid(left_tiles, [f"{label} left" for label in labels], int(args_cli.grid_cols)),
        )
        _save_png(
            right_grid_dir / f"frame_{saved_frames:06d}.png",
            _make_labeled_grid(right_tiles, [f"{label} right" for label in labels], int(args_cli.grid_cols)),
        )
        _save_png(
            dual_grid_dir / f"frame_{saved_frames:06d}.png",
            _make_labeled_grid(dual_tiles, [f"{label} left | right" for label in labels], int(args_cli.grid_cols)),
        )
        saved_frames += 1

    _write_stats(output_dir / "resnet_input_stats.csv", stats)
    videos_ok = {
        "left": _make_video(left_grid_dir, output_dir / "left_resnet_input_grid.mp4", int(args_cli.fps)),
        "right": _make_video(right_grid_dir, output_dir / "right_resnet_input_grid.mp4", int(args_cli.fps)),
        "dual": _make_video(dual_grid_dir, output_dir / "dual_resnet_input_grid.mp4", int(args_cli.fps)),
    }

    camera_far = None
    try:
        camera_far = float(getattr(raw_env.cfg, "dual_camera_far_clip_m"))
    except Exception:
        try:
            camera_far = float(raw_env.cfg.tiled_camera_left.spawn.clipping_range[1])
        except Exception:
            pass
    sample_policy_obs = _policy_obs(obs)
    sample_left = _resnet_input(sample_policy_obs["image_left"])
    summary = {
        "task": args_cli.task,
        "num_envs": int(raw_env.num_envs),
        "saved_frames": int(saved_frames),
        "steps": int(args_cli.steps),
        "warmup_steps": int(args_cli.warmup_steps),
        "record_every": int(args_cli.record_every),
        "env_spacing": float(raw_env.cfg.scene.env_spacing),
        "filter_collisions": bool(getattr(raw_env.cfg.scene, "filter_collisions", True)),
        "vision_room_enable": bool(getattr(raw_env.cfg, "vision_room_enable", False)),
        "camera_far": camera_far,
        "action": {
            "action_dim": int(action_dim),
            "drive": float(args_cli.drive),
            "steer": float(args_cli.steer),
            "lift": float(args_cli.lift),
        },
        "resnet_input_shape": list(sample_left.shape),
        "resnet_input_dtype": str(sample_left.dtype),
        "preprocess": "NCHW float; uint8 /255 or float fallback /255 when batch max > 1; clamp to [0, 1]; ImageNet mean/std normalize before image_encoder.",
        "first_resnet_input_batch": str(output_dir / "first_resnet_input_batch.pt"),
        "stats_csv": str(output_dir / "resnet_input_stats.csv"),
        "left_grid_video": str(output_dir / "left_resnet_input_grid.mp4"),
        "right_grid_video": str(output_dir / "right_resnet_input_grid.mp4"),
        "dual_grid_video": str(output_dir / "dual_resnet_input_grid.mp4"),
        "videos_ok": videos_ok,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
