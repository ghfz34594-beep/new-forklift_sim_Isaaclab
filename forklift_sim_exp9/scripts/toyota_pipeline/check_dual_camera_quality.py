"""Smoke-check Toyota dual-camera observations before visual training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check Toyota dual-camera image quality")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--min_mean", type=float, default=0.05)
parser.add_argument("--min_std", type=float, default=0.02)
parser.add_argument("--min_active_ratio", type=float, default=0.20)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torchvision.utils import save_image

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401


def _normalize(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu()
    if image.max() > 1.0:
        image = image / 255.0
    return torch.clamp(image, 0.0, 1.0)


def _stats(name: str, image: torch.Tensor) -> dict[str, float]:
    image = _normalize(image)
    return {
        "mean": float(image.mean().item()),
        "std": float(image.std().item()),
        "min": float(image.min().item()),
        "max": float(image.max().item()),
        "active_ratio": float((image > 0.10).float().mean().item()),
    }


def _format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
        f"active_ratio={stats['active_ratio']:.4f}"
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    for _ in range(max(0, int(args_cli.steps))):
        action = torch.zeros((env.unwrapped.num_envs, 2), dtype=torch.float32, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(torch.as_tensor(terminated | truncated).any().item()):
            obs, _ = env.reset()

    raw_obs = env.unwrapped._get_observations()
    left = raw_obs["image_left"]
    right = raw_obs["image_right"]
    left_stats = _stats("left", left)
    right_stats = _stats("right", right)

    print(_format_stats("left", left_stats), flush=True)
    print(_format_stats("right", right_stats), flush=True)
    print(
        "shape: "
        f"left={tuple(left.shape)}, right={tuple(right.shape)}, "
        f"wait_for_textures={getattr(env.unwrapped.cfg, 'wait_for_textures', None)}",
        flush=True,
    )
    print(
        "forklift_usd_path: "
        f"{getattr(env.unwrapped.cfg, 'forklift_usd_path', '<unknown>')}",
        flush=True,
    )

    if args_cli.output_dir:
        output_dir = Path(args_cli.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(_normalize(left[0]), output_dir / "camera_left.png")
        save_image(_normalize(right[0]), output_dir / "camera_right.png")
        print(f"saved: {output_dir / 'camera_left.png'}")
        print(f"saved: {output_dir / 'camera_right.png'}")

    failures = []
    for name, stats in (("left", left_stats), ("right", right_stats)):
        if stats["mean"] < float(args_cli.min_mean):
            failures.append(f"{name} mean below {args_cli.min_mean}")
        if stats["std"] < float(args_cli.min_std):
            failures.append(f"{name} std below {args_cli.min_std}")
        if stats["active_ratio"] < float(args_cli.min_active_ratio):
            failures.append(f"{name} active_ratio below {args_cli.min_active_ratio}")

    env.close()
    simulation_app.close()

    if failures:
        raise RuntimeError("camera quality check failed: " + "; ".join(failures))
    print("[check_dual_camera_quality] PASS", flush=True)


if __name__ == "__main__":
    main()
