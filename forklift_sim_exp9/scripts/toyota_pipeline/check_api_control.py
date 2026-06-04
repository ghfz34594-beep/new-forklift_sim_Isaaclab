"""Validate the non-web forklift API for cameras, drive, steer, lift, stop."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check ForkliftIsaacApi control surface")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--drive", type=float, default=0.65)
parser.add_argument("--steer", type=float, default=0.65)
parser.add_argument("--lift", type=float, default=0.9)
parser.add_argument("--drive_steps", type=int, default=40)
parser.add_argument("--steer_steps", type=int, default=45)
parser.add_argument("--lift_steps", type=int, default=80)
parser.add_argument("--stop_steps", type=int, default=8)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--min_drive_delta_m", type=float, default=0.03)
parser.add_argument("--min_yaw_delta_deg", type=float, default=0.5)
parser.add_argument("--min_lift_delta_m", type=float, default=0.02)
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

from forklift_api import ForkliftIsaacApi


def _distance_xy(a: dict, b: dict) -> float:
    return math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))


def _angle_delta_deg(a: float, b: float) -> float:
    delta = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return abs(delta)


def _save_camera_samples(api: ForkliftIsaacApi, output_dir: str | None) -> None:
    cameras = api.get_cameras()
    print("camera_keys", sorted(cameras.keys()), flush=True)
    for name, image in cameras.items():
        image_f = image.float()
        if image_f.max() > 1.0:
            image_f = image_f / 255.0
        image_f = image_f.clamp(0.0, 1.0)
        print(
            f"camera/{name}: shape={tuple(image.shape)} "
            f"mean={float(image_f.mean()):.4f} std={float(image_f.std()):.4f}",
            flush=True,
        )
    if "left" not in cameras or "right" not in cameras:
        raise RuntimeError("dual cameras not available through API")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, image in cameras.items():
            image_f = image.float()
            if image_f.max() > 1.0:
                image_f = image_f / 255.0
            save_image(image_f[0].clamp(0.0, 1.0), out / f"api_camera_{name}.png")


def _repeat(api: ForkliftIsaacApi, drive: float, steer: float, lift: float, steps: int) -> dict:
    state = api.get_state()
    for _ in range(int(steps)):
        state, terminated, truncated, _ = api.set_command(drive, steer, lift)
        if bool(torch.as_tensor(terminated | truncated).any().item()):
            break
    return state


def main() -> None:
    if int(args_cli.num_envs) != 1:
        raise ValueError("ForkliftIsaacApi check expects --num_envs 1")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = False
    env_cfg.action_space = 3

    env = gym.make(args_cli.task, cfg=env_cfg)
    api = ForkliftIsaacApi(env)
    reset_state = api.reset()
    _save_camera_samples(api, args_cli.output_dir)

    after_drive = _repeat(api, args_cli.drive, 0.0, 0.0, args_cli.drive_steps)
    after_steer = _repeat(api, args_cli.drive, args_cli.steer, 0.0, args_cli.steer_steps)
    before_lift = api.get_state()
    after_lift = _repeat(api, 0.0, 0.0, args_cli.lift, args_cli.lift_steps)
    for _ in range(int(args_cli.stop_steps)):
        stopped_state, _, _, _ = api.stop()

    drive_delta = _distance_xy(reset_state, after_drive)
    yaw_delta = _angle_delta_deg(after_drive["yaw_deg"], after_steer["yaw_deg"])
    lift_delta = float(after_lift["lift_joint_m"]) - float(before_lift["lift_joint_m"])

    print("state/reset", reset_state, flush=True)
    print("state/after_drive", after_drive, flush=True)
    print("state/after_steer", after_steer, flush=True)
    print("state/before_lift", before_lift, flush=True)
    print("state/after_lift", after_lift, flush=True)
    print("state/stopped", stopped_state, flush=True)
    print(
        f"metrics drive_delta_m={drive_delta:.4f} "
        f"yaw_delta_deg={yaw_delta:.4f} lift_delta_m={lift_delta:.4f}",
        flush=True,
    )

    failures = []
    if drive_delta < float(args_cli.min_drive_delta_m):
        failures.append(f"drive delta {drive_delta:.4f} < {args_cli.min_drive_delta_m}")
    if yaw_delta < float(args_cli.min_yaw_delta_deg):
        failures.append(f"yaw delta {yaw_delta:.4f} < {args_cli.min_yaw_delta_deg}")
    if lift_delta < float(args_cli.min_lift_delta_m):
        failures.append(f"lift delta {lift_delta:.4f} < {args_cli.min_lift_delta_m}")

    env.close()
    simulation_app.close()

    if failures:
        raise RuntimeError("API control check failed: " + "; ".join(failures))
    print("[check_api_control] PASS", flush=True)


if __name__ == "__main__":
    main()
