"""Check that Toyota dual camera poses stay fixed in the forklift body frame."""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check dual-camera mount sync")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--drive", type=float, default=0.45)
parser.add_argument("--env_spacing", type=float, default=20.0)
parser.add_argument("--camera_far", type=float, default=8.0)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=100.0)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(150.0, 75.0, 140.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(150.0, -75.0, 140.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, -20.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, 20.0))
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)


def _set_camera_far(env_cfg, far: float) -> None:
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


def _fmt(values: torch.Tensor) -> str:
    values = values.detach().cpu().tolist()
    return "(" + ", ".join(f"{float(v):+.4f}" for v in values) + ")"


def _fmt6(values: torch.Tensor) -> str:
    values = values.detach().cpu().tolist()
    return "(" + ", ".join(f"{float(v):+.6f}" for v in values) + ")"


def _camera_local_pos_to_m(raw_env, values: tuple[float, float, float]) -> torch.Tensor:
    pos = torch.tensor(values, dtype=torch.float32, device=raw_env.device)
    if torch.max(torch.abs(pos)).item() > 10.0:
        pos = pos * 0.01
    return pos


def _quat_from_rpy_deg(raw_env, rpy_deg: tuple[float, float, float]) -> torch.Tensor:
    """Return a wxyz quaternion from XYZ Euler angles in degrees."""
    roll_deg, pitch_deg, yaw_deg = rpy_deg
    cr = math.cos(math.radians(roll_deg) * 0.5)
    sr = math.sin(math.radians(roll_deg) * 0.5)
    cp = math.cos(math.radians(pitch_deg) * 0.5)
    sp = math.sin(math.radians(pitch_deg) * 0.5)
    cy = math.cos(math.radians(yaw_deg) * 0.5)
    sy = math.sin(math.radians(yaw_deg) * 0.5)
    return torch.tensor(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=torch.float32,
        device=raw_env.device,
    )


def _canonical_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[..., 0:1] < 0.0, -quat, quat)


def _rpy_deg_from_quat(quat: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler_xyz_from_quat(quat.unsqueeze(0), wrap_to_2pi=False)
    return torch.stack((roll[0], pitch[0], yaw[0])) * (180.0 / math.pi)


def _camera_axes_in_body(quat_local_world_convention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # In IsaacLab's "world" camera convention, camera +X is forward and +Z is up.
    basis = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
        device=quat_local_world_convention.device,
    )
    q = quat_local_world_convention.unsqueeze(0).repeat(2, 1)
    axes = quat_apply(q, basis)
    return axes[0], axes[1]


def _print_camera_orientation(
    name: str,
    actual_local_quat: torch.Tensor,
    expected_local_quat: torch.Tensor,
) -> None:
    actual_local_quat = _canonical_quat(actual_local_quat)
    expected_local_quat = _canonical_quat(expected_local_quat)
    actual_rpy = _rpy_deg_from_quat(actual_local_quat)
    expected_rpy = _rpy_deg_from_quat(expected_local_quat)
    forward_body, up_body = _camera_axes_in_body(actual_local_quat)
    print(
        f"[camera_quat] {name} actual_local_wxyz={_fmt6(actual_local_quat)} "
        f"expected_local_wxyz={_fmt6(expected_local_quat)} "
        f"actual_rpy_deg={_fmt(actual_rpy)} expected_rpy_deg={_fmt(expected_rpy)}",
        flush=True,
    )
    print(
        f"[camera_axes] {name} forward_in_body_xyz={_fmt(forward_body)} "
        f"up_in_body_xyz={_fmt(up_body)}",
        flush=True,
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.action_space = 2
    env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    env_cfg.dual_camera_hfov_deg = float(args_cli.dual_camera_hfov_deg)
    env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args_cli.dual_camera_left_pos)
    env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args_cli.dual_camera_right_pos)
    env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)
    env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_right_rpy_deg)
    _set_camera_far(env_cfg, float(args_cli.camera_far))
    if hasattr(env_cfg, "preinsert_action_guard_enable"):
        env_cfg.preinsert_action_guard_enable = False
    if hasattr(env_cfg, "toyota_action_noise_std"):
        env_cfg.toyota_action_noise_std = 0.0
    if hasattr(env_cfg, "toyota_velocity_obs_noise_std"):
        env_cfg.toyota_velocity_obs_noise_std = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    env.reset()
    print(
        "[camera_mount] prim_paths="
        f"left={raw_env._camera_left._view.prim_paths[0]} "
        f"right={raw_env._camera_right._view.prim_paths[0]}",
        flush=True,
    )

    action = torch.tensor([[float(args_cli.drive), 0.0, 0.0]], device=raw_env.device).repeat(raw_env.num_envs, 1)
    left_pos_l = _camera_local_pos_to_m(raw_env, tuple(float(v) for v in raw_env.cfg.dual_camera_left_pos_local))
    right_pos_l = _camera_local_pos_to_m(raw_env, tuple(float(v) for v in raw_env.cfg.dual_camera_right_pos_local))
    left_quat_l = _quat_from_rpy_deg(raw_env, tuple(float(v) for v in raw_env.cfg.dual_camera_left_rpy_local_deg))
    right_quat_l = _quat_from_rpy_deg(raw_env, tuple(float(v) for v in raw_env.cfg.dual_camera_right_rpy_local_deg))
    max_pos_err = 0.0
    max_angle_err = 0.0
    for step in range(int(args_cli.steps)):
        env.step(action)
        root_pos_batch, root_quat_batch = raw_env._latest_robot_root_pose_from_physx()
        root_pos = root_pos_batch[0]
        root_quat = root_quat_batch[0]
        left_pose, left_quat_opengl = raw_env._camera_left._view.get_world_poses([0])
        right_pose, right_quat_opengl = raw_env._camera_right._view.get_world_poses([0])
        left_pos = left_pose[0]
        right_pos = right_pose[0]
        left_quat_world = convert_camera_frame_orientation_convention(
            left_quat_opengl, origin="opengl", target="world"
        )[0]
        right_quat_world = convert_camera_frame_orientation_convention(
            right_quat_opengl, origin="opengl", target="world"
        )[0]
        left_local = quat_apply_inverse(root_quat.unsqueeze(0), (left_pos - root_pos).unsqueeze(0))[0]
        right_local = quat_apply_inverse(root_quat.unsqueeze(0), (right_pos - root_pos).unsqueeze(0))[0]
        expected_left_pos = root_pos + quat_apply(root_quat.unsqueeze(0), left_pos_l.unsqueeze(0))[0]
        expected_right_pos = root_pos + quat_apply(root_quat.unsqueeze(0), right_pos_l.unsqueeze(0))[0]
        expected_left_quat = quat_mul(root_quat.unsqueeze(0), left_quat_l.unsqueeze(0))[0]
        expected_right_quat = quat_mul(root_quat.unsqueeze(0), right_quat_l.unsqueeze(0))[0]
        left_local_quat = quat_mul(quat_inv(root_quat.unsqueeze(0)), left_quat_world.unsqueeze(0))[0]
        right_local_quat = quat_mul(quat_inv(root_quat.unsqueeze(0)), right_quat_world.unsqueeze(0))[0]
        left_pos_err = torch.linalg.norm(left_pos - expected_left_pos).item()
        right_pos_err = torch.linalg.norm(right_pos - expected_right_pos).item()
        left_dot = torch.sum(left_quat_world * expected_left_quat).abs().clamp(max=1.0)
        right_dot = torch.sum(right_quat_world * expected_right_quat).abs().clamp(max=1.0)
        left_angle_err = (2.0 * torch.acos(left_dot)).item()
        right_angle_err = (2.0 * torch.acos(right_dot)).item()
        max_pos_err = max(max_pos_err, left_pos_err, right_pos_err)
        max_angle_err = max(max_angle_err, left_angle_err, right_angle_err)
        print(
            f"[camera_mount] step={step} root={_fmt(root_pos)} "
            f"left_local={_fmt(left_local)} right_local={_fmt(right_local)} "
            f"pos_err_m=({left_pos_err:.6f},{right_pos_err:.6f}) "
            f"angle_err_rad=({left_angle_err:.6f},{right_angle_err:.6f})",
            flush=True,
        )
        if step == 0:
            _print_camera_orientation("left", left_local_quat, left_quat_l)
            _print_camera_orientation("right", right_local_quat, right_quat_l)

    print(
        f"[camera_mount] max_pos_err_m={max_pos_err:.6f} max_angle_err_rad={max_angle_err:.6f}",
        flush=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
