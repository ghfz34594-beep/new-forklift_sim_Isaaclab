"""Audit the dual-camera visual actor data path before long training.

This script creates the requested visual task, reads one real observation batch,
instantiates the configured VisionActorCritic, and checks every expected tensor
shape in the camera/proprio/actor chain.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit Toyota dual-camera visual actor tensor shapes.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV40Direct-v0",
    help="Visual task to instantiate for the audit.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=20260529)
parser.add_argument("--output", type=str, default=None, help="Optional JSON summary path.")
parser.add_argument("--camera_width", type=int, default=224)
parser.add_argument("--camera_height", type=int, default=224)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(120.0, 55.0, 150.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(120.0, -55.0, 150.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 68.0, -8.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 68.0, 8.0))
parser.add_argument("--dual_camera_hfov_deg", type=float, default=60.0)
parser.add_argument("--camera_far", type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.shape]


def _range(tensor: torch.Tensor) -> list[float]:
    return [float(tensor.min().item()), float(tensor.max().item())]


def _to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _resolve_class(class_name: str):
    module_name, _, attr = class_name.rpartition(".")
    if not module_name:
        raise ValueError(f"Policy class must be fully qualified, got {class_name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _expect(checks: list[dict[str, Any]], name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    checks.append(
        {
            "name": name,
            "actual": list(actual),
            "expected": list(expected),
            "pass": tuple(actual) == tuple(expected),
        }
    )


def _configure_visual_env(env_cfg) -> None:
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    if args.device is not None:
        env_cfg.sim.device = args.device

    env_cfg.action_space = 2
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.use_asymmetric_critic = True
    env_cfg.enable_geo_edge_obs = False
    env_cfg.camera_width = int(args.camera_width)
    env_cfg.camera_height = int(args.camera_height)
    env_cfg.dual_camera_width = int(args.camera_width)
    env_cfg.dual_camera_height = int(args.camera_height)
    env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args.dual_camera_left_pos)
    env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args.dual_camera_right_pos)
    env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args.dual_camera_left_rpy_deg)
    env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args.dual_camera_right_rpy_deg)
    env_cfg.dual_camera_hfov_deg = float(args.dual_camera_hfov_deg)
    env_cfg.dual_camera_far_clip_m = float(args.camera_far)


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg) -> int:
    _configure_visual_env(env_cfg)
    if args.device is not None:
        agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    raw_env = env.unwrapped
    try:
        obs, _ = env.reset()
        zero_action = torch.zeros(
            (int(args.num_envs), int(getattr(raw_env.cfg, "action_space", 2))),
            dtype=torch.float32,
            device=raw_env.device,
        )
        obs, _, _, _, _ = env.step(zero_action)

        policy_cfg = _to_dict(agent_cfg.policy)
        class_name = str(policy_cfg.pop("class_name"))
        policy_cls = _resolve_class(class_name)
        num_actions = int(getattr(raw_env, "num_actions", getattr(raw_env.cfg, "action_space", 2)))
        policy = policy_cls(
            obs=obs,
            obs_groups=_to_dict(agent_cfg.obs_groups),
            num_actions=num_actions,
            **policy_cfg,
        ).to(raw_env.device)
        policy.eval()

        with torch.inference_mode():
            images, proprio = policy._extract_policy_terms(obs)
            encoded_images = []
            image_input_shapes = []
            image_input_ranges = []
            image_normalized_ranges = []
            for image in images:
                image_tensor = policy._ensure_image_tensor(image).float()
                image_input_shapes.append(_shape(image_tensor))
                image_input_ranges.append(_range(image_tensor))
                image_tensor = policy._preprocess_image_for_encoder(image)
                image_normalized_ranges.append(_range(image_tensor))
                encoded_images.append(policy.image_encoder(image_tensor))

            image_concat = torch.cat(encoded_images, dim=-1)
            image_proj = policy.image_proj(image_concat)
            proprio_norm = policy.actor_obs_normalizer(proprio.float())
            proprio_feat = policy.proprio_encoder(proprio_norm)
            actor_input = torch.cat([image_proj, proprio_feat], dim=-1)
            actor_raw = policy.actor(actor_input)
            actor_mean = policy._actor_mean(obs)
            critic_obs = policy.get_critic_obs(obs)
            critic_value = policy.evaluate(obs)

        n = int(args.num_envs)
        h = int(args.camera_height)
        w = int(args.camera_width)
        checks: list[dict[str, Any]] = []
        _expect(checks, "left camera image", tuple(images[0].shape), (n, 3, h, w))
        _expect(checks, "right camera image", tuple(images[1].shape), (n, 3, h, w))
        _expect(checks, "left ResNet34 feature", tuple(encoded_images[0].shape), (n, 512))
        _expect(checks, "right ResNet34 feature", tuple(encoded_images[1].shape), (n, 512))
        _expect(checks, "two-camera concat", tuple(image_concat.shape), (n, 1024))
        _expect(checks, "image projection", tuple(image_proj.shape), (n, 256))
        _expect(checks, "5D Toyota proprio", tuple(proprio.shape), (n, 5))
        _expect(checks, "proprio encoder", tuple(proprio_feat.shape), (n, 128))
        _expect(checks, "actor fused input", tuple(actor_input.shape), (n, 384))
        _expect(checks, "actor raw output", tuple(actor_raw.shape), (n, 2))
        _expect(checks, "actor bounded mean/action", tuple(actor_mean.shape), (n, 2))

        summary = {
            "task": args.task,
            "num_envs": n,
            "seed": int(args.seed),
            "policy_class": class_name,
            "backbone_type": getattr(policy, "image_encoder", None).__class__.__name__,
            "dual_camera": bool(getattr(policy, "dual_camera", False)),
            "action_semantics_project_convention": ["drive_or_reverse", "steer"],
            "camera": {
                "left_pos_local": list(args.dual_camera_left_pos),
                "right_pos_local": list(args.dual_camera_right_pos),
                "left_rpy_deg": list(args.dual_camera_left_rpy_deg),
                "right_rpy_deg": list(args.dual_camera_right_rpy_deg),
                "hfov_deg": float(args.dual_camera_hfov_deg),
                "far_clip_m": float(args.camera_far),
            },
            "observed": {
                "image_input_shapes": image_input_shapes,
                "image_input_ranges": image_input_ranges,
                "image_normalized_ranges": image_normalized_ranges,
                "left_resnet_feature_shape": _shape(encoded_images[0]),
                "right_resnet_feature_shape": _shape(encoded_images[1]),
                "image_concat_shape": _shape(image_concat),
                "image_projection_shape": _shape(image_proj),
                "proprio_shape": _shape(proprio),
                "proprio_feature_shape": _shape(proprio_feat),
                "actor_input_shape": _shape(actor_input),
                "actor_raw_output_shape": _shape(actor_raw),
                "actor_mean_shape": _shape(actor_mean),
                "critic_obs_shape": _shape(critic_obs),
                "critic_value_shape": _shape(critic_value),
            },
            "checks": checks,
            "pass": all(bool(check["pass"]) for check in checks),
            "notes": [
                "Visual actor uses dual camera RGB plus 5D Toyota proprio.",
                "Teacher-only 9D privileged proprio is not part of the visual actor input.",
            ],
        }

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["pass"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
