"""Collect simulated loading-decision snapshots from a checkpoint or random policy.

The dataset stores dual-camera images, Toyota proprio, low-level geometry
metrics, and an auto-generated binary lift label.  It is intentionally separate
from PPO training so the loading decision can be trained as a supervised
classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Toyota loading-decision dataset")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.forklift_pallet_insert_lift.toyota_pipeline import decision_label_from_metrics
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401


def _snapshot_metrics(raw_env) -> dict[str, torch.Tensor]:
    logs = raw_env.extras.get("log", {}) if hasattr(raw_env, "extras") else {}
    inserted = logs.get("phase/frac_inserted")
    # Step-level logs are scalars, so derive per-env labels from current state
    # using the already maintained env buffers where possible.
    push_free = raw_env._max_pallet_disp_xy_eval < raw_env.cfg.push_free_disp_thresh_m
    dirty = raw_env._ever_dirty_insert_eval
    hold_entry = raw_env._last_hold_entry
    clean = hold_entry & push_free & (~dirty)
    return {
        "inserted": hold_entry.detach().clone(),
        "clean_geometry": clean.detach().clone(),
        "push_free": push_free.detach().clone(),
        "dirty_insert": dirty.detach().clone(),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    policy = None
    if args_cli.checkpoint:
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=raw_env.device)

    obs = wrapped.get_observations()
    images_left = []
    images_right = []
    proprio = []
    actions_all = []
    labels = []
    metrics_rows = []

    for _ in range(args_cli.steps):
        if policy is None:
            actions = torch.zeros((raw_env.num_envs, 2), device=raw_env.device)
        else:
            with torch.inference_mode():
                actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions.detach().clone())

        current_obs = raw_env._get_observations()
        metrics = _snapshot_metrics(raw_env)
        label = decision_label_from_metrics(**metrics)
        images_left.append((current_obs["image_left"].detach().cpu() * 255.0).to(torch.uint8))
        images_right.append((current_obs["image_right"].detach().cpu() * 255.0).to(torch.uint8))
        proprio.append(current_obs["proprio"].detach().cpu())
        actions_all.append(actions.detach().cpu())
        labels.append(label.detach().cpu())
        metrics_rows.append({k: v.detach().cpu() for k, v in metrics.items()})

        if bool(torch.as_tensor(dones).any().item()):
            obs, _ = wrapped.reset()

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_left": torch.cat(images_left, dim=0),
        "image_right": torch.cat(images_right, dim=0),
        "proprio": torch.cat(proprio, dim=0),
        "actions": torch.cat(actions_all, dim=0),
        "label": torch.cat(labels, dim=0),
    }
    torch.save(payload, output_path)
    meta = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "samples": int(payload["label"].numel()),
        "positive_rate": float(payload["label"].float().mean().item()),
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    env.close()
    simulation_app.close()
    print(f"[collect_decision_dataset] wrote {output_path} with {meta['samples']} samples")


if __name__ == "__main__":
    main()
