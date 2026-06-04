"""Convert a legacy GeoEdge actor checkpoint into a current trainable checkpoint.

Legacy v2 checkpoints used a 21D actor and a 15D critic. Current GeoEdge training
uses symmetric 21D actor/critic, so direct resume is unsafe. This script loads
only actor-compatible tensors into the current runner and saves a fresh
checkpoint with a newly initialized critic and optimizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert legacy GeoEdge actor checkpoint")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--legacy_checkpoint", type=str, required=True)
parser.add_argument("--output_checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--stage1",
    action="store_true",
    help="Build the runner with Stage A insert-only environment settings.",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _copy_actor_compatible_state(runner: OnPolicyRunner, checkpoint: str, device: str) -> tuple[list[str], list[str]]:
    loaded = torch.load(checkpoint, weights_only=False, map_location=device)
    source_state = loaded["model_state_dict"]
    target_state = runner.alg.policy.state_dict()

    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        if key.startswith("critic.") or key.startswith("critic_obs_normalizer."):
            skipped.append(key)
            continue
        target_value = target_state.get(key)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value

    target_state.update(compatible_state)
    runner.alg.policy.load_state_dict(target_state, strict=True)
    return sorted(compatible_state), sorted(skipped)


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
    env_cfg.use_camera = False
    env_cfg.use_asymmetric_critic = False
    env_cfg.enable_geo_edge_obs = True
    env_cfg.hold_gate_curriculum_enable = False
    env_cfg.tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
    env_cfg.strict_tip_align_entry_m = float(getattr(env_cfg, "strict_tip_align_entry_m", 0.12))
    if args.stage1:
        env_cfg.stage_1_mode = True
        env_cfg.stage1_success_without_lift = True
    if args.device is not None:
        agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    loaded_keys, skipped_keys = _copy_actor_compatible_state(runner, args.legacy_checkpoint, env_wrapped.unwrapped.device)
    runner.current_learning_iteration = 0

    output_checkpoint = Path(args.output_checkpoint).expanduser().resolve()
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "converted_from": str(Path(args.legacy_checkpoint).expanduser().resolve()),
                "conversion": "legacy_geoedge_actor_to_current_trainable",
                "loaded_actor_keys": len(loaded_keys),
                "skipped_keys": len(skipped_keys),
            },
        },
        output_checkpoint,
    )

    report_path = output_checkpoint.with_suffix(".conversion.json")
    report = {
        "legacy_checkpoint": str(Path(args.legacy_checkpoint).expanduser().resolve()),
        "output_checkpoint": str(output_checkpoint),
        "loaded_actor_keys": loaded_keys,
        "skipped_keys": skipped_keys,
        "iter": 0,
        "stage1": bool(args.stage1),
        "num_envs": args.num_envs,
        "seed": args.seed,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("legacy_checkpoint", "output_checkpoint", "iter", "stage1")}, indent=2))
    print(f"[INFO] loaded_actor_keys={len(loaded_keys)} skipped_keys={len(skipped_keys)}")
    print(f"[INFO] wrote conversion report: {report_path}")
    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
