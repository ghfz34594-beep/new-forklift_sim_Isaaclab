"""Run approach policy, loading decision, then scripted lift sequence."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Toyota approach -> decision -> scripted lift rollout")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--decision_checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--approach_steps", type=int, default=900)
parser.add_argument("--stop_steps", type=int, default=45)
parser.add_argument("--decision_threshold", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.forklift_pallet_insert_lift.toyota_pipeline import (
    DualCameraLoadingDecisionModel,
    ScriptedLiftSequence,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)

    decision_net = None
    if args_cli.decision_checkpoint:
        decision_net = DualCameraLoadingDecisionModel().to(raw_env.device)
        payload = torch.load(args_cli.decision_checkpoint, map_location=raw_env.device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        decision_net.load_state_dict(state)
        decision_net.eval()

    obs = wrapped.get_observations()
    for _ in range(args_cli.approach_steps):
        with torch.inference_mode():
            action = policy(obs)
        obs, _, dones, _ = wrapped.step(action.detach().clone())
        if bool(torch.as_tensor(dones).any().item()):
            break

    stop = torch.zeros((raw_env.num_envs, 2), device=raw_env.device)
    for _ in range(args_cli.stop_steps):
        obs, _, _, _ = wrapped.step(stop)

    should_lift = torch.ones((raw_env.num_envs,), dtype=torch.bool, device=raw_env.device)
    if decision_net is not None:
        current_obs = raw_env._get_observations()
        with torch.inference_mode():
            logits = decision_net(
                current_obs["image_left"],
                current_obs["image_right"],
                current_obs["proprio"],
            )
            should_lift = torch.sigmoid(logits) >= float(args_cli.decision_threshold)

    print(f"[pipeline] decision should_lift={should_lift.detach().cpu().tolist()}")
    if bool(should_lift.any().item()):
        raw_env._stage_1_mode = False
        sequence = ScriptedLiftSequence.from_env_cfg(raw_env.cfg)
        for step in range(sequence.total_steps):
            action3 = sequence.action_at(step, device=raw_env.device, batch_size=raw_env.num_envs)
            raw_env.step(action3)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
