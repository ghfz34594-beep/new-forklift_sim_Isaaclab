"""Trace a GeoEdge checkpoint on a fixed Stage-1 pose."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Trace GeoEdge policy actions and geometry on fixed reset poses")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--seed", type=int, default=20260427)
parser.add_argument("--fixed_stage1_init", "--fixed-stage1-init", type=float, nargs=3, required=True)
parser.add_argument("--output_csv", "--output-csv", type=str, required=True)
parser.add_argument("--actor_only_load", "--actor-only-load", action="store_true")

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


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _load_checkpoint(runner: OnPolicyRunner, checkpoint: str, actor_only: bool, device: str) -> None:
    if not actor_only:
        runner.load(checkpoint)
        return

    loaded = torch.load(checkpoint, weights_only=False, map_location=device)
    source_state = loaded["model_state_dict"]
    target_state = runner.alg.policy.state_dict()
    compatible_state = {}
    for key, value in source_state.items():
        if key.startswith("critic.") or key.startswith("critic_obs_normalizer."):
            continue
        target_value = target_state.get(key)
        if target_value is not None and tuple(target_value.shape) == tuple(value.shape):
            compatible_state[key] = value
    runner.alg.policy.load_state_dict(compatible_state, strict=False)


def _state(raw_env) -> dict[str, torch.Tensor]:
    pallet_pos = raw_env.pallet.data.root_pos_w
    root_pos = raw_env.robot.data.root_pos_w
    tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()

    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
    yaw_err_deg = yaw_err * (180.0 / math.pi)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / max(float(raw_env._insert_thresh), 1e-6), 0.0, 1.5)

    rel_fc = fork_center[:, :2] - pallet_pos[:, :2]
    center_y = torch.sum(rel_fc * v_lat, dim=-1)
    tip_y = torch.sum(rel_tip * v_lat, dim=-1)
    rel_root = root_pos[:, :2] - pallet_pos[:, :2]
    root_y = torch.sum(rel_root * v_lat, dim=-1)
    pallet_disp = raw_env._pallet_disp_xy()
    return {
        "dist_front": dist_front,
        "insert_norm": insert_norm,
        "center_y": center_y,
        "tip_y": tip_y,
        "root_y": root_y,
        "yaw_err_deg": yaw_err_deg,
        "pallet_disp_xy": pallet_disp,
        "hold_counter": raw_env._hold_counter.clone(),
    }


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
    env_cfg.use_camera = False
    env_cfg.use_asymmetric_critic = False
    env_cfg.enable_geo_edge_obs = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.hold_gate_curriculum_enable = False
    env_cfg.stage1_near_hard_curriculum_enable = False
    env_cfg.stage1_near_hard_curriculum_frac = 0.0
    if hasattr(env_cfg, "teacher_reference_reset_enable"):
        env_cfg.teacher_reference_reset_enable = False
    fixed_x, fixed_y, fixed_yaw_deg = (float(v) for v in args.fixed_stage1_init)
    env_cfg.stage1_init_x_min_m = fixed_x
    env_cfg.stage1_init_x_max_m = fixed_x
    env_cfg.stage1_init_y_min_m = fixed_y
    env_cfg.stage1_init_y_max_m = fixed_y
    env_cfg.stage1_init_yaw_deg_min = fixed_yaw_deg
    env_cfg.stage1_init_yaw_deg_max = fixed_yaw_deg

    if args.device is not None:
        agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_checkpoint(runner, args.checkpoint, args.actor_only_load, raw_env.device)
    policy = runner.get_inference_policy(device=raw_env.device)

    obs = env_wrapped.get_observations()
    rows = []
    for step in range(args.steps):
        state = _state(raw_env)
        with torch.inference_mode():
            actions = policy(obs)
        for env_id in range(args.num_envs):
            rows.append(
                {
                    "label": args.label,
                    "step": step,
                    "env_id": env_id,
                    "action_drive": float(actions[env_id, 0].item()),
                    "action_steer": float(actions[env_id, 1].item()),
                    "applied_drive": float(raw_env.actions[env_id, 0].item()),
                    "applied_steer": float(raw_env.actions[env_id, 1].item()),
                    "dist_front": float(state["dist_front"][env_id].item()),
                    "insert_norm": float(state["insert_norm"][env_id].item()),
                    "center_y": float(state["center_y"][env_id].item()),
                    "tip_y": float(state["tip_y"][env_id].item()),
                    "root_y": float(state["root_y"][env_id].item()),
                    "yaw_err_deg": float(state["yaw_err_deg"][env_id].item()),
                    "pallet_disp_xy": float(state["pallet_disp_xy"][env_id].item()),
                    "hold_counter": float(state["hold_counter"][env_id].item()),
                }
            )
        obs, _, dones, _ = env_wrapped.step(actions.detach().clone())
        if torch.as_tensor(dones, device=raw_env.device).bool().all():
            break

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote trace rows={len(rows)} to {output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
