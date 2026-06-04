"""Smoke checks for Toyota PushSafe API, displacement and push termination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check PushSafe Toyota task and non-web API metrics")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--push_steps", type=int, default=160)
parser.add_argument("--max_initial_disp", type=float, default=1e-3)
parser.add_argument("--keep_action_guard", action="store_true", help="Keep the PushSafe guard during the forced-push check")
parser.add_argument(
    "--check_mode",
    choices=("mask", "rollout"),
    default="mask",
    help="Use direct pallet offsets for deterministic termination checks, or rollout forward actions.",
)
parser.add_argument(
    "--summary_path",
    type=Path,
    default=Path("/data/jianshi/projects/forklift_sim_exp9/outputs/toyota_api_check/pushsafe_api_summary.json"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401

from forklift_api import ForkliftIsaacApi


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    if not args_cli.keep_action_guard and hasattr(env_cfg, "preinsert_action_guard_enable"):
        env_cfg.preinsert_action_guard_enable = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    api = ForkliftIsaacApi(env)
    state0 = api.reset()
    disp0 = raw._pallet_disp_xy().detach().cpu()
    print(f"initial_disp={disp0.tolist()}")
    if float(disp0.max().item()) > float(args_cli.max_initial_disp):
        raise RuntimeError(f"initial pallet displacement too large: max={float(disp0.max().item())}")

    cameras = api.get_cameras()
    print(f"camera_keys={list(cameras.keys())}")
    if "left" not in cameras or "right" not in cameras:
        raise RuntimeError("left/right cameras missing from API")
    print(f"api_state0={state0}")

    push_termination_seen = False
    done_seen = False
    last_state = state0
    if args_cli.check_mode == "mask":
        env_ids = torch.arange(raw.num_envs, device=raw.device)
        pallet_pos = raw.pallet.data.root_pos_w.clone()
        pallet_quat = raw.pallet.data.root_quat_w.clone()
        pallet_pos[:, 0] += float(raw.cfg.preinsert_push_termination_m) + 0.02
        raw._write_root_pose(raw.pallet, pallet_pos, pallet_quat, env_ids)
        raw.episode_length_buf[:] = max(
            int(raw.cfg.preinsert_push_termination_min_steps),
            int(raw.cfg.dirty_push_termination_min_steps),
        )
        raw._preinsert_push_termination = raw._preinsert_push_termination_mask()
        raw._dirty_push_termination = raw._dirty_push_termination_mask()
        last_state = api.get_state()
        push_termination_seen = bool(
            raw._preinsert_push_termination.any().item()
            or raw._dirty_push_termination.any().item()
        )
        done_seen = push_termination_seen
    else:
        for _ in range(int(args_cli.push_steps)):
            action = torch.zeros((raw.num_envs, 2), device=raw.device)
            action[:, 0] = 1.0
            _, _, terminated, truncated, _ = env.step(action)
            last_state = api.get_state()
            push_termination_seen = push_termination_seen or bool(
                raw._preinsert_push_termination.any().item()
                or raw._dirty_push_termination.any().item()
            )
            done_seen = done_seen or bool(
                push_termination_seen
                or torch.as_tensor(terminated | truncated).any().item()
            )
            if push_termination_seen:
                break

    disp1 = raw._pallet_disp_xy().detach().cpu()
    print(f"post_push_disp={disp1.tolist()}")
    print(
        "termination="
        f"preinsert={raw._preinsert_push_termination.detach().cpu().tolist()} "
        f"dirty={raw._dirty_push_termination.detach().cpu().tolist()}"
    )
    print(f"api_last_state={last_state}")
    summary = {
        "task": args_cli.task,
        "num_envs": int(raw.num_envs),
        "action_guard_enabled": bool(getattr(raw.cfg, "preinsert_action_guard_enable", False)),
        "check_mode": args_cli.check_mode,
        "initial_disp": disp0.tolist(),
        "post_push_disp": disp1.tolist(),
        "camera_keys": list(cameras.keys()),
        "done_seen": bool(done_seen),
        "push_termination_seen": bool(push_termination_seen),
        "preinsert_push_termination": raw._preinsert_push_termination.detach().cpu().tolist(),
        "dirty_push_termination": raw._dirty_push_termination.detach().cpu().tolist(),
        "api_state0": dict(state0),
        "api_last_state": dict(last_state),
    }
    args_cli.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args_cli.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not push_termination_seen:
        raise RuntimeError("push termination did not trigger during PushSafe smoke check")
    print("[check_pushsafe_api] PASS", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
