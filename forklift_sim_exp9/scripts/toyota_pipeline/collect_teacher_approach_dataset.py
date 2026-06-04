"""Collect clean RGB distillation data from the GeoEdge teacher.

The teacher policy observes the 21D geometric edge state.  This script runs it
inside a camera-enabled collection task and records dual RGB images
plus the teacher's clipped [drive, steer] action in the same metadata format
used by ``train_approach_bc.py``.  This is teacher-student distillation, not
human teleop BC.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


VISUAL_TO_COLLECT_TASK = {
    "Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0": (
        "Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectRoom60-v0"
    ),
    "Isaac-Forklift-PalletApproach-ToyotaDualCameraCleanView-v0": (
        "Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectCleanView-v0"
    ),
    "Isaac-Forklift-PalletApproach-ToyotaProgressStudentCleanView-v0": (
        "Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectCleanView-v0"
    ),
    "Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV40Direct-v0": (
        "Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollect-v0"
    ),
    "Isaac-Forklift-PalletApproach-V311LegacyAcceptedTeacherVisualFresh-v0": (
        "Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311LegacyAcceptedVisualCollect-v0"
    ),
    "Isaac-Forklift-PalletApproach-V311LegacyAcceptedTeacherVisualFreshActionGuidanceW01-v0": (
        "Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311LegacyAcceptedVisualCollect-v0"
    ),
}


def _accepted_collect_task(summary_task: str) -> str:
    return str(VISUAL_TO_COLLECT_TASK.get(str(summary_task), str(summary_task)))


parser = argparse.ArgumentParser(description="Collect teacher->RGB student approach dataset")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectCleanView-v0",
    help="Camera-enabled collection task. Use visual isolation before num_envs > 1 RGB collection.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Teacher PPO checkpoint.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument(
    "--target_clean_episodes",
    type=int,
    default=None,
    help="Stop after this many clean episodes are kept. Defaults to --episodes.",
)
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--image_every", type=int, default=1)
parser.add_argument("--flush_every", type=int, default=25)
parser.add_argument("--seed", type=int, default=20260521)
parser.add_argument("--record_failures", action="store_true", help="Keep failed episodes instead of only clean successes.")
parser.add_argument(
    "--hard_lateral_abs_init_y_m",
    type=float,
    default=0.40,
    help="Initial |y| threshold used to tag hard-lateral episodes for visual dataset filtering.",
)
parser.add_argument(
    "--hard_lateral_max_episode_pallet_disp_xy_m",
    type=float,
    default=0.030,
    help="Episode max pallet displacement above which a hard-lateral episode is tagged high-disp.",
)
parser.add_argument(
    "--drop_hard_lateral_high_disp_episodes",
    action="store_true",
    help="Drop otherwise clean hard-lateral episodes if max pallet displacement exceeds the visual-clean threshold.",
)
parser.add_argument("--env_spacing", type=float, default=None, help="Optional override for scene.env_spacing.")
parser.add_argument("--camera_far", type=float, default=None, help="Optional override for dual-camera far clipping range.")
parser.add_argument("--dual_camera_hfov_deg", type=float, default=None, help="Override dual-camera horizontal FoV.")
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"))
parser.add_argument("--relabel_teacher_actions", action="store_true", help="Add clipped/relabelled BC action targets.")
parser.add_argument("--relabel_steer_abs", type=float, default=0.85)
parser.add_argument("--relabel_drive_far_cap", type=float, default=0.70)
parser.add_argument("--relabel_drive_near_dist_m", type=float, default=0.75)
parser.add_argument("--relabel_drive_near_cap", type=float, default=0.35)
parser.add_argument("--relabel_drive_insert_depth_m", type=float, default=0.05)
parser.add_argument("--relabel_drive_insert_cap", type=float, default=0.25)
parser.add_argument("--relabel_near_center_err_m", type=float, default=0.12)
parser.add_argument("--relabel_near_tip_err_m", type=float, default=0.12)
parser.add_argument("--relabel_near_yaw_err_deg", type=float, default=8.0)
parser.add_argument("--relabel_reverse_cap", type=float, default=0.40)
parser.add_argument("--vision_room", action="store_true", default=None, help="Force per-env room occlusion on.")
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room", help="Force per-env room occlusion off.")
parser.add_argument(
    "--vision_acceptance_summary",
    type=str,
    default=None,
    help="Path to a passing visual_isolation_summary.json required when collecting RGB with num_envs > 1.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
if Path(args_cli.output_dir).expanduser().name == "progress_v311_multi_env_clean_v1":
    raise RuntimeError(
        "Refusing to write formal student data into legacy dataset progress_v311_multi_env_clean_v1."
    )
if int(args_cli.num_envs) > 1:
    if int(args_cli.image_every) != 1:
        raise RuntimeError(
            "Formal multi-env RGB student collection requires --image_every 1. "
            "A global step image stride can otherwise sample only a subset of env ids and bias the visual dataset."
        )
    if not bool(args_cli.relabel_teacher_actions):
        raise RuntimeError("Formal multi-env RGB student collection requires --relabel_teacher_actions.")
    if not args_cli.vision_acceptance_summary:
        raise RuntimeError(
            "Refusing multi-env RGB teacher collection without a passing visual acceptance report. "
            "Run validate_room60_visual_isolation.py and pass --vision_acceptance_summary."
        )
    with open(args_cli.vision_acceptance_summary, encoding="utf-8") as f:
        _visual_acceptance_preflight = json.load(f)
    if _visual_acceptance_preflight.get("foreign_leakage_pass") is not True:
        raise RuntimeError(
            f"Visual acceptance summary foreign_leakage_pass is not true: {args_cli.vision_acceptance_summary}"
        )
    if _visual_acceptance_preflight.get("camera_learnability_pass") is not True:
        raise RuntimeError(
            f"Visual acceptance summary camera_learnability_pass is not true: {args_cli.vision_acceptance_summary}"
        )
    if _visual_acceptance_preflight.get("pass") is not True:
        raise RuntimeError(
            f"Visual acceptance summary did not pass both student gates: {args_cli.vision_acceptance_summary}"
        )
    summary_task = str(_visual_acceptance_preflight.get("task", ""))
    accepted_task = _accepted_collect_task(summary_task)
    if accepted_task != str(args_cli.task):
        raise RuntimeError(
            "Visual acceptance summary task does not match this multi-env teacher collection config: "
            f"summary={summary_task} accepted_collect={accepted_task} requested={args_cli.task}"
        )
    if int(_visual_acceptance_preflight.get("num_envs", -1)) != int(args_cli.num_envs):
        raise RuntimeError(
            "Visual acceptance summary num_envs does not match this multi-env teacher collection config: "
            f"summary={_visual_acceptance_preflight.get('num_envs')} requested={args_cli.num_envs}"
        )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import _quat_to_yaw
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401

from rollout_recorder import ToyotaRolloutRecorder


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


def _apply_dual_camera_overrides(env_cfg) -> None:
    if args_cli.dual_camera_hfov_deg is not None:
        env_cfg.dual_camera_hfov_deg = float(args_cli.dual_camera_hfov_deg)
    if args_cli.dual_camera_left_pos is not None:
        env_cfg.dual_camera_left_pos_local = tuple(float(v) for v in args_cli.dual_camera_left_pos)
    if args_cli.dual_camera_right_pos is not None:
        env_cfg.dual_camera_right_pos_local = tuple(float(v) for v in args_cli.dual_camera_right_pos)
    if args_cli.dual_camera_left_rpy_deg is not None:
        env_cfg.dual_camera_left_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_left_rpy_deg)
    if args_cli.dual_camera_right_rpy_deg is not None:
        env_cfg.dual_camera_right_rpy_local_deg = tuple(float(v) for v in args_cli.dual_camera_right_rpy_deg)


def _camera_far_from_cfg(env_cfg) -> float | None:
    if hasattr(env_cfg, "dual_camera_far_clip_m"):
        return float(env_cfg.dual_camera_far_clip_m)
    camera_cfg = getattr(env_cfg, "tiled_camera_left", None)
    if camera_cfg is None:
        return None
    try:
        return float(camera_cfg.spawn.clipping_range[1])
    except Exception:
        return None


def _assert_visual_acceptance_matches(env_cfg, task: str, num_envs: int, summary_path: str | None) -> dict:
    if int(num_envs) <= 1:
        return {}
    if not summary_path:
        raise RuntimeError(
            "Refusing multi-env RGB teacher collection without a passing visual acceptance report. "
            "Run validate_room60_visual_isolation.py and pass --vision_acceptance_summary."
        )

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("foreign_leakage_pass") is not True:
        raise RuntimeError(f"Visual acceptance summary foreign_leakage_pass is not true: {summary_path}")
    if summary.get("camera_learnability_pass") is not True:
        raise RuntimeError(f"Visual acceptance summary camera_learnability_pass is not true: {summary_path}")
    if summary.get("pass") is not True:
        raise RuntimeError(f"Visual acceptance summary did not pass both student gates: {summary_path}")

    mismatches = []
    summary_task = str(summary.get("task", ""))
    accepted_task = _accepted_collect_task(summary_task)
    if accepted_task != str(task):
        mismatches.append(f"task summary={summary_task} accepted_collect={accepted_task} requested={task}")
    if int(summary.get("num_envs", -1)) != int(num_envs):
        mismatches.append(f"num_envs summary={summary.get('num_envs')} requested={num_envs}")
    if abs(float(summary.get("env_spacing", -999.0)) - float(env_cfg.scene.env_spacing)) > 1e-6:
        mismatches.append(
            f"env_spacing summary={summary.get('env_spacing')} requested={float(env_cfg.scene.env_spacing)}"
        )
    if bool(summary.get("vision_room_enable", False)) != bool(getattr(env_cfg, "vision_room_enable", False)):
        mismatches.append(
            "vision_room_enable summary="
            f"{summary.get('vision_room_enable')} requested={bool(getattr(env_cfg, 'vision_room_enable', False))}"
        )

    summary_dual = summary.get("dual_camera_config") or {}
    if abs(float(summary_dual.get("hfov_deg", -999.0)) - float(getattr(env_cfg, "dual_camera_hfov_deg", -1))) > 1e-6:
        mismatches.append(
            f"hfov summary={summary_dual.get('hfov_deg')} requested={float(getattr(env_cfg, 'dual_camera_hfov_deg', -1))}"
        )
    for key, attr in (
        ("left_pos_local", "dual_camera_left_pos_local"),
        ("right_pos_local", "dual_camera_right_pos_local"),
        ("left_rpy_local_deg", "dual_camera_left_rpy_local_deg"),
        ("right_rpy_local_deg", "dual_camera_right_rpy_local_deg"),
    ):
        summary_values = summary_dual.get(key)
        cfg_values = [float(v) for v in getattr(env_cfg, attr, [])]
        if summary_values is None or len(summary_values) != len(cfg_values):
            mismatches.append(f"{key} summary={summary_values} requested={cfg_values}")
            continue
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(summary_values, cfg_values)):
            mismatches.append(f"{key} summary={summary_values} requested={cfg_values}")
    summary_far = summary.get("camera_far", None)
    cfg_far = _camera_far_from_cfg(env_cfg)
    if summary_far is not None and cfg_far is not None and abs(float(summary_far) - float(cfg_far)) > 1e-6:
        mismatches.append(f"camera_far summary={summary_far} requested={cfg_far}")

    if mismatches:
        raise RuntimeError(
            "Visual acceptance summary does not match this multi-env teacher collection config: "
            + "; ".join(mismatches)
        )
    print(f"[teacher_collect] Multi-env visual isolation accepted by {summary_path}", flush=True)
    return summary


def _approach_geometry(raw_env, env_id: int) -> dict[str, float]:
    pallet_pos = raw_env.pallet.data.root_pos_w
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)
    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
    tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()
    u_in = torch.stack([torch.cos(pallet_yaw), torch.sin(pallet_yaw)], dim=-1)
    v_lat = torch.stack([-torch.sin(pallet_yaw), torch.cos(pallet_yaw)], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    rel_center = fork_center[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * float(raw_env.cfg.pallet_depth_m)
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / max(float(raw_env._insert_thresh), 1e-6), 0.0, 1.5)
    center_lateral_err = torch.abs(torch.sum(rel_center * v_lat, dim=-1))
    tip_lateral_err = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
    yaw_err_deg = (
        torch.abs(torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw)))
        * 180.0
        / math.pi
    )
    return {
        "dist_front_m": float(dist_front[env_id].item()),
        "insert_depth_m": float(insert_depth[env_id].item()),
        "center_lateral_err_m": float(center_lateral_err[env_id].item()),
        "tip_lateral_err_m": float(tip_lateral_err[env_id].item()),
        "yaw_err_deg": float(yaw_err_deg[env_id].item()),
        "insert_norm": float(insert_norm[env_id].item()),
    }


def _relabel_action(action: tuple[float, float, float], geom: dict[str, float]) -> tuple[float, float]:
    drive = float(action[0])
    steer = max(-float(args_cli.relabel_steer_abs), min(float(args_cli.relabel_steer_abs), float(action[1])))

    if drive >= 0.0:
        drive = min(drive, float(args_cli.relabel_drive_far_cap))
        if float(geom["dist_front_m"]) <= float(args_cli.relabel_drive_near_dist_m):
            drive = min(drive, float(args_cli.relabel_drive_near_cap))
        if float(geom["insert_depth_m"]) >= float(args_cli.relabel_drive_insert_depth_m):
            drive = min(drive, float(args_cli.relabel_drive_insert_cap))
        near_misaligned = (
            float(geom["dist_front_m"]) <= float(args_cli.relabel_drive_near_dist_m)
            and (
                float(geom["center_lateral_err_m"]) > float(args_cli.relabel_near_center_err_m)
                or float(geom["tip_lateral_err_m"]) > float(args_cli.relabel_near_tip_err_m)
                or float(geom["yaw_err_deg"]) > float(args_cli.relabel_near_yaw_err_deg)
            )
        )
        if near_misaligned:
            drive = 0.0
    else:
        drive = max(drive, -float(args_cli.relabel_reverse_cap))
    return drive, steer


def _state_from_env(raw_env, env_id: int = 0) -> dict:
    root_pos = raw_env.robot.data.root_pos_w[env_id]
    root_lin_vel = raw_env.robot.data.root_lin_vel_w[env_id]
    root_ang_vel = raw_env.robot.data.root_ang_vel_w[env_id]
    tip = raw_env._compute_fork_tip()[env_id]
    pallet_disp = raw_env._pallet_disp_xy()[env_id]
    insert_depth = getattr(raw_env, "_last_insert_depth", torch.zeros(raw_env.num_envs, device=raw_env.device))[env_id]
    done_reason = "running"
    if bool(raw_env._success_termination[env_id].item()):
        done_reason = "success"
    elif bool(raw_env._preinsert_push_termination[env_id].item()):
        done_reason = "preinsert_push"
    elif bool(raw_env._dirty_push_termination[env_id].item()):
        done_reason = "dirty_push"
    geometry = _approach_geometry(raw_env, env_id)
    init_x = getattr(raw_env, "_debug_reset_x", torch.zeros(raw_env.num_envs, device=raw_env.device))[env_id]
    init_y = getattr(raw_env, "_debug_reset_y", torch.zeros(raw_env.num_envs, device=raw_env.device))[env_id]
    init_yaw = getattr(raw_env, "_debug_reset_yaw_deg", torch.zeros(raw_env.num_envs, device=raw_env.device))[env_id]
    return {
        "init_x_m": float(init_x.item()),
        "init_y_m": float(init_y.item()),
        "init_yaw_deg": float(init_yaw.item()),
        "x": float(root_pos[0].item()),
        "y": float(root_pos[1].item()),
        "z": float(root_pos[2].item()),
        "lift_height_m": float((tip[2] - raw_env._fork_tip_z0[env_id]).item()),
        "vx_mps": float(root_lin_vel[0].item()),
        "vy_mps": float(root_lin_vel[1].item()),
        "yaw_rate_radps": float(root_ang_vel[2].item()),
        "lift_joint_m": float(raw_env._joint_pos[env_id, raw_env._lift_id].item()),
        "pallet_disp_xy_m": float(pallet_disp.item()),
        "insert_depth_m": float(insert_depth.item()),
        **geometry,
        "hold_counter": float(raw_env._hold_counter[env_id].item()),
        "push_free": bool((pallet_disp < float(raw_env.cfg.push_free_disp_thresh_m)).item()),
        "done_reason": done_reason,
    }


def _cameras_from_env(raw_env, env_id: int = 0) -> dict[str, torch.Tensor]:
    left, right = raw_env._get_dual_camera_images()
    return {
        "left": left[env_id : env_id + 1].detach().cpu(),
        "right": right[env_id : env_id + 1].detach().cpu(),
    }


def _episode_is_clean(rows: list[dict]) -> bool:
    if not rows:
        return False
    max_insert = max(float(row["insert_depth_m"]) for row in rows)
    max_disp = max(float(row["pallet_disp_xy_m"]) for row in rows)
    return max_insert >= 0.45 and max_disp <= 0.05


def _episode_quality_tags(rows: list[dict]) -> dict[str, float | int | bool]:
    if not rows:
        return {
            "episode_max_pallet_disp_xy_m": 0.0,
            "episode_max_insert_depth_m": 0.0,
            "episode_init_y_m": 0.0,
            "episode_hard_lateral": False,
            "episode_high_pallet_disp": False,
            "episode_hard_lateral_high_disp": False,
        }
    max_disp = max(float(row["pallet_disp_xy_m"]) for row in rows)
    max_insert = max(float(row["insert_depth_m"]) for row in rows)
    init_y = float(rows[0].get("init_y_m", 0.0))
    hard_lateral = abs(init_y) >= float(args_cli.hard_lateral_abs_init_y_m)
    high_disp = max_disp > float(args_cli.hard_lateral_max_episode_pallet_disp_xy_m)
    return {
        "episode_max_pallet_disp_xy_m": max_disp,
        "episode_max_insert_depth_m": max_insert,
        "episode_init_y_m": init_y,
        "episode_hard_lateral": hard_lateral,
        "episode_high_pallet_disp": high_disp,
        "episode_hard_lateral_high_disp": hard_lateral and high_disp,
    }


def _annotate_episode_rows(rows: list[dict], tags: dict[str, float | int | bool], keep_for_visual: bool) -> None:
    for row in rows:
        row.update(tags)
        row["keep_for_visual_training"] = bool(keep_for_visual)


def _episode_rows(rows: list[dict], env_id: int, episode_id: int) -> list[dict]:
    return [
        row
        for row in rows
        if int(row.get("env_id", -1)) == int(env_id) and int(row.get("episode_id", -1)) == int(episode_id)
    ]


def _drop_episode_rows(rows: list[dict], env_id: int, episode_id: int) -> tuple[list[dict], int]:
    kept_rows = []
    dropped = 0
    for row in rows:
        if int(row.get("env_id", -1)) == int(env_id) and int(row.get("episode_id", -1)) == int(episode_id):
            dropped += 1
            continue
        kept_rows.append(row)
    return kept_rows, dropped


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    num_envs = int(args_cli.num_envs)
    target_clean = int(args_cli.target_clean_episodes or args_cli.episodes)
    max_attempts = int(args_cli.episodes)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = int(args_cli.seed)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.geo_edge_record_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.toyota_action_noise_std = 0.0
    env_cfg.toyota_velocity_obs_noise_std = 0.0
    env_cfg.scene.filter_collisions = True
    _apply_dual_camera_overrides(env_cfg)
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    if args_cli.camera_far is not None:
        _set_camera_far(env_cfg, float(args_cli.camera_far))
    if args_cli.vision_room is not None:
        env_cfg.vision_room_enable = bool(args_cli.vision_room)
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    visual_acceptance_summary = _assert_visual_acceptance_matches(
        env_cfg, str(args_cli.task), num_envs, args_cli.vision_acceptance_summary
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)

    recorder = ToyotaRolloutRecorder(
        output_dir,
        save_images=True,
        image_every=int(args_cli.image_every),
        flush_every=int(args_cli.flush_every),
        metadata={
            "source": "geoedge_teacher",
            "task": args_cli.task,
            "teacher_checkpoint": str(args_cli.checkpoint),
            "num_envs": num_envs,
            "target_clean_episodes": target_clean,
            "max_attempted_episodes": max_attempts,
            "env_spacing": float(env_cfg.scene.env_spacing),
            "filter_collisions": bool(getattr(env_cfg.scene, "filter_collisions", True)),
            "vision_room_enable": bool(getattr(env_cfg, "vision_room_enable", False)),
            "vision_acceptance_summary": str(args_cli.vision_acceptance_summary or ""),
            "vision_acceptance_source": str(visual_acceptance_summary.get("source", "")),
            "camera_far": (
                _camera_far_from_cfg(env_cfg)
            ),
            "dual_camera_config": {
                "hfov_deg": float(env_cfg.dual_camera_hfov_deg),
                "left_pos_local": [float(v) for v in env_cfg.dual_camera_left_pos_local],
                "right_pos_local": [float(v) for v in env_cfg.dual_camera_right_pos_local],
                "left_rpy_local_deg": [float(v) for v in env_cfg.dual_camera_left_rpy_local_deg],
                "right_rpy_local_deg": [float(v) for v in env_cfg.dual_camera_right_rpy_local_deg],
            },
            "relabel_teacher_actions": bool(args_cli.relabel_teacher_actions),
            "relabel_config": {
                "steer_abs": float(args_cli.relabel_steer_abs),
                "drive_far_cap": float(args_cli.relabel_drive_far_cap),
                "drive_near_dist_m": float(args_cli.relabel_drive_near_dist_m),
                "drive_near_cap": float(args_cli.relabel_drive_near_cap),
                "drive_insert_depth_m": float(args_cli.relabel_drive_insert_depth_m),
                "drive_insert_cap": float(args_cli.relabel_drive_insert_cap),
                "near_center_err_m": float(args_cli.relabel_near_center_err_m),
                "near_tip_err_m": float(args_cli.relabel_near_tip_err_m),
                "near_yaw_err_deg": float(args_cli.relabel_near_yaw_err_deg),
                "reverse_cap": float(args_cli.relabel_reverse_cap),
            },
            "hard_lateral_filter": {
                "abs_init_y_m": float(args_cli.hard_lateral_abs_init_y_m),
                "max_episode_pallet_disp_xy_m": float(args_cli.hard_lateral_max_episode_pallet_disp_xy_m),
                "drop_hard_lateral_high_disp_episodes": bool(args_cli.drop_hard_lateral_high_disp_episodes),
            },
        },
    )

    kept = 0
    skipped = 0
    global_step = 0
    attempted = 0
    episode_id_by_env = list(range(num_envs))
    next_episode_id = num_envs
    prev_actions = [(0.0, 0.0, 0.0) for _ in range(num_envs)]
    episode_steps = [0 for _ in range(num_envs)]
    dropped_partial_rows = 0
    outer_steps = 0
    loop_exit_reason = "not_started"
    try:
        obs, _ = wrapped.reset()
        print(
            f"[teacher_collect] start target_clean={target_clean} max_attempts={max_attempts} "
            f"num_envs={num_envs} max_steps={int(args_cli.max_steps)}",
            flush=True,
        )

        while kept < target_clean and attempted < max_attempts:
            outer_steps += 1
            with torch.inference_mode():
                action = policy(obs)
            if agent_cfg.clip_actions is not None:
                effective = torch.clamp(action.detach(), -float(agent_cfg.clip_actions), float(agent_cfg.clip_actions))
            else:
                effective = action.detach()

            pre_step = []
            for env_id in range(num_envs):
                pre_step.append(
                    (
                        _state_from_env(raw_env, env_id),
                        _cameras_from_env(raw_env, env_id),
                        (
                            float(effective[env_id, 0].item()),
                            float(effective[env_id, 1].item()),
                            0.0,
                        ),
                    )
                )

            obs, _, dones, _ = wrapped.step(action.detach().clone())
            dones_tensor = torch.as_tensor(dones, device=raw_env.device).bool()
            manual_reset_env_ids = []

            for env_id in range(num_envs):
                if kept >= target_clean or attempted >= max_attempts:
                    print(
                        f"[teacher_collect] inner stop before env={env_id} "
                        f"kept={kept}/{target_clean} attempted={attempted}/{max_attempts}",
                        flush=True,
                    )
                    break
                episode_steps[env_id] += 1
                state, cameras, effective_tuple = pre_step[env_id]
                relabel_drive, relabel_steer = _relabel_action(effective_tuple, state)
                done = bool(dones_tensor[env_id].item()) or episode_steps[env_id] >= int(args_cli.max_steps)
                done_reason = state.get("done_reason", "running")
                if done and done_reason == "running":
                    done_reason = "max_steps" if episode_steps[env_id] >= int(args_cli.max_steps) else "terminated"

                recorder.record_step(
                    step=global_step,
                    env_id=env_id,
                    episode_id=episode_id_by_env[env_id],
                    cameras=cameras,
                    state=state,
                    command=effective_tuple,
                    effective_action=effective_tuple,
                    prev_action=prev_actions[env_id],
                    extra_fields={
                        "action_drive_teacher": float(effective_tuple[0]),
                        "action_steer_teacher": float(effective_tuple[1]),
                        "action_drive_relabel": float(relabel_drive),
                        "action_steer_relabel": float(relabel_steer),
                        "relabel_enabled": bool(args_cli.relabel_teacher_actions),
                        "hard_lateral": bool(
                            abs(float(state.get("init_y_m", 0.0))) >= float(args_cli.hard_lateral_abs_init_y_m)
                        ),
                        "high_pallet_disp": bool(
                            float(state.get("pallet_disp_xy_m", 0.0))
                            > float(args_cli.hard_lateral_max_episode_pallet_disp_xy_m)
                        ),
                    },
                    done=done,
                    done_reason=done_reason,
                )
                prev_actions[env_id] = effective_tuple
                global_step += 1

                if done:
                    current_episode_id = episode_id_by_env[env_id]
                    episode_rows = _episode_rows(recorder.rows, env_id, current_episode_id)
                    clean = _episode_is_clean(episode_rows)
                    quality_tags = _episode_quality_tags(episode_rows)
                    hard_lateral_high_disp = bool(quality_tags["episode_hard_lateral_high_disp"])
                    keep_for_visual = bool(clean or args_cli.record_failures)
                    if bool(args_cli.drop_hard_lateral_high_disp_episodes) and hard_lateral_high_disp:
                        keep_for_visual = False
                    _annotate_episode_rows(episode_rows, quality_tags, keep_for_visual)
                    attempted += 1
                    if keep_for_visual:
                        kept += 1
                    else:
                        recorder.rows, _ = _drop_episode_rows(recorder.rows, env_id, current_episode_id)
                        skipped += 1

                    print(
                        f"[teacher_collect] env={env_id} episode={current_episode_id} "
                        f"clean={int(clean)} hard_lat={int(bool(quality_tags['episode_hard_lateral']))} "
                        f"high_disp={int(bool(quality_tags['episode_high_pallet_disp']))} "
                        f"keep={int(keep_for_visual)} kept={kept} skipped={skipped} attempted={attempted} "
                        f"max_disp={float(quality_tags['episode_max_pallet_disp_xy_m']):.4f} done={done_reason}",
                        flush=True,
                    )

                    episode_id_by_env[env_id] = next_episode_id
                    next_episode_id += 1
                    prev_actions[env_id] = (0.0, 0.0, 0.0)
                    episode_steps[env_id] = 0
                    if not bool(dones_tensor[env_id].item()):
                        manual_reset_env_ids.append(env_id)

            if manual_reset_env_ids:
                reset_ids = torch.tensor(manual_reset_env_ids, device=raw_env.device, dtype=torch.long)
                raw_env._reset_idx(reset_ids)
                if raw_env.sim.has_rtx_sensors() and raw_env.cfg.rerender_on_reset:
                    raw_env.sim.render()
                obs = wrapped.get_observations()
        if kept >= target_clean:
            loop_exit_reason = "target_clean_reached"
        elif attempted >= max_attempts:
            loop_exit_reason = "max_attempts_reached"
        else:
            loop_exit_reason = "while_condition_ended_unexpectedly"
        print(
            f"[teacher_collect] loop exit reason={loop_exit_reason} "
            f"kept={kept}/{target_clean} attempted={attempted}/{max_attempts} outer_steps={outer_steps}",
            flush=True,
        )
    except BaseException as exc:
        loop_exit_reason = f"exception:{type(exc).__name__}"
        print(
            f"[teacher_collect] exception type={type(exc).__name__} repr={exc!r} "
            f"kept={kept}/{target_clean} attempted={attempted}/{max_attempts} outer_steps={outer_steps}",
            flush=True,
        )
        raise
    finally:
        for env_id in range(num_envs):
            if episode_steps[env_id] <= 0:
                continue
            recorder.rows, dropped = _drop_episode_rows(recorder.rows, env_id, episode_id_by_env[env_id])
            dropped_partial_rows += dropped
        recorder.close(
            {
                "kept_episodes": kept,
                "skipped_episodes": skipped,
                "attempted_episodes": attempted,
                "next_episode_id": next_episode_id,
                "dropped_partial_rows": dropped_partial_rows,
                "loop_exit_reason": loop_exit_reason,
                "outer_steps": outer_steps,
            }
        )
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
