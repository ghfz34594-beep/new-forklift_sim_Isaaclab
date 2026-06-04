"""Visual rollout evaluation for dual-camera approach/insertion checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record visual rollouts for dual-camera approach/insertion checkpoints")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--checkpoint_type",
    choices=("ppo", "bc"),
    default="ppo",
    help="Load a normal RSL-RL checkpoint or actor-compatible BC warm-start checkpoint.",
)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--record_every", type=int, default=3)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--seed", type=int, default=20260521)
parser.add_argument(
    "--no_video",
    action="store_true",
    help="Run rollout statistics without saving PNG frames, MP4 videos, or frame metadata.",
)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
parser.add_argument("--third_person_mode", choices=("topdown", "oblique"), default="topdown")
parser.add_argument("--camera_eye", type=float, nargs=3, default=(-2.7, 0.0, 7.0))
parser.add_argument("--camera_lookat", type=float, nargs=3, default=(-2.7, 0.0, 0.0))
parser.add_argument("--oblique_camera_eye", type=float, nargs=3, default=(-4.0, -6.0, 4.0))
parser.add_argument("--oblique_camera_lookat", type=float, nargs=3, default=(-1.5, 0.0, 0.2))
parser.add_argument("--topdown_camera_eye", type=float, nargs=3, default=(-2.7, 0.0, 7.0))
parser.add_argument("--topdown_camera_lookat", type=float, nargs=3, default=(-2.7, 0.0, 0.0))
parser.add_argument("--env_spacing", type=float, default=None, help="Optional override for scene.env_spacing.")
parser.add_argument("--camera_far", type=float, default=None, help="Optional override for dual-camera far clipping range.")
parser.add_argument("--dual_camera_hfov_deg", type=float, default=None, help="Override dual-camera horizontal FoV.")
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"))
parser.add_argument("--vision_room", action="store_true", default=None, help="Force per-env room occlusion on.")
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room", help="Force per-env room occlusion off.")
parser.add_argument(
    "--vision_room_color",
    type=float,
    nargs=3,
    default=None,
    metavar=("R", "G", "B"),
    help="Override per-env visual-isolation room material color.",
)
parser.add_argument(
    "--vision_room_ceiling",
    action="store_true",
    default=None,
    help="Force the per-env visual-isolation room ceiling on.",
)
parser.add_argument(
    "--no_vision_room_ceiling",
    action="store_false",
    dest="vision_room_ceiling",
    help="Force the per-env visual-isolation room ceiling off for brighter smoke-style recordings.",
)
parser.add_argument(
    "--vision_room_floor",
    action="store_true",
    default=None,
    help="Force the per-env visual-isolation room floor on.",
)
parser.add_argument(
    "--no_vision_room_floor",
    action="store_false",
    dest="vision_room_floor",
    help="Force the per-env visual-isolation room floor off.",
)
parser.add_argument(
    "--fixed_stage1_init",
    type=float,
    nargs=3,
    metavar=("X_M", "Y_M", "YAW_DEG"),
    default=None,
    help="Use one fixed Stage A reset pose instead of sampling x/y/yaw.",
)
parser.add_argument(
    "--disable_teacher_reference_reset",
    action="store_true",
    help="Evaluate from the task's normal Stage A reset distribution instead of V35/V36 teacher-reference starts.",
)
parser.add_argument(
    "--teacher_reference_reset_mix",
    type=float,
    default=None,
    help="Override teacher-reference reset probability for eval by setting both mix_start and mix_end.",
)
parser.add_argument("--baseline_mean_max_insert_depth_m", type=float, default=None)
parser.add_argument("--acceptance_max_raw_drive_sat_frac", type=float, default=0.30)
parser.add_argument("--acceptance_max_env_drive_mean_abs", type=float, default=0.75)
parser.add_argument("--acceptance_max_mean_pallet_disp_xy_m", type=float, default=0.08)
parser.add_argument(
    "--visual_clean_max_pallet_disp_xy_m",
    type=float,
    default=0.030,
    help="Stricter visual-clean displacement threshold used for hard-lateral bucket reporting.",
)
parser.add_argument(
    "--hard_lateral_abs_init_y_m",
    type=float,
    default=0.40,
    help="Initial signed lateral threshold for hard-lateral visual eval buckets.",
)
parser.add_argument("--acceptance_min_insert_rate", type=float, default=0.20)
parser.add_argument("--acceptance_min_depth_improvement_m", type=float, default=0.10)
parser.add_argument(
    "--save_raw_camera_frames",
    action="store_true",
    help="Save raw left/right camera PNGs beside the annotated dual-camera frames.",
)
parser.add_argument(
    "--save_frame_metadata",
    action="store_true",
    help="Write per-recorded-frame pose/action/camera metadata for exact visual replay diagnostics.",
)
parser.add_argument(
    "--record_done_reset_frame",
    action="store_true",
    help=(
        "Record the frame returned by DirectRLEnv after a done step. "
        "By default this reset frame is skipped so videos do not append the next initial pose "
        "to the previous episode."
    ),
)
parser.add_argument(
    "--video_camera_source",
    choices=("synced_render", "policy_obs"),
    default="synced_render",
    help=(
        "Camera images used for recorded videos. synced_render forces a fresh body-mounted camera render "
        "for visual debugging; policy_obs records exactly the actor observation buffer."
    ),
)
parser.add_argument(
    "--visual_envelope_termination",
    action="store_true",
    help="End diagnostic rollouts when the pose leaves the default Wide100 camera visibility envelope.",
)
parser.add_argument("--visual_envelope_center_y_m", type=float, default=None)
parser.add_argument("--visual_envelope_tip_y_m", type=float, default=None)
parser.add_argument("--visual_envelope_yaw_deg", type=float, default=None)
parser.add_argument("--visual_envelope_insert_norm_max", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import _quat_to_yaw
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401


def _to_uint8_hwc(image: Any) -> np.ndarray:
    """Convert a torch/np RGB image into uint8 HWC."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        arr = image.numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _dual_camera_images(raw_env) -> tuple[Any, Any]:
    """Read dual-camera tensors from either visual obs or teacher-collect side cameras."""
    raw_env.scene.write_data_to_sim()
    raw_env.scene.update(dt=0.0)
    if hasattr(raw_env, "_sync_dual_camera_poses"):
        raw_env._sync_dual_camera_poses()
    elif hasattr(raw_env, "_sync_camera_poses_to_robot"):
        raw_env._sync_camera_poses_to_robot()
    if raw_env.sim.has_rtx_sensors():
        raw_env.sim.render()
        if hasattr(raw_env, "_force_dual_camera_update"):
            raw_env._force_dual_camera_update()
        raw_env.scene.update(dt=0.0)
    raw_obs = raw_env._get_observations()
    if "image_left" in raw_obs and "image_right" in raw_obs:
        return raw_obs["image_left"], raw_obs["image_right"]
    if hasattr(raw_env, "_get_dual_camera_images"):
        return raw_env._get_dual_camera_images()
    raise KeyError("Dual camera images are not available from obs or side camera accessors.")


def _dual_camera_images_from_policy_obs(obs: Any) -> tuple[Any, Any] | None:
    """Return the policy-observation camera tensors that the actor just received."""
    try:
        policy_obs = obs["policy"]
        if "image_left" in policy_obs.keys() and "image_right" in policy_obs.keys():
            return policy_obs["image_left"], policy_obs["image_right"]
    except Exception:
        return None
    return None


def _safe_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _draw_text_panel(image: np.ndarray, lines: list[str], panel_h: int = 82) -> np.ndarray:
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + panel_h), (18, 18, 18))
    canvas.paste(Image.fromarray(image), (0, panel_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    y = 6
    for line in lines[:4]:
        draw.text((8, y), line, fill=(235, 235, 235), font=font)
        y += 19
    return np.asarray(canvas)


def _concat_dual_with_divider(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 4, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _sha256_image(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _tensor_list(value: Any, env_id: int | None = None) -> list[float]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if env_id is not None:
            tensor = tensor[env_id]
        return [float(v) for v in tensor.cpu().reshape(-1).tolist()]
    arr = np.asarray(value, dtype=np.float64)
    if env_id is not None:
        arr = arr[env_id]
    return [float(v) for v in arr.reshape(-1).tolist()]


def _camera_pose(raw_env: Any, side: str, env_id: int) -> dict[str, Any]:
    camera = getattr(raw_env, f"_camera_{side}", None)
    if camera is None:
        return {"available": False}
    pose_update_error = None
    try:
        camera._update_poses([int(env_id)])
    except Exception as exc:
        pose_update_error = repr(exc)
    data = camera.data
    result: dict[str, Any] = {
        "available": True,
        "prim_path": None,
        "pos_w": _tensor_list(data.pos_w, env_id) if data.pos_w is not None else None,
        "quat_w_world": _tensor_list(data.quat_w_world, env_id) if data.quat_w_world is not None else None,
        "frame_counter": int(camera.frame[env_id].detach().cpu().item()) if hasattr(camera, "frame") else None,
        "pose_update_error": pose_update_error,
    }
    try:
        result["prim_path"] = str(camera._view.prim_paths[env_id])
    except Exception:
        pass
    try:
        k = data.intrinsic_matrices[env_id].detach().cpu().numpy()
        result["intrinsic_matrix"] = [[float(v) for v in row] for row in k.tolist()]
    except Exception:
        pass
    return result


def _pose_metadata(
    raw_env: Any,
    env_id: int,
    episode: int,
    step: int,
    frame_index: int,
    row: dict[str, Any],
    raw_action: torch.Tensor,
    effective_action: torch.Tensor,
    done_vector: torch.Tensor,
    left: np.ndarray,
    right: np.ndarray,
    image_source: str,
) -> dict[str, Any]:
    root_pos = raw_env.robot.data.root_pos_w
    root_quat = raw_env.robot.data.root_quat_w
    pallet_pos = raw_env.pallet.data.root_pos_w
    pallet_quat = raw_env.pallet.data.root_quat_w
    fork_tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()
    done_cpu = done_vector.detach().cpu().bool().tolist()
    env_origin = raw_env.scene.env_origins[env_id]
    return {
        "episode": int(episode),
        "frame_index": int(frame_index),
        "episode_step": int(step),
        "record_every": int(record_every_global()),
        "env_id": int(env_id),
        "num_envs": int(raw_env.num_envs),
        "done_env0": bool(done_cpu[env_id]),
        "done_any": bool(any(done_cpu)),
        "done_env_ids": [int(i) for i, done in enumerate(done_cpu) if bool(done)],
        "row": row,
        "raw_action_env0": _tensor_list(raw_action, env_id),
        "effective_action_env0": _tensor_list(effective_action, env_id),
        "env_action_env0": _tensor_list(getattr(raw_env, "actions", effective_action), env_id),
        "env_origin": _tensor_list(env_origin),
        "robot": {
            "root_pos_w": _tensor_list(root_pos, env_id),
            "root_quat_w": _tensor_list(root_quat, env_id),
            "yaw_deg": float(row["robot_yaw_deg"]),
        },
        "pallet": {
            "root_pos_w": _tensor_list(pallet_pos, env_id),
            "root_quat_w": _tensor_list(pallet_quat, env_id),
        },
        "fork": {
            "tip_pos_w": _tensor_list(fork_tip, env_id),
            "center_pos_w": _tensor_list(fork_center, env_id),
        },
        "camera": {
            "left": _camera_pose(raw_env, "left", env_id),
            "right": _camera_pose(raw_env, "right", env_id),
            "config": {
                "hfov_deg": float(raw_env.cfg.dual_camera_hfov_deg),
                "near_clip_m": float(getattr(raw_env.cfg, "dual_camera_near_clip_m", 0.1)),
                "far_clip_m": float(getattr(raw_env.cfg, "dual_camera_far_clip_m", 40.0)),
                "left_pos_local": [float(v) for v in raw_env.cfg.dual_camera_left_pos_local],
                "right_pos_local": [float(v) for v in raw_env.cfg.dual_camera_right_pos_local],
                "left_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_left_rpy_local_deg],
                "right_rpy_local_deg": [float(v) for v in raw_env.cfg.dual_camera_right_rpy_local_deg],
                "width": int(raw_env.cfg.dual_camera_width),
                "height": int(raw_env.cfg.dual_camera_height),
            },
        },
        "image": {
            "source": str(image_source),
            "left_shape": [int(v) for v in left.shape],
            "right_shape": [int(v) for v in right.shape],
            "left_sha256": _sha256_image(left),
            "right_sha256": _sha256_image(right),
        },
    }


_RECORD_EVERY_FOR_METADATA = 1


def record_every_global() -> int:
    return int(_RECORD_EVERY_FOR_METADATA)


def _make_video(frame_dir: Path, pattern: str, output_path: Path, fps: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frame_dir / pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(output_path),
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def _fixed_env_camera(raw_env, env_id: int = 0) -> None:
    origin = raw_env.scene.env_origins[env_id].detach().cpu().numpy()
    if str(args_cli.third_person_mode) == "oblique":
        eye_cfg = args_cli.oblique_camera_eye
        lookat_cfg = args_cli.oblique_camera_lookat
    else:
        eye_cfg = args_cli.camera_eye
        lookat_cfg = args_cli.camera_lookat
    eye = origin + np.asarray(eye_cfg, dtype=np.float32)
    lookat = origin + np.asarray(lookat_cfg, dtype=np.float32)
    raw_env.sim.set_camera_view(eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in lookat))


def _fixed_topdown_camera(raw_env, env_id: int = 0) -> None:
    origin = raw_env.scene.env_origins[env_id].detach().cpu().numpy()
    eye = origin + np.asarray(args_cli.topdown_camera_eye, dtype=np.float32)
    lookat = origin + np.asarray(args_cli.topdown_camera_lookat, dtype=np.float32)
    raw_env.sim.set_camera_view(eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in lookat))


def _set_camera_far(env_cfg: Any, far: float) -> None:
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


def _apply_dual_camera_overrides(env_cfg: Any) -> None:
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


def _apply_visual_envelope_overrides(env_cfg: Any) -> None:
    if not bool(args_cli.visual_envelope_termination):
        return
    env_cfg.visual_envelope_termination_enable = True
    if args_cli.visual_envelope_center_y_m is not None:
        env_cfg.visual_envelope_center_y_m = float(args_cli.visual_envelope_center_y_m)
    if args_cli.visual_envelope_tip_y_m is not None:
        env_cfg.visual_envelope_tip_y_m = float(args_cli.visual_envelope_tip_y_m)
    if args_cli.visual_envelope_yaw_deg is not None:
        env_cfg.visual_envelope_yaw_deg = float(args_cli.visual_envelope_yaw_deg)
    if args_cli.visual_envelope_insert_norm_max is not None:
        env_cfg.visual_envelope_insert_norm_max = float(args_cli.visual_envelope_insert_norm_max)


def _camera_signature(env_cfg: Any) -> dict[str, Any]:
    signature = {
        "camera_version": str(getattr(env_cfg, "camera_version", "unspecified")),
        "hfov_deg": float(getattr(env_cfg, "dual_camera_hfov_deg", 0.0)),
        "far_clip_m": float(getattr(env_cfg, "dual_camera_far_clip_m", 0.0)),
        "left_pos_local": [float(v) for v in getattr(env_cfg, "dual_camera_left_pos_local", ())],
        "right_pos_local": [float(v) for v in getattr(env_cfg, "dual_camera_right_pos_local", ())],
        "left_rpy_local_deg": [float(v) for v in getattr(env_cfg, "dual_camera_left_rpy_local_deg", ())],
        "right_rpy_local_deg": [float(v) for v in getattr(env_cfg, "dual_camera_right_rpy_local_deg", ())],
        "vision_room_enable": bool(getattr(env_cfg, "vision_room_enable", False)),
    }
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    signature["config_hash_sha1"] = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return signature


def _metrics(
    raw_env,
    raw_action: torch.Tensor,
    effective_action: torch.Tensor,
    step: int,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> dict[str, float | int | bool]:
    env_id = 0
    root_pos = raw_env.robot.data.root_pos_w
    root_quat = raw_env.robot.data.root_quat_w
    pallet_pos = raw_env.pallet.data.root_pos_w
    pallet_quat = raw_env.pallet.data.root_quat_w

    yaw = _quat_to_yaw(root_quat)
    pallet_yaw = _quat_to_yaw(pallet_quat)
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    tip = raw_env._compute_fork_tip()
    center = raw_env._compute_fork_center()
    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    rel_center = center[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * float(raw_env.cfg.pallet_depth_m)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / (float(raw_env.cfg.pallet_depth_m) + 1e-6), 0.0, 1.0)
    tip_y_err = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
    center_y_err = torch.abs(torch.sum(rel_center * v_lat, dim=-1))
    yaw_err_deg = torch.abs(torch.atan2(torch.sin(yaw - pallet_yaw), torch.cos(yaw - pallet_yaw))) * 180.0 / math.pi
    signed_lateral_err = torch.sum((root_pos[:, :2] - pallet_pos[:, :2]) * v_lat, dim=-1)
    signed_tip_lateral_err = torch.sum(rel_tip * v_lat, dim=-1)
    signed_center_lateral_err = torch.sum(rel_center * v_lat, dim=-1)
    yaw_err_signed_deg = torch.atan2(torch.sin(yaw - pallet_yaw), torch.cos(yaw - pallet_yaw)) * 180.0 / math.pi

    if hasattr(raw_env, "_pallet_disp_xy"):
        pallet_disp_xy = raw_env._pallet_disp_xy()
    else:
        pallet_init_xy = torch.tensor(raw_env.cfg.pallet_cfg.init_state.pos[:2], device=raw_env.device)
        origin_xy = raw_env.scene.env_origins[:, :2]
        pallet_disp_xy = torch.norm(pallet_pos[:, :2] - (origin_xy + pallet_init_xy), dim=-1)
    push_free = pallet_disp_xy < float(raw_env.cfg.push_free_disp_thresh_m)
    inserted = insert_depth >= float(raw_env._insert_thresh)
    dirty_insert = inserted & (~push_free)

    success = getattr(raw_env, "_success_termination", torch.zeros(raw_env.num_envs, device=raw_env.device, dtype=torch.bool))
    pre_push = getattr(raw_env, "_preinsert_push_termination", torch.zeros_like(success))
    dirty_push = getattr(raw_env, "_dirty_push_termination", torch.zeros_like(success))
    visual_envelope = getattr(raw_env, "_visual_envelope_termination", torch.zeros_like(success))
    last_reset_visual_envelope = getattr(raw_env, "_last_reset_visual_envelope_termination", torch.zeros_like(success))
    visual_envelope = visual_envelope | last_reset_visual_envelope
    max_disp = getattr(raw_env, "_max_pallet_disp_xy_eval", pallet_disp_xy)
    hold_counter = getattr(raw_env, "_hold_counter", torch.zeros(raw_env.num_envs, device=raw_env.device))
    env_action = getattr(raw_env, "actions", effective_action)
    if env_action.shape[1] < 2:
        env_action = effective_action

    return {
        "step": int(step),
        "raw_drive": _safe_float(raw_action[env_id, 0]),
        "raw_steer": _safe_float(raw_action[env_id, 1]),
        "drive": _safe_float(effective_action[env_id, 0]),
        "steer": _safe_float(effective_action[env_id, 1]),
        "env_drive": _safe_float(env_action[env_id, 0]),
        "env_steer": _safe_float(env_action[env_id, 1]),
        "speed_xy_mps": _safe_float(torch.linalg.norm(raw_env.robot.data.root_lin_vel_w[env_id, :2])),
        "yaw_rate_dps": _safe_float(raw_env.robot.data.root_ang_vel_w[env_id, 2] * 180.0 / math.pi),
        "reward": _safe_float(reward[env_id]),
        "done": bool(done[env_id].detach().cpu().item()),
        "success": bool(success[env_id].detach().cpu().item()),
        "preinsert_push_done": bool(pre_push[env_id].detach().cpu().item()),
        "dirty_push_done": bool(dirty_push[env_id].detach().cpu().item()),
        "visual_envelope_done": bool(visual_envelope[env_id].detach().cpu().item()),
        "robot_x": _safe_float(root_pos[env_id, 0]),
        "robot_y": _safe_float(root_pos[env_id, 1]),
        "robot_yaw_deg": _safe_float(yaw[env_id] * 180.0 / math.pi),
        "signed_lateral_err_m": _safe_float(signed_lateral_err[env_id]),
        "pallet_disp_xy_m": _safe_float(pallet_disp_xy[env_id]),
        "max_pallet_disp_xy_m": _safe_float(max_disp[env_id]),
        "insert_depth_m": _safe_float(insert_depth[env_id]),
        "insert_norm": _safe_float(insert_norm[env_id]),
        "tip_lateral_err_m": _safe_float(tip_y_err[env_id]),
        "center_lateral_err_m": _safe_float(center_y_err[env_id]),
        "yaw_err_deg": _safe_float(yaw_err_deg[env_id]),
        "tip_lateral_signed_m": _safe_float(signed_tip_lateral_err[env_id]),
        "center_lateral_signed_m": _safe_float(signed_center_lateral_err[env_id]),
        "yaw_err_signed_deg": _safe_float(yaw_err_signed_deg[env_id]),
        "hold_counter": _safe_float(hold_counter[env_id]),
        "inserted": bool(inserted[env_id].detach().cpu().item()),
        "dirty_insert": bool(dirty_insert[env_id].detach().cpu().item()),
        "push_free": bool(push_free[env_id].detach().cpu().item()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_bc_warm_start(policy: torch.nn.Module, checkpoint: str, device: torch.device | str) -> tuple[int, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source_state = payload.get("model_state_dict", payload)
    target_state = policy.state_dict()
    compatible_state = {}
    skipped = []
    skip_prefixes = ("critic.", "critic_obs_normalizer.", "std", "log_std")
    for key, value in source_state.items():
        if key.startswith(skip_prefixes):
            skipped.append(key)
            continue
        target_value = target_state.get(key)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value
    policy.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), len(skipped)


def _summarize_episode(rows: list[dict[str, Any]], episode: int, done_reason: str) -> dict[str, Any]:
    if not rows:
        return {"episode": episode, "steps": 0, "done_reason": done_reason}
    last = rows[-1]
    first = rows[0]
    max_pallet_disp = max(float(row["max_pallet_disp_xy_m"]) for row in rows)
    ever_success = any(bool(row["success"]) for row in rows)
    hard_lateral = abs(float(first["signed_lateral_err_m"])) >= float(args_cli.hard_lateral_abs_init_y_m)
    return {
        "episode": episode,
        "steps": len(rows),
        "done_reason": done_reason,
        "init_signed_lateral_err_m": first["signed_lateral_err_m"],
        "init_tip_lateral_signed_m": first["tip_lateral_signed_m"],
        "init_center_lateral_signed_m": first["center_lateral_signed_m"],
        "init_yaw_err_signed_deg": first["yaw_err_signed_deg"],
        "final_insert_depth_m": last["insert_depth_m"],
        "final_insert_norm": last["insert_norm"],
        "final_tip_lateral_err_m": last["tip_lateral_err_m"],
        "final_center_lateral_err_m": last["center_lateral_err_m"],
        "final_yaw_err_deg": last["yaw_err_deg"],
        "final_pallet_disp_xy_m": last["pallet_disp_xy_m"],
        "max_pallet_disp_xy_m": max_pallet_disp,
        "max_insert_depth_m": max(float(row["insert_depth_m"]) for row in rows),
        "ever_inserted": any(bool(row["inserted"]) for row in rows),
        "ever_dirty_insert": any(bool(row["dirty_insert"]) for row in rows),
        "ever_success": ever_success,
        "visual_clean_success": ever_success and max_pallet_disp <= float(args_cli.visual_clean_max_pallet_disp_xy_m),
        "hard_lateral": hard_lateral,
        "hard_lateral_high_disp": hard_lateral and max_pallet_disp > float(args_cli.visual_clean_max_pallet_disp_xy_m),
        "mean_abs_drive": float(np.mean([abs(float(row["drive"])) for row in rows])),
        "mean_abs_steer": float(np.mean([abs(float(row["steer"])) for row in rows])),
        "mean_raw_drive": float(np.mean([float(row["raw_drive"]) for row in rows])),
        "mean_raw_steer": float(np.mean([float(row["raw_steer"]) for row in rows])),
        "mean_env_drive": float(np.mean([float(row["env_drive"]) for row in rows])),
        "mean_env_steer": float(np.mean([float(row["env_steer"]) for row in rows])),
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "mean_abs": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "neg_frac": 0.0,
            "pos_frac": 0.0,
            "near_zero_frac": 0.0,
            "sat_abs_gt_095_frac": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "neg_frac": float(np.mean(arr < -1e-6)),
        "pos_frac": float(np.mean(arr > 1e-6)),
        "near_zero_frac": float(np.mean(np.abs(arr) <= 0.05)),
        "sat_abs_gt_095_frac": float(np.mean(np.abs(arr) > 0.95)),
    }


def _action_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        key: _stats([float(row[key]) for row in rows if key in row])
        for key in ("raw_drive", "raw_steer", "drive", "steer", "env_drive", "env_steer")
    }


def _group_summaries(all_summaries: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for group_name, pred in (
        ("init_y_negative", lambda value: value < 0.0),
        ("init_y_nonnegative", lambda value: value >= 0.0),
        ("hard_lateral_abs_y_ge_threshold", lambda value: abs(value) >= float(args_cli.hard_lateral_abs_init_y_m)),
    ):
        eps = [ep for ep in all_summaries if pred(float(ep.get("init_signed_lateral_err_m", 0.0)))]
        ep_ids = {int(ep.get("episode", -1)) for ep in eps}
        rows = [row for row in all_rows if int(row.get("episode", -1)) in ep_ids]
        actions = _action_stats(rows)
        groups[group_name] = {
            "num_episodes": len(eps),
            "success_rate": float(np.mean([bool(ep.get("ever_success", False)) for ep in eps])) if eps else 0.0,
            "insert_rate": float(np.mean([bool(ep.get("ever_inserted", False)) for ep in eps])) if eps else 0.0,
            "dirty_insert_rate": float(np.mean([bool(ep.get("ever_dirty_insert", False)) for ep in eps])) if eps else 0.0,
            "visual_clean_success_rate": float(np.mean([bool(ep.get("visual_clean_success", False)) for ep in eps]))
            if eps
            else 0.0,
            "hard_lateral_high_disp_rate": float(np.mean([bool(ep.get("hard_lateral_high_disp", False)) for ep in eps]))
            if eps
            else 0.0,
            "mean_init_signed_lateral_err_m": float(
                np.mean([float(ep.get("init_signed_lateral_err_m", 0.0)) for ep in eps])
            )
            if eps
            else 0.0,
            "mean_max_insert_depth_m": float(np.mean([float(ep.get("max_insert_depth_m", 0.0)) for ep in eps]))
            if eps
            else 0.0,
            "mean_max_pallet_disp_xy_m": float(
                np.mean([float(ep.get("max_pallet_disp_xy_m", 0.0)) for ep in eps])
            )
            if eps
            else 0.0,
            "actions": actions,
        }
    return groups


def _steer_collapse(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    neg = groups.get("init_y_negative", {})
    pos = groups.get("init_y_nonnegative", {})
    neg_steer = float(neg.get("actions", {}).get("env_steer", {}).get("mean", 0.0))
    pos_steer = float(pos.get("actions", {}).get("env_steer", {}).get("mean", 0.0))
    enough_groups = int(neg.get("num_episodes", 0)) > 0 and int(pos.get("num_episodes", 0)) > 0
    same_sign = enough_groups and abs(neg_steer) > 0.05 and abs(pos_steer) > 0.05 and (neg_steer * pos_steer) > 0.0
    return {
        "same_sign_steer_collapse": bool(same_sign),
        "init_y_negative_mean_env_steer": neg_steer,
        "init_y_nonnegative_mean_env_steer": pos_steer,
        "has_both_initial_y_groups": bool(enough_groups),
    }


def _acceptance(aggregate: dict[str, Any]) -> dict[str, Any]:
    actions = aggregate.get("actions", {})
    raw_drive_sat = float(actions.get("raw_drive", {}).get("sat_abs_gt_095_frac", 0.0))
    env_drive_mean_abs = float(actions.get("env_drive", {}).get("mean_abs", 0.0))
    mean_disp = float(aggregate.get("aggregate", {}).get("mean_max_pallet_disp_xy_m", 0.0))
    insert_rate = float(aggregate.get("aggregate", {}).get("insert_rate", 0.0))
    mean_depth = float(aggregate.get("aggregate", {}).get("mean_max_insert_depth_m", 0.0))
    baseline_depth = args_cli.baseline_mean_max_insert_depth_m
    depth_improvement = None if baseline_depth is None else float(mean_depth - float(baseline_depth))
    steer = aggregate.get("steer_health", {})

    drive_sat_pass = raw_drive_sat < float(args_cli.acceptance_max_raw_drive_sat_frac)
    env_drive_pass = env_drive_mean_abs < float(args_cli.acceptance_max_env_drive_mean_abs)
    steer_pass = bool(steer.get("has_both_initial_y_groups", False)) and not bool(
        steer.get("same_sign_steer_collapse", True)
    )
    displacement_pass = mean_disp <= float(args_cli.acceptance_max_mean_pallet_disp_xy_m)
    insertion_pass = insert_rate >= float(args_cli.acceptance_min_insert_rate) or (
        depth_improvement is not None and depth_improvement >= float(args_cli.acceptance_min_depth_improvement_m)
    )
    return {
        "pass": bool(drive_sat_pass and env_drive_pass and steer_pass and displacement_pass and insertion_pass),
        "raw_drive_saturation_pass": bool(drive_sat_pass),
        "env_drive_mean_abs_pass": bool(env_drive_pass),
        "steer_health_pass": bool(steer_pass),
        "pallet_displacement_pass": bool(displacement_pass),
        "insertion_or_depth_pass": bool(insertion_pass),
        "raw_drive_sat_abs_gt_095_frac": raw_drive_sat,
        "env_drive_mean_abs": env_drive_mean_abs,
        "mean_max_pallet_disp_xy_m": mean_disp,
        "insert_rate": insert_rate,
        "mean_max_insert_depth_m": mean_depth,
        "baseline_mean_max_insert_depth_m": None if baseline_depth is None else float(baseline_depth),
        "depth_improvement_m": depth_improvement,
            "thresholds": {
                "max_raw_drive_sat_frac": float(args_cli.acceptance_max_raw_drive_sat_frac),
                "max_env_drive_mean_abs": float(args_cli.acceptance_max_env_drive_mean_abs),
                "max_mean_pallet_disp_xy_m": float(args_cli.acceptance_max_mean_pallet_disp_xy_m),
                "visual_clean_max_pallet_disp_xy_m": float(args_cli.visual_clean_max_pallet_disp_xy_m),
                "hard_lateral_abs_init_y_m": float(args_cli.hard_lateral_abs_init_y_m),
                "min_insert_rate": float(args_cli.acceptance_min_insert_rate),
                "min_depth_improvement_m": float(args_cli.acceptance_min_depth_improvement_m),
            },
    }


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = int(args_cli.seed)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = True
    env_cfg.stage1_success_without_lift = True
    env_cfg.viewer.resolution = (int(args_cli.video_width), int(args_cli.video_height))
    env_cfg.scene.filter_collisions = True
    _apply_dual_camera_overrides(env_cfg)
    _apply_visual_envelope_overrides(env_cfg)
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    if args_cli.camera_far is not None:
        _set_camera_far(env_cfg, float(args_cli.camera_far))
    if args_cli.vision_room is not None:
        env_cfg.vision_room_enable = bool(args_cli.vision_room)
    if args_cli.vision_room_color is not None:
        env_cfg.vision_room_color = tuple(float(v) for v in args_cli.vision_room_color)
    if args_cli.vision_room_ceiling is not None:
        env_cfg.vision_room_ceiling_enable = bool(args_cli.vision_room_ceiling)
    if args_cli.vision_room_floor is not None:
        env_cfg.vision_room_floor_enable = bool(args_cli.vision_room_floor)
    if bool(args_cli.disable_teacher_reference_reset):
        env_cfg.teacher_reference_reset_enable = False
    if args_cli.teacher_reference_reset_mix is not None:
        mix = max(0.0, min(1.0, float(args_cli.teacher_reference_reset_mix)))
        env_cfg.teacher_reference_reset_enable = mix > 0.0
        env_cfg.teacher_reference_reset_mix_start = mix
        env_cfg.teacher_reference_reset_mix_end = mix
    if args_cli.fixed_stage1_init is not None:
        fixed_x, fixed_y, fixed_yaw_deg = (float(v) for v in args_cli.fixed_stage1_init)
        env_cfg.stage1_init_x_min_m = fixed_x
        env_cfg.stage1_init_x_max_m = fixed_x
        env_cfg.stage1_init_y_min_m = fixed_y
        env_cfg.stage1_init_y_max_m = fixed_y
        env_cfg.stage1_init_yaw_deg_min = fixed_yaw_deg
        env_cfg.stage1_init_yaw_deg_max = fixed_yaw_deg
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    render_mode = None if bool(args_cli.no_video) else "rgb_array"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    if not bool(args_cli.no_video):
        _fixed_env_camera(raw_env)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args_cli.checkpoint_type == "ppo":
        runner.load(args_cli.checkpoint)
    else:
        loaded, skipped = _load_bc_warm_start(runner.alg.policy, args_cli.checkpoint, raw_env.device)
        print(f"[bc] loaded {loaded} actor tensors, skipped {skipped} tensors from {args_cli.checkpoint}", flush=True)
    policy = runner.get_inference_policy(device=raw_env.device)

    all_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    record_every = max(1, int(args_cli.record_every))
    global _RECORD_EVERY_FOR_METADATA
    _RECORD_EVERY_FOR_METADATA = record_every

    for episode in range(int(args_cli.episodes)):
        obs, _ = wrapped.reset()
        if not bool(args_cli.no_video):
            _fixed_env_camera(raw_env)
        ep_dir = output_dir / f"episode_{episode:03d}"
        global_dir = ep_dir / "global_frames"
        topdown_dir = ep_dir / "topdown_frames"
        dual_dir = ep_dir / "dual_camera_frames"
        raw_left_dir = ep_dir / "left_raw_frames"
        raw_right_dir = ep_dir / "right_raw_frames"
        rows: list[dict[str, Any]] = []
        frame_metadata_rows: list[dict[str, Any]] = []
        done_reason = "max_steps"
        skipped_done_reset_frame = False
        saved_frames = 0

        for step in range(int(args_cli.steps)):
            with torch.inference_mode():
                action = policy(obs)
            if agent_cfg.clip_actions is not None:
                effective_action = torch.clamp(
                    action.detach().clone(),
                    -float(agent_cfg.clip_actions),
                    float(agent_cfg.clip_actions),
                )
            else:
                effective_action = action.detach().clone()
            obs, reward, dones, _ = wrapped.step(action.detach().clone())
            done_bool = dones.bool()

            # DirectRLEnv resets finished envs before returning obs/raw state.
            # Recording this post-done state appends the next initial pose to the
            # previous episode, which looks like a body-mounted camera teleport.
            if bool(done_bool.any().item()) and not bool(args_cli.record_done_reset_frame):
                if bool(getattr(raw_env, "reset_time_outs", torch.zeros_like(done_bool)).bool().any().item()):
                    done_reason = "timeout"
                else:
                    done_reason = "terminated_or_truncated"
                skipped_done_reset_frame = True
                break

            row = _metrics(raw_env, action, effective_action, step, reward, done_bool)
            row["episode"] = int(episode)
            rows.append(row)
            all_rows.append(row)

            if not bool(args_cli.no_video) and step % record_every == 0:
                label_lines = [
                    (
                        f"ep={episode} step={step} drive={row['drive']:+.2f} steer={row['steer']:+.2f} "
                        f"raw=({row['raw_drive']:+.1f},{row['raw_steer']:+.1f}) "
                        f"done={int(row['done'])} success={int(row['success'])} "
                        f"visual_env={int(row['visual_envelope_done'])}"
                    ),
                    (
                        f"insert={row['insert_depth_m']:.3f}m norm={row['insert_norm']:.2f} "
                        f"pallet_disp={row['pallet_disp_xy_m']:.3f}m max={row['max_pallet_disp_xy_m']:.3f}m"
                    ),
                    (
                        f"tip_y={row['tip_lateral_err_m']:.3f}m center_y={row['center_lateral_err_m']:.3f}m "
                        f"yaw_err={row['yaw_err_deg']:.1f}deg"
                    ),
                    (
                        f"speed={row['speed_xy_mps']:.3f}m/s yaw_rate={row['yaw_rate_dps']:.1f}deg/s "
                        f"done_reset_frame_skipped_by_default={int(not bool(args_cli.record_done_reset_frame))}"
                    ),
                ]
                policy_camera_images = None
                if str(args_cli.video_camera_source) == "policy_obs":
                    policy_camera_images = _dual_camera_images_from_policy_obs(obs)
                if policy_camera_images is None:
                    image_left, image_right = _dual_camera_images(raw_env)
                    image_source = "synced_render"
                else:
                    image_left, image_right = policy_camera_images
                    image_source = "policy_obs"
                left = _to_uint8_hwc(image_left)
                right = _to_uint8_hwc(image_right)
                if bool(args_cli.save_raw_camera_frames):
                    _save_png(raw_left_dir / f"frame_{saved_frames:06d}.png", left)
                    _save_png(raw_right_dir / f"frame_{saved_frames:06d}.png", right)
                if bool(args_cli.save_frame_metadata):
                    meta = _pose_metadata(
                        raw_env,
                        0,
                        episode,
                        step,
                        saved_frames,
                        row,
                        action,
                        effective_action,
                        done_bool,
                        left,
                        right,
                        image_source,
                    )
                    if bool(args_cli.save_raw_camera_frames):
                        meta["image"]["left_raw_path"] = str(raw_left_dir / f"frame_{saved_frames:06d}.png")
                        meta["image"]["right_raw_path"] = str(raw_right_dir / f"frame_{saved_frames:06d}.png")
                    frame_metadata_rows.append(
                        meta
                    )
                dual = _concat_dual_with_divider(left, right)
                dual = _draw_text_panel(dual, label_lines)
                _save_png(dual_dir / f"frame_{saved_frames:06d}.png", dual)

                global_frame = env.render()
                if global_frame is not None:
                    global_img = _draw_text_panel(_to_uint8_hwc(global_frame), label_lines)
                    _save_png(global_dir / f"frame_{saved_frames:06d}.png", global_img)
                _fixed_topdown_camera(raw_env)
                topdown_frame = env.render()
                if topdown_frame is not None:
                    topdown_img = _draw_text_panel(_to_uint8_hwc(topdown_frame), label_lines)
                    _save_png(topdown_dir / f"frame_{saved_frames:06d}.png", topdown_img)
                _fixed_env_camera(raw_env)
                saved_frames += 1

            if bool(done_bool.any().item()):
                if bool(row["success"]):
                    done_reason = "success"
                elif bool(row["preinsert_push_done"]):
                    done_reason = "preinsert_push"
                elif bool(row["dirty_push_done"]):
                    done_reason = "dirty_push"
                elif bool(row["visual_envelope_done"]):
                    done_reason = "visual_envelope"
                else:
                    done_reason = "terminated_or_truncated"
                break

        _write_csv(ep_dir / "metrics.csv", rows)
        if frame_metadata_rows:
            meta_path = ep_dir / "frame_meta.jsonl"
            with meta_path.open("w", encoding="utf-8") as f:
                for meta in frame_metadata_rows:
                    f.write(json.dumps(meta, sort_keys=True) + "\n")
        if not bool(args_cli.no_video) and (dual_dir / "frame_000000.png").is_file():
            _make_video(dual_dir, "frame_%06d.png", ep_dir / "dual_camera.mp4", int(args_cli.fps))
        if not bool(args_cli.no_video) and (global_dir / "frame_000000.png").is_file():
            _make_video(global_dir, "frame_%06d.png", ep_dir / "global.mp4", int(args_cli.fps))
        if not bool(args_cli.no_video) and (topdown_dir / "frame_000000.png").is_file():
            _make_video(
                topdown_dir,
                "frame_%06d.png",
                ep_dir / "kinematic_check_topdown.mp4",
                int(args_cli.fps),
            )

        summary = _summarize_episode(rows, episode, done_reason)
        summary["dual_camera_video"] = None if bool(args_cli.no_video) else str(ep_dir / "dual_camera.mp4")
        summary["global_video"] = None if bool(args_cli.no_video) else str(ep_dir / "global.mp4")
        summary["kinematic_check_topdown_video"] = (
            None if bool(args_cli.no_video) else str(ep_dir / "kinematic_check_topdown.mp4")
        )
        summary["metrics_csv"] = str(ep_dir / "metrics.csv")
        summary["frame_metadata_jsonl"] = str(ep_dir / "frame_meta.jsonl") if frame_metadata_rows else None
        summary["raw_left_frames_dir"] = str(raw_left_dir) if bool(args_cli.save_raw_camera_frames) else None
        summary["raw_right_frames_dir"] = str(raw_right_dir) if bool(args_cli.save_raw_camera_frames) else None
        summary["done_reset_frame_skipped"] = bool(skipped_done_reset_frame)
        summary["videos_present"] = {
            "dual_camera": (ep_dir / "dual_camera.mp4").is_file(),
            "global": (ep_dir / "global.mp4").is_file(),
            "kinematic_check_topdown": (ep_dir / "kinematic_check_topdown.mp4").is_file(),
        }
        (ep_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        all_summaries.append(summary)
        print(f"[episode {episode}] {json.dumps(summary, sort_keys=True)}", flush=True)

    action_stats = _action_stats(all_rows)
    initial_y_groups = _group_summaries(all_summaries, all_rows)
    steer_health = _steer_collapse(initial_y_groups)
    aggregate = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "checkpoint_type": args_cli.checkpoint_type,
        "output_dir": str(output_dir),
        "eval_config": {
            "episodes": int(args_cli.episodes),
            "steps": int(args_cli.steps),
            "record_every": int(args_cli.record_every),
            "fps": int(args_cli.fps),
            "video_recording": not bool(args_cli.no_video),
            "seed": int(args_cli.seed),
            "num_envs": int(args_cli.num_envs),
            "teacher_reference_reset_enable": bool(getattr(env_cfg, "teacher_reference_reset_enable", False)),
            "teacher_reference_reset_mix_start": float(getattr(env_cfg, "teacher_reference_reset_mix_start", 0.0)),
            "teacher_reference_reset_mix_end": float(getattr(env_cfg, "teacher_reference_reset_mix_end", 0.0)),
            "visual_curriculum_reset_mode": str(getattr(env_cfg, "visual_curriculum_reset_mode", "none")),
            "third_person_mode": str(args_cli.third_person_mode),
            "third_person_camera_eye": [float(v) for v in args_cli.camera_eye],
            "third_person_camera_lookat": [float(v) for v in args_cli.camera_lookat],
            "oblique_camera_eye": [float(v) for v in args_cli.oblique_camera_eye],
            "oblique_camera_lookat": [float(v) for v in args_cli.oblique_camera_lookat],
            "topdown_camera_eye": [float(v) for v in args_cli.topdown_camera_eye],
            "topdown_camera_lookat": [float(v) for v in args_cli.topdown_camera_lookat],
            "dual_camera": {
                "camera_version": str(getattr(env_cfg, "camera_version", "unspecified")),
                "config_hash_sha1": _camera_signature(env_cfg)["config_hash_sha1"],
                "hfov_deg": float(getattr(env_cfg, "dual_camera_hfov_deg", 0.0)),
                "far_clip_m": float(getattr(env_cfg, "dual_camera_far_clip_m", 0.0)),
                "left_pos_local": [float(v) for v in getattr(env_cfg, "dual_camera_left_pos_local", ())],
                "right_pos_local": [float(v) for v in getattr(env_cfg, "dual_camera_right_pos_local", ())],
                "left_rpy_local_deg": [float(v) for v in getattr(env_cfg, "dual_camera_left_rpy_local_deg", ())],
                "right_rpy_local_deg": [float(v) for v in getattr(env_cfg, "dual_camera_right_rpy_local_deg", ())],
            },
            "camera_signature": _camera_signature(env_cfg),
            "visual_envelope_termination_enable": bool(
                getattr(env_cfg, "visual_envelope_termination_enable", False)
            ),
            "fixed_stage1_init": None
            if args_cli.fixed_stage1_init is None
            else [float(v) for v in args_cli.fixed_stage1_init],
            "stage1_init_x_range_m": [
                float(getattr(env_cfg, "stage1_init_x_min_m", 0.0)),
                float(getattr(env_cfg, "stage1_init_x_max_m", 0.0)),
            ],
            "stage1_init_y_range_m": [
                float(getattr(env_cfg, "stage1_init_y_min_m", 0.0)),
                float(getattr(env_cfg, "stage1_init_y_max_m", 0.0)),
            ],
            "stage1_init_yaw_range_deg": [
                float(getattr(env_cfg, "stage1_init_yaw_deg_min", 0.0)),
                float(getattr(env_cfg, "stage1_init_yaw_deg_max", 0.0)),
            ],
            "visual_clean_max_pallet_disp_xy_m": float(args_cli.visual_clean_max_pallet_disp_xy_m),
            "hard_lateral_abs_init_y_m": float(args_cli.hard_lateral_abs_init_y_m),
        },
        "episodes": all_summaries,
        "aggregate": {
            "num_episodes": len(all_summaries),
            "success_rate": float(np.mean([bool(ep.get("ever_success", False)) for ep in all_summaries]))
            if all_summaries
            else 0.0,
            "insert_rate": float(np.mean([bool(ep.get("ever_inserted", False)) for ep in all_summaries]))
            if all_summaries
            else 0.0,
            "dirty_insert_rate": float(np.mean([bool(ep.get("ever_dirty_insert", False)) for ep in all_summaries]))
            if all_summaries
            else 0.0,
            "visual_clean_success_rate": float(
                np.mean([bool(ep.get("visual_clean_success", False)) for ep in all_summaries])
            )
            if all_summaries
            else 0.0,
            "hard_lateral_high_disp_rate": float(
                np.mean([bool(ep.get("hard_lateral_high_disp", False)) for ep in all_summaries])
            )
            if all_summaries
            else 0.0,
            "mean_max_pallet_disp_xy_m": float(
                np.mean([float(ep.get("max_pallet_disp_xy_m", 0.0)) for ep in all_summaries])
            )
            if all_summaries
            else 0.0,
            "mean_max_insert_depth_m": float(
                np.mean([float(ep.get("max_insert_depth_m", 0.0)) for ep in all_summaries])
            )
            if all_summaries
            else 0.0,
            "timeout_rate": float(np.mean([ep.get("done_reason") == "max_steps" for ep in all_summaries]))
            if all_summaries
            else 0.0,
        },
        "actions": action_stats,
        "initial_y_groups": initial_y_groups,
        "steer_health": steer_health,
    }
    aggregate["acceptance"] = _acceptance(aggregate)
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"[summary] wrote {output_dir / 'summary.json'}", flush=True)

    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
