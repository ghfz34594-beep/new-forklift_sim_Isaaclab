"""Keyboard teleoperation for the Toyota dual-camera forklift task.

Run through IsaacLab, for example:

    /data/jianshi/projects/forklift_sim_exp9/scripts/toyota_pipeline/run_isaaclab_env.sh \
      -p /data/jianshi/projects/forklift_sim_exp9/scripts/toyota_pipeline/teleop_dual_camera.py \
      --task Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0 \
      --num_envs 1 --final_rect_init --overview_camera_20260604

Keys:
    W/S drive forward/backward
    A/D steer left/right
    Q/E lift up/down when --enable_lift is set
    Space stop
    R reset
    Esc/X quit
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Toyota dual-camera forklift keyboard teleop")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--drive", type=float, default=0.45)
parser.add_argument("--steer", type=float, default=0.75)
parser.add_argument("--lift", type=float, default=0.8)
parser.add_argument(
    "--enable_lift",
    action="store_true",
    help="Use full 3D action mode so Q/E can move lift. This uses the non-Stage-1 reset path, not the approach training range.",
)
parser.add_argument(
    "--enable_action_guard",
    action="store_true",
    help="Keep the PushSafe near-field action guard. Default is off for human learnability checks.",
)
parser.add_argument(
    "--final_rect_init",
    action="store_true",
    help="Force the final direct-visual rectangular init distribution: x[-4,-3], y[-0.6,0.6], yaw +/-14.3239 deg.",
)
parser.add_argument("--fixed_init_x", type=float, default=None, help="Fix reset x in env-local meters.")
parser.add_argument("--fixed_init_y", type=float, default=None, help="Fix reset y in env-local meters.")
parser.add_argument("--fixed_init_yaw_deg", type=float, default=None, help="Fix reset yaw in degrees.")
parser.add_argument("--env_spacing", type=float, default=None)
parser.add_argument("--camera_far", type=float, default=None)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=None)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=None)
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=None)
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=None)
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=None)
parser.add_argument("--vision_room", action="store_true", default=None)
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room")
parser.add_argument(
    "--overview_camera_20260604",
    action="store_true",
    help="Apply the camera/view settings used by outputs/topdown_dual_camera_fork_visible_20260604/overview.png.",
)
parser.add_argument("--highlight_pallet", action="store_true", help="Temporarily color the pallet green for inspection.")
parser.add_argument(
    "--isaac_dual_viewports",
    action="store_true",
    default=False,
    help="Open Isaac viewport windows for env0 CameraLeft and CameraRight. These are USD viewports, not the training tensor.",
)
parser.add_argument(
    "--no_isaac_dual_viewports",
    action="store_false",
    dest="isaac_dual_viewports",
    help="Do not create/switch Isaac camera viewports.",
)
parser.add_argument(
    "--isaac_sensor_window",
    action="store_true",
    default=True,
    help="Show the actual dual-camera model input inside an Isaac UI window.",
)
parser.add_argument(
    "--no_isaac_sensor_window",
    action="store_false",
    dest="isaac_sensor_window",
    help="Disable the Isaac UI model-input camera window.",
)
parser.add_argument("--viewport_width", type=int, default=640)
parser.add_argument("--viewport_height", type=int, default=480)
parser.add_argument(
    "--show_camera_window",
    action="store_true",
    default=False,
    help="Also show live left/right sensor images in an OpenCV window. This is the same model-input view.",
)
parser.add_argument(
    "--no_camera_window",
    action="store_false",
    dest="show_camera_window",
    help="Disable the OpenCV model-input camera window.",
)
parser.add_argument(
    "--opencv_keyboard",
    action="store_true",
    help="Read sticky WASD controls from the OpenCV camera window instead of the Isaac window.",
)
parser.add_argument("--opencv_hold_steps", type=int, default=8, help="How long an OpenCV key press persists in env steps.")
parser.add_argument("--display_scale", type=float, default=1.4, help="Scale factor for the OpenCV camera window.")
parser.add_argument("--status_every", type=int, default=60, help="Print state every N env steps; 0 disables periodic status.")
parser.add_argument("--debug_keys", action="store_true", help="Print Isaac keyboard press/release events.")
parser.add_argument("--no_reset_on_done", action="store_true", help="Do not auto-reset on task termination/time-out.")
parser.add_argument("--scripted_smoke_steps", type=int, default=0, help="Run N zero-action steps without keyboard/UI, then exit.")
parser.add_argument("--topdown_view", action="store_true", help="Set the Isaac viewport to the same overhead review angle.")
parser.add_argument("--topdown_camera_eye", type=float, nargs=3, default=(-2.7, 0.0, 7.0))
parser.add_argument("--topdown_camera_lookat", type=float, nargs=3, default=(-2.7, 0.0, 0.0))
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if bool(args_cli.opencv_keyboard):
    args_cli.show_camera_window = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import carb
import omni.appwindow
import omni.kit.app
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.forklift_pallet_insert_lift  # noqa: F401


FINAL_RECT_INIT = {
    "x_min": -4.0,
    "x_max": -3.0,
    "y_min": -0.6,
    "y_max": 0.6,
    "yaw_min": -14.32394487827058,
    "yaw_max": 14.32394487827058,
}

OVERVIEW_CAMERA_20260604 = {
    "env_spacing": 20.0,
    "camera_far": 8.0,
    "hfov_deg": 100.0,
    "left_pos": (150.0, 75.0, 140.0),
    "right_pos": (150.0, -75.0, 140.0),
    "left_rpy": (0.0, 40.0, -20.0),
    "right_rpy": (0.0, 40.0, 20.0),
    "vision_room": False,
}


def _to_uint8_hwc(image: Any):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        arr = image.numpy()
    else:
        import numpy as np

        arr = np.asarray(image)
    if arr.ndim == 2:
        import numpy as np

        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr.astype("float32")
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    import numpy as np

    return np.clip(arr, 0, 255).astype("uint8")


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


def _apply_camera_overrides(env_cfg: Any) -> None:
    if args_cli.overview_camera_20260604:
        if args_cli.env_spacing is None:
            args_cli.env_spacing = OVERVIEW_CAMERA_20260604["env_spacing"]
        if args_cli.camera_far is None:
            args_cli.camera_far = OVERVIEW_CAMERA_20260604["camera_far"]
        if args_cli.dual_camera_hfov_deg is None:
            args_cli.dual_camera_hfov_deg = OVERVIEW_CAMERA_20260604["hfov_deg"]
        if args_cli.dual_camera_left_pos is None:
            args_cli.dual_camera_left_pos = list(OVERVIEW_CAMERA_20260604["left_pos"])
        if args_cli.dual_camera_right_pos is None:
            args_cli.dual_camera_right_pos = list(OVERVIEW_CAMERA_20260604["right_pos"])
        if args_cli.dual_camera_left_rpy_deg is None:
            args_cli.dual_camera_left_rpy_deg = list(OVERVIEW_CAMERA_20260604["left_rpy"])
        if args_cli.dual_camera_right_rpy_deg is None:
            args_cli.dual_camera_right_rpy_deg = list(OVERVIEW_CAMERA_20260604["right_rpy"])
        if args_cli.vision_room is None:
            args_cli.vision_room = bool(OVERVIEW_CAMERA_20260604["vision_room"])

    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    if args_cli.camera_far is not None:
        _set_camera_far(env_cfg, float(args_cli.camera_far))
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
    if args_cli.vision_room is not None:
        env_cfg.vision_room_enable = bool(args_cli.vision_room)


def _apply_init_overrides(env_cfg: Any) -> None:
    if args_cli.final_rect_init:
        env_cfg.stage1_use_triangular_visible_init = False
        env_cfg.stage1_init_x_min_m = FINAL_RECT_INIT["x_min"]
        env_cfg.stage1_init_x_max_m = FINAL_RECT_INIT["x_max"]
        env_cfg.stage1_init_y_min_m = FINAL_RECT_INIT["y_min"]
        env_cfg.stage1_init_y_max_m = FINAL_RECT_INIT["y_max"]
        env_cfg.stage1_init_yaw_deg_min = FINAL_RECT_INIT["yaw_min"]
        env_cfg.stage1_init_yaw_deg_max = FINAL_RECT_INIT["yaw_max"]
        if hasattr(env_cfg, "teacher_reference_reset_enable"):
            env_cfg.teacher_reference_reset_enable = False
        if hasattr(env_cfg, "stage1_near_hard_curriculum_enable"):
            env_cfg.stage1_near_hard_curriculum_enable = False

    if args_cli.fixed_init_x is not None:
        env_cfg.stage1_use_triangular_visible_init = False
        env_cfg.stage1_init_x_min_m = float(args_cli.fixed_init_x)
        env_cfg.stage1_init_x_max_m = float(args_cli.fixed_init_x)
    if args_cli.fixed_init_y is not None:
        env_cfg.stage1_use_triangular_visible_init = False
        env_cfg.stage1_init_y_min_m = float(args_cli.fixed_init_y)
        env_cfg.stage1_init_y_max_m = float(args_cli.fixed_init_y)
    if args_cli.fixed_init_yaw_deg is not None:
        env_cfg.stage1_use_triangular_visible_init = False
        env_cfg.stage1_init_yaw_deg_min = float(args_cli.fixed_init_yaw_deg)
        env_cfg.stage1_init_yaw_deg_max = float(args_cli.fixed_init_yaw_deg)


def _format_init_range(env_cfg: Any) -> str:
    if bool(getattr(env_cfg, "stage1_use_triangular_visible_init", False)):
        return (
            "triangular "
            f"x=[{env_cfg.stage1_tri_x_far_m:.3f},{env_cfg.stage1_tri_x_near_m:.3f}] m, "
            f"|y|={env_cfg.stage1_tri_y_half_width_far_m:.3f}->{env_cfg.stage1_tri_y_half_width_near_m:.3f} m, "
            f"|yaw|={env_cfg.stage1_tri_yaw_deg_far:.3f}->{env_cfg.stage1_tri_yaw_deg_near:.3f} deg"
        )
    return (
        f"rect x=[{env_cfg.stage1_init_x_min_m:.3f},{env_cfg.stage1_init_x_max_m:.3f}] m, "
        f"y=[{env_cfg.stage1_init_y_min_m:.3f},{env_cfg.stage1_init_y_max_m:.3f}] m, "
        f"yaw=[{env_cfg.stage1_init_yaw_deg_min:.6f},{env_cfg.stage1_init_yaw_deg_max:.6f}] deg"
    )


def _make_preview_material(stage, path: str, diffuse_color: tuple[float, float, float]):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material_recursive(root_prim, material) -> int:
    from pxr import Usd, UsdGeom, UsdShade

    count = 0
    if root_prim and root_prim.IsValid():
        UsdShade.MaterialBindingAPI(root_prim).Bind(material)
        count += 1
        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim:
                continue
            if prim.IsA(UsdGeom.Gprim) or prim.GetTypeName() in ("Mesh", "Cube"):
                UsdShade.MaterialBindingAPI(prim).Bind(material)
                count += 1
    return count


def _highlight_pallets(raw_env: Any) -> None:
    stage = raw_env.sim.stage
    material = _make_preview_material(stage, "/World/TeleopPalletVisibilityMaterial", (0.0, 1.0, 0.0))
    bound = 0
    for env_id in range(int(raw_env.num_envs)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Pallet")
        bound += _bind_material_recursive(prim, material)
    print(f"[teleop] highlighted pallet prims={bound}", flush=True)


def _set_topdown_view(raw_env: Any) -> None:
    if not (bool(args_cli.topdown_view) or bool(args_cli.overview_camera_20260604)):
        return
    if bool(getattr(args_cli, "headless", False)):
        return
    origin = raw_env.scene.env_origins[0].detach().cpu()
    eye = origin + torch.tensor(args_cli.topdown_camera_eye)
    lookat = origin + torch.tensor(args_cli.topdown_camera_lookat)
    raw_env.sim.set_camera_view(
        eye=tuple(float(v) for v in eye.tolist()),
        target=tuple(float(v) for v in lookat.tolist()),
    )
    print(
        "[teleop] active Isaac viewport set to overview topdown "
        f"eye={tuple(float(v) for v in eye.tolist())} target={tuple(float(v) for v in lookat.tolist())}",
        flush=True,
    )


def _env0_camera_path(side: str) -> str:
    if side == "left":
        return "/World/envs/env_0/CameraLeft"
    if side == "right":
        return "/World/envs/env_0/CameraRight"
    raise ValueError(f"unknown camera side: {side}")


def _set_viewport_camera(viewport_api: Any, camera_path: str, label: str) -> bool:
    import omni.kit.commands
    from pxr import Sdf, UsdGeom

    sdf_path = Sdf.Path(camera_path)
    stage = getattr(viewport_api, "stage", None)
    if stage is not None:
        prim = stage.GetPrimAtPath(sdf_path)
        if not prim or not prim.IsValid():
            print(f"[teleop] warning: {label} camera prim does not exist: {camera_path}", flush=True)
            return False
        if not prim.IsA(UsdGeom.Camera):
            print(f"[teleop] warning: {label} prim is not UsdGeom.Camera: {camera_path}", flush=True)
            return False

    ok = False
    for setter in (
        lambda: omni.kit.commands.execute("SetViewportCameraCommand", camera_path=sdf_path, viewport_api=viewport_api),
        lambda: setattr(viewport_api, "camera_path", sdf_path),
        lambda: viewport_api.set_active_camera(sdf_path),
        lambda: viewport_api.set_active_camera(camera_path),
    ):
        try:
            setter()
            ok = True
        except Exception:
            pass

    try:
        viewport_api.viewport_changed(viewport_api.camera_path, viewport_api.stage)
    except Exception:
        pass
    actual = getattr(viewport_api, "camera_path", None)
    print(f"[teleop] {label} viewport camera requested={camera_path} actual={actual}", flush=True)
    return ok and str(actual) == camera_path


def _set_viewport_resolution(viewport_api: Any) -> None:
    try:
        viewport_api.resolution = (int(args_cli.viewport_width), int(args_cli.viewport_height))
    except Exception:
        pass


def _sync_dual_camera_prims(raw_env: Any) -> None:
    sync_fn = getattr(raw_env, "_sync_dual_camera_poses", None)
    if callable(sync_fn):
        try:
            sync_fn()
        except Exception as exc:
            print(f"[teleop] warning: dual camera pose sync failed: {exc}", flush=True)


def _setup_isaac_camera_viewports(raw_env: Any) -> list[Any]:
    if not bool(args_cli.isaac_dual_viewports):
        return []
    if bool(getattr(args_cli, "headless", False)):
        print("[teleop] headless mode: Isaac camera viewports skipped; run without --headless for manual WASD.", flush=True)
        return []

    left_path = _env0_camera_path("left")
    right_path = _env0_camera_path("right")
    stage = raw_env.sim.stage
    missing = [path for path in (left_path, right_path) if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        print(f"[teleop] warning: camera prim(s) not found yet: {missing}", flush=True)

    _sync_dual_camera_prims(raw_env)

    from omni.kit.viewport.utility import create_viewport_window
    from pxr import Sdf

    windows: list[Any] = []
    left_window = create_viewport_window(
        name="Forklift Left Camera",
        width=int(args_cli.viewport_width),
        height=int(args_cli.viewport_height),
        position_x=40,
        position_y=80,
        camera_path=Sdf.Path(left_path),
    )
    if left_window is not None:
        _set_viewport_resolution(left_window.viewport_api)
        windows.append(left_window)

    right_window = create_viewport_window(
        name="Forklift Right Camera",
        width=int(args_cli.viewport_width),
        height=int(args_cli.viewport_height),
        position_x=80 + int(args_cli.viewport_width),
        position_y=80,
        camera_path=Sdf.Path(right_path),
    )
    if right_window is not None:
        _set_viewport_resolution(right_window.viewport_api)
        windows.append(right_window)

    for _ in range(8):
        omni.kit.app.get_app().update()
    if left_window is not None:
        _set_viewport_camera(left_window.viewport_api, left_path, "left")
    if right_window is not None:
        _set_viewport_camera(right_window.viewport_api, right_path, "right")
    for _ in range(3):
        omni.kit.app.get_app().update()
    if left_window is not None:
        _set_viewport_camera(left_window.viewport_api, left_path, "left")
    if right_window is not None:
        _set_viewport_camera(right_window.viewport_api, right_path, "right")

    print(f"[teleop] Isaac viewport cameras: left={left_path}, right={right_path}", flush=True)
    return windows


class OpenCvStickyKeyboard:
    def __init__(self, cv2_module) -> None:
        self.cv2 = cv2_module
        self.drive_cmd = 0.0
        self.steer_cmd = 0.0
        self.lift_cmd = 0.0
        self.drive_hold = 0
        self.steer_hold = 0
        self.lift_hold = 0
        self.reset_requested = False
        self.quit_requested = False

    def poll(self) -> tuple[float, float, float]:
        key = self.cv2.waitKey(1) & 0xFF
        hold_steps = max(1, int(args_cli.opencv_hold_steps))
        if key in (27, ord("x"), ord("X")):
            self.quit_requested = True
        elif key in (ord("r"), ord("R")):
            self.reset_requested = True
        elif key in (ord(" "),):
            self.drive_cmd = self.steer_cmd = self.lift_cmd = 0.0
            self.drive_hold = self.steer_hold = self.lift_hold = 0
        elif key in (ord("w"), ord("W")):
            self.drive_cmd = float(args_cli.drive)
            self.drive_hold = hold_steps
        elif key in (ord("s"), ord("S")):
            self.drive_cmd = -float(args_cli.drive)
            self.drive_hold = hold_steps
        elif key in (ord("a"), ord("A")):
            self.steer_cmd = float(args_cli.steer)
            self.steer_hold = hold_steps
        elif key in (ord("d"), ord("D")):
            self.steer_cmd = -float(args_cli.steer)
            self.steer_hold = hold_steps
        elif key in (ord("q"), ord("Q")) and bool(args_cli.enable_lift):
            self.lift_cmd = float(args_cli.lift)
            self.lift_hold = hold_steps
        elif key in (ord("e"), ord("E")) and bool(args_cli.enable_lift):
            self.lift_cmd = -float(args_cli.lift)
            self.lift_hold = hold_steps

        self.drive_hold = max(0, self.drive_hold - 1)
        self.steer_hold = max(0, self.steer_hold - 1)
        self.lift_hold = max(0, self.lift_hold - 1)
        if self.drive_hold == 0:
            self.drive_cmd = 0.0
        if self.steer_hold == 0:
            self.steer_cmd = 0.0
        if self.lift_hold == 0:
            self.lift_cmd = 0.0
        return self.drive_cmd, self.steer_cmd, self.lift_cmd


def _keyboard_command(keyboard: KeyboardState) -> tuple[float, float, float]:
    drive = 0.0
    steer = 0.0
    lift = 0.0
    keys = keyboard.pressed
    if "W" in keys:
        drive += args_cli.drive
    if "S" in keys:
        drive -= args_cli.drive
    if "A" in keys:
        steer += args_cli.steer
    if "D" in keys:
        steer -= args_cli.steer
    if "Q" in keys and bool(args_cli.enable_lift):
        lift += args_cli.lift
    if "E" in keys and bool(args_cli.enable_lift):
        lift -= args_cli.lift
    if "SPACE" in keys:
        drive = steer = lift = 0.0
    return drive, steer, lift


def _reset_and_report(env: Any, raw_env: Any) -> None:
    env.reset()
    x = float(raw_env._debug_reset_x[0].detach().cpu().item())
    y = float(raw_env._debug_reset_y[0].detach().cpu().item())
    yaw = float(raw_env._debug_reset_yaw_deg[0].detach().cpu().item())
    print(f"[teleop] reset init env0: x={x:+.3f} m y={y:+.3f} m yaw={yaw:+.3f} deg", flush=True)


def _dual_camera_rgb(raw_env: Any):
    left_batch, right_batch = raw_env._get_dual_camera_images()
    left = _to_uint8_hwc(left_batch[0])
    right = _to_uint8_hwc(right_batch[0])
    import numpy as np

    divider = np.full((left.shape[0], 4, 3), 245, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


class IsaacDualCameraSensorWindow:
    def __init__(self, raw_env: Any) -> None:
        import omni.ui as ui
        import numpy as np

        self.ui = ui
        self.window = ui.Window(
            "Forklift Dual Camera Model Input",
            width=max(720, int(args_cli.viewport_width) * 2),
            height=max(360, int(args_cli.viewport_height)),
        )
        self.provider = ui.ByteImageProvider()
        self.status_label = None
        initial = _dual_camera_rgb(raw_env)
        height, width = initial.shape[:2]
        rgba = np.dstack((initial, np.full((height, width, 1), 255, dtype=np.uint8)))
        self.provider.set_bytes_data(rgba.flatten().data, [width, height])
        with self.window.frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=18):
                    ui.Label("Left", alignment=ui.Alignment.CENTER, style={"font_size": 14})
                    ui.Label("Right", alignment=ui.Alignment.CENTER, style={"font_size": 14})
                ui.ImageWithProvider(self.provider)
                self.status_label = ui.Label("", height=20, style={"font_size": 13})

    def update(self, raw_env: Any, step: int, command: tuple[float, float, float]) -> None:
        import numpy as np

        image = _dual_camera_rgb(raw_env)
        height, width = image.shape[:2]
        rgba = np.dstack((image, np.full((height, width, 1), 255, dtype=np.uint8)))
        self.provider.set_bytes_data(rgba.flatten().data, [width, height])
        if self.status_label is not None:
            x = float(raw_env._debug_reset_x[0].detach().cpu().item())
            y = float(raw_env._debug_reset_y[0].detach().cpu().item())
            yaw = float(raw_env._debug_reset_yaw_deg[0].detach().cpu().item())
            self.status_label.text = (
                f"step={step} init x={x:+.2f} y={y:+.2f} yaw={yaw:+.1f} "
                f"cmd drive={command[0]:+.2f} steer={command[1]:+.2f}"
            )


def _setup_isaac_sensor_window(raw_env: Any):
    if not bool(args_cli.isaac_sensor_window):
        return None
    if bool(getattr(args_cli, "headless", False)):
        return None
    try:
        window = IsaacDualCameraSensorWindow(raw_env)
    except Exception as exc:
        print(f"[teleop] warning: Isaac sensor window failed: {exc}", flush=True)
        return None
    print("[teleop] Isaac sensor window shows raw env0 left/right model input.", flush=True)
    return window


def _report_dual_camera_prims(raw_env: Any) -> None:
    try:
        left_path = raw_env._camera_left._view.prim_paths[0]
        right_path = raw_env._camera_right._view.prim_paths[0]
    except Exception as exc:
        print(f"[teleop] warning: failed to read dual camera prim paths: {exc}", flush=True)
        return
    print(f"[teleop] dual camera prims: left={left_path} right={right_path}", flush=True)


def _draw_camera_window(cv2_module, raw_env: Any, step: int, command: tuple[float, float, float]) -> None:
    image = _dual_camera_rgb(raw_env)
    scale = max(0.2, float(args_cli.display_scale))
    if abs(scale - 1.0) > 1e-3:
        image = cv2_module.resize(image, None, fx=scale, fy=scale, interpolation=cv2_module.INTER_LINEAR)
    bgr = cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR)
    x = float(raw_env._debug_reset_x[0].detach().cpu().item())
    y = float(raw_env._debug_reset_y[0].detach().cpu().item())
    yaw = float(raw_env._debug_reset_yaw_deg[0].detach().cpu().item())
    text = (
        f"step={step} init x={x:+.2f} y={y:+.2f} yaw={yaw:+.1f} "
        f"cmd d={command[0]:+.2f} s={command[1]:+.2f}"
    )
    cv2_module.putText(bgr, text, (8, 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3)
    cv2_module.putText(bgr, text, (8, 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)
    cv2_module.imshow("forklift left/right cameras", bgr)


class KeyboardState:
    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.reset_requested = False
        self.quit_requested = False

    @staticmethod
    def _key_name(event) -> str:
        try:
            return str(event.input.name).upper()
        except Exception:
            try:
                return str(carb.input.KeyboardInput(int(event.input)).name).upper()
            except Exception:
                return str(event.input).upper()

    def on_keyboard_event(self, event) -> bool:
        key = self._key_name(event)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed.add(key)
            if bool(args_cli.debug_keys):
                print(f"[teleop] key_down={key} pressed={sorted(self.pressed)}", flush=True)
            if key == "SPACE":
                self.pressed.clear()
            if key == "R":
                self.reset_requested = True
            if key in ("ESCAPE", "X"):
                self.quit_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(key)
            if bool(args_cli.debug_keys):
                print(f"[teleop] key_up={key} pressed={sorted(self.pressed)}", flush=True)
        return True


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = not bool(args_cli.enable_lift)
    env_cfg.action_space = 3 if bool(args_cli.enable_lift) else 2
    _apply_camera_overrides(env_cfg)
    _apply_init_overrides(env_cfg)
    if hasattr(env_cfg, "preinsert_action_guard_enable"):
        env_cfg.preinsert_action_guard_enable = bool(args_cli.enable_action_guard)
    if hasattr(env_cfg, "toyota_action_noise_std"):
        env_cfg.toyota_action_noise_std = 0.0
    if hasattr(env_cfg, "toyota_velocity_obs_noise_std"):
        env_cfg.toyota_velocity_obs_noise_std = 0.0

    print(f"[teleop] task={args_cli.task}", flush=True)
    print(f"[teleop] init_distribution={_format_init_range(env_cfg)}", flush=True)
    if bool(args_cli.enable_lift) and (
        bool(args_cli.final_rect_init)
        or args_cli.fixed_init_x is not None
        or args_cli.fixed_init_y is not None
        or args_cli.fixed_init_yaw_deg is not None
    ):
        print(
            "[teleop] warning: --enable_lift switches the env to the non-Stage-1 reset path; "
            "do not use it for the training-range visual learnability check.",
            flush=True,
        )
    print(
        "[teleop] camera="
        f"hfov={float(getattr(env_cfg, 'dual_camera_hfov_deg', -1.0)):.1f} "
        f"far={float(getattr(env_cfg, 'dual_camera_far_clip_m', -1.0)):.1f} "
        f"left_pos={tuple(float(v) for v in env_cfg.dual_camera_left_pos_local)} "
        f"left_rpy={tuple(float(v) for v in env_cfg.dual_camera_left_rpy_local_deg)} "
        f"room={bool(getattr(env_cfg, 'vision_room_enable', False))}",
        flush=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    _reset_and_report(env, raw_env)
    _report_dual_camera_prims(raw_env)
    _set_topdown_view(raw_env)
    viewport_windows = _setup_isaac_camera_viewports(raw_env)
    sensor_window = _setup_isaac_sensor_window(raw_env)
    if bool(args_cli.highlight_pallet):
        _highlight_pallets(raw_env)

    if int(args_cli.scripted_smoke_steps) > 0:
        print(f"[teleop] scripted smoke: zero-action steps={int(args_cli.scripted_smoke_steps)}", flush=True)
        for step in range(int(args_cli.scripted_smoke_steps)):
            action = torch.zeros((int(raw_env.num_envs), 3), dtype=torch.float32, device=raw_env.device)
            _, _, terminated, truncated, _ = env.step(action)
            done = bool(torch.as_tensor(terminated | truncated).any().item())
            if int(args_cli.status_every) > 0:
                pos = raw_env.robot.data.root_pos_w[0].detach().cpu()
                print(
                    f"[teleop] smoke step={step} robot=({float(pos[0]):+.3f},{float(pos[1]):+.3f}) done={int(done)}",
                    flush=True,
                )
        env.close()
        simulation_app.close()
        return

    cv2 = None
    cv2_keyboard = None
    if bool(args_cli.show_camera_window):
        import cv2 as cv2_module

        cv2 = cv2_module
        cv2.namedWindow("forklift left/right cameras", cv2.WINDOW_NORMAL)
        if bool(args_cli.opencv_keyboard):
            cv2_keyboard = OpenCvStickyKeyboard(cv2)

    keyboard = None
    input_iface = None
    keyboard_sub = None
    if cv2_keyboard is None:
        keyboard = KeyboardState()
        app_window = omni.appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard_sub = input_iface.subscribe_to_keyboard_events(
            app_window.get_keyboard(), keyboard.on_keyboard_event
        )

    print("[teleop] W/S drive, A/D steer, Space stop, R reset, Esc/X quit", flush=True)
    print("[teleop] Focus the Isaac window/viewport before pressing WASD.", flush=True)
    if not bool(args_cli.enable_lift):
        print("[teleop] Stage-1 manual check: lift is disabled so reset stays in the training approach range.", flush=True)
    if cv2_keyboard is not None:
        print("[teleop] OpenCV keyboard is sticky: tap/hold WASD in the camera window; Space stops.", flush=True)

    step = 0
    try:
        while simulation_app.is_running():
            if cv2_keyboard is not None:
                drive, steer, lift = cv2_keyboard.poll()
                reset_requested = cv2_keyboard.reset_requested
                quit_requested = cv2_keyboard.quit_requested
            else:
                assert keyboard is not None
                drive, steer, lift = _keyboard_command(keyboard)
                reset_requested = keyboard.reset_requested
                quit_requested = keyboard.quit_requested
            if quit_requested:
                break

            action = torch.tensor([[drive, steer, lift]], dtype=torch.float32, device=raw_env.device).repeat(
                int(raw_env.num_envs), 1
            )
            _, _, terminated, truncated, _ = env.step(action)
            done = bool(torch.as_tensor(terminated | truncated).any().item())
            if reset_requested or (done and not bool(args_cli.no_reset_on_done)):
                _reset_and_report(env, raw_env)
                if cv2_keyboard is not None:
                    cv2_keyboard.reset_requested = False
                elif keyboard is not None:
                    keyboard.reset_requested = False
                    keyboard.pressed.clear()
            if cv2 is not None:
                _draw_camera_window(cv2, raw_env, step, (drive, steer, lift))
                if cv2_keyboard is None:
                    cv2.waitKey(1)
            if sensor_window is not None:
                sensor_window.update(raw_env, step, (drive, steer, lift))
            if int(args_cli.status_every) > 0 and step % int(args_cli.status_every) == 0:
                pos = raw_env.robot.data.root_pos_w[0].detach().cpu()
                print(
                    f"[teleop] step={step} robot=({float(pos[0]):+.3f},{float(pos[1]):+.3f}) "
                    f"cmd=({drive:+.2f},{steer:+.2f},{lift:+.2f}) done={int(done)}",
                    flush=True,
                )
            omni.kit.app.get_app().update()
            step += 1
    finally:
        if input_iface is not None and keyboard_sub is not None:
            input_iface.unsubscribe_to_keyboard_events(keyboard_sub)
        if cv2 is not None:
            cv2.destroyAllWindows()
        del sensor_window
        del viewport_windows
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
