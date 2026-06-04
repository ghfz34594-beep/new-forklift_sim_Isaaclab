"""Collect Toyota approach demonstrations through the non-web API.

This script is intended for interactive IsaacSim runs.  It records left/right
camera frames, keyboard/API commands, state metrics and done reasons into a
directory that can be consumed by ``train_approach_bc.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect Toyota dual-camera approach demonstrations")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--drive", type=float, default=0.45)
parser.add_argument("--steer", type=float, default=0.75)
parser.add_argument("--lift", type=float, default=0.8)
parser.add_argument("--image_every", type=int, default=1)
parser.add_argument("--flush_every", type=int, default=10, help="Write metadata.csv/summary.json every N steps.")
parser.add_argument(
    "--stop_file_name",
    type=str,
    default="STOP",
    help="Create this file inside output_dir from another terminal to stop and save cleanly.",
)
parser.add_argument(
    "--enable_action_guard",
    action="store_true",
    help="Keep the PushSafe near-field forward-action guard during teleop. Default is off for manual collection.",
)
parser.add_argument("--headless_scripted_zero", action="store_true", help="Headless smoke mode: record zero commands.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

if not args_cli.headless_scripted_zero:
    import carb
    import omni.appwindow
    import omni.kit.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401

from forklift_api import ForkliftIsaacApi
from rollout_recorder import ToyotaRolloutRecorder


class KeyboardState:
    def __init__(self) -> None:
        self.pressed: set[int] = set()
        self.reset_requested = False
        self.quit_requested = False

    def on_keyboard_event(self, event) -> bool:
        key = int(event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed.add(key)
            if key == int(carb.input.KeyboardInput.R):
                self.reset_requested = True
            quit_names = ("ESCAPE", "X", "F10", "END")
            quit_keys = [int(getattr(carb.input.KeyboardInput, name)) for name in quit_names if hasattr(carb.input.KeyboardInput, name)]
            if key in quit_keys:
                self.quit_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(key)
        return True


def _keyboard_command(keyboard: KeyboardState) -> tuple[float, float, float]:
    drive = 0.0
    steer = 0.0
    lift = 0.0
    keys = keyboard.pressed
    if int(carb.input.KeyboardInput.W) in keys:
        drive += args_cli.drive
    if int(carb.input.KeyboardInput.S) in keys:
        drive -= args_cli.drive
    if int(carb.input.KeyboardInput.A) in keys:
        steer += args_cli.steer
    if int(carb.input.KeyboardInput.D) in keys:
        steer -= args_cli.steer
    if int(carb.input.KeyboardInput.Q) in keys:
        lift += args_cli.lift
    if int(carb.input.KeyboardInput.E) in keys:
        lift -= args_cli.lift
    if int(carb.input.KeyboardInput.SPACE) in keys:
        drive = steer = lift = 0.0
    return drive, steer, lift


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    stop_file = output_dir / str(args_cli.stop_file_name)
    if stop_file.exists():
        stop_file.unlink()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_camera = True
    env_cfg.use_dual_cameras = True
    env_cfg.stage_1_mode = False
    env_cfg.action_space = 3
    if hasattr(env_cfg, "preinsert_action_guard_enable"):
        env_cfg.preinsert_action_guard_enable = bool(args_cli.enable_action_guard)
    if hasattr(env_cfg, "toyota_action_noise_std"):
        env_cfg.toyota_action_noise_std = 0.0
    if hasattr(env_cfg, "toyota_velocity_obs_noise_std"):
        env_cfg.toyota_velocity_obs_noise_std = 0.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    api = ForkliftIsaacApi(env)
    api.reset()
    recorder = ToyotaRolloutRecorder(
        args_cli.output_dir,
        save_images=True,
        image_every=args_cli.image_every,
        flush_every=args_cli.flush_every,
        metadata={"task": args_cli.task},
    )

    keyboard = None
    input_iface = None
    keyboard_sub = None
    if not args_cli.headless_scripted_zero:
        keyboard = KeyboardState()
        app_window = omni.appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard_sub = input_iface.subscribe_to_keyboard_events(app_window.get_keyboard(), keyboard.on_keyboard_event)
        print(
            "[collect_teleop] W/S drive, A/D steer, Q/E lift, Space stop, R reset, "
            "Esc/X/F10/End quit. Or create output_dir/STOP from another terminal.",
            flush=True,
        )

    prev_action = (0.0, 0.0, 0.0)
    episode_id = 0
    try:
        for step in range(int(args_cli.max_steps)):
            if args_cli.headless_scripted_zero:
                command = (0.0, 0.0, 0.0)
            else:
                assert keyboard is not None
                if keyboard.quit_requested:
                    break
                if stop_file.exists():
                    print(f"[collect_teleop] stop file detected: {stop_file}", flush=True)
                    break
                if keyboard.reset_requested:
                    api.reset()
                    keyboard.reset_requested = False
                    prev_action = (0.0, 0.0, 0.0)
                    episode_id += 1
                command = _keyboard_command(keyboard)

            state, terminated, truncated, _ = api.set_command(*command)
            done = bool((terminated | truncated).any().item())
            cameras = api.get_cameras()
            effective = api.get_applied_action()
            recorder.record_step(
                step=step,
                episode_id=episode_id,
                cameras=cameras,
                state=state,
                command=command,
                effective_action=effective,
                prev_action=prev_action,
                done=done,
                done_reason=state.get("done_reason", "running"),
            )
            prev_action = effective
            if done:
                api.reset()
                prev_action = (0.0, 0.0, 0.0)
                episode_id += 1
            if not args_cli.headless_scripted_zero:
                omni.kit.app.get_app().update()
    finally:
        if input_iface is not None and keyboard_sub is not None:
            input_iface.unsubscribe_to_keyboard_events(keyboard_sub)
        recorder.close({"task": args_cli.task})
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
