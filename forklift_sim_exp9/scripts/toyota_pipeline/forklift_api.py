"""Non-web control API for IsaacLab forklift environments.

This wraps a single IsaacLab env with imperative methods used by teleop,
scripted validation, and data collection.  It intentionally does not start a
web server.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass
class ForkliftApiState:
    x: float
    y: float
    z: float
    yaw_deg: float
    lift_height_m: float
    vx_mps: float
    vy_mps: float
    yaw_rate_radps: float
    lift_joint_m: float
    pallet_disp_xy_m: float
    insert_depth_m: float
    hold_counter: float
    push_free: bool
    done_reason: str


class ForkliftIsaacApi:
    def __init__(self, env) -> None:
        self.env = env
        self.raw_env = env.unwrapped
        self.obs = None

    def reset(self):
        self.obs, _ = self.env.reset()
        return self.get_state()

    def stop(self):
        return self.set_command(0.0, 0.0, 0.0)

    def set_command(self, drive: float, steer: float, lift: float = 0.0):
        action = torch.tensor(
            [[float(drive), float(steer), float(lift)] for _ in range(int(self.raw_env.num_envs))],
            dtype=torch.float32,
            device=self.raw_env.device,
        ).clamp(-1.0, 1.0)
        self.obs, _, terminated, truncated, info = self.env.step(action)
        return self.get_state(), terminated, truncated, info

    def get_applied_action(self, env_id: int = 0) -> tuple[float, float, float]:
        action = self.raw_env.actions[int(env_id)].detach().cpu().tolist()
        return tuple(float(action[i]) for i in range(min(3, len(action))))

    def get_cameras(self) -> dict[str, torch.Tensor]:
        obs = self.raw_env._get_observations()
        cameras = {}
        if "image_left" in obs:
            cameras["left"] = obs["image_left"].detach().cpu()
            cameras["right"] = obs["image_right"].detach().cpu()
        elif "image" in obs:
            cameras["single"] = obs["image"].detach().cpu()
        return cameras

    def get_state(self) -> dict:
        raw = self.raw_env
        root_pos = raw.robot.data.root_pos_w[0]
        try:
            from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import _quat_to_yaw

            yaw = raw.robot.data.root_quat_w[0:1]
            yaw_deg = float((_quat_to_yaw(yaw)[0] * 180.0 / torch.pi).item())
        except Exception:
            yaw_deg = 0.0
        tip = raw._compute_fork_tip()[0]
        pallet_pos = raw.pallet.data.root_pos_w[0]
        if hasattr(raw, "_pallet_disp_xy"):
            pallet_disp = raw._pallet_disp_xy()[0]
        else:
            pallet_init = torch.tensor(raw.cfg.pallet_cfg.init_state.pos[:2], device=raw.device)
            pallet_init_world = raw.scene.env_origins[0, :2] + pallet_init
            pallet_disp = torch.norm(pallet_pos[:2] - pallet_init_world)
        insert_depth = getattr(raw, "_last_insert_depth", torch.zeros(raw.num_envs, device=raw.device))[0]
        root_lin_vel = raw.robot.data.root_lin_vel_w[0]
        root_ang_vel = raw.robot.data.root_ang_vel_w[0]
        done_reason = "running"
        if bool(getattr(raw, "_success_termination", torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool))[0].item()):
            done_reason = "success"
        elif bool(getattr(raw, "_preinsert_push_termination", torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool))[0].item()):
            done_reason = "preinsert_push"
        elif bool(getattr(raw, "_dirty_push_termination", torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool))[0].item()):
            done_reason = "dirty_push"
        push_free = bool((pallet_disp < float(raw.cfg.push_free_disp_thresh_m)).item())
        state = ForkliftApiState(
            x=float(root_pos[0].item()),
            y=float(root_pos[1].item()),
            z=float(root_pos[2].item()),
            yaw_deg=yaw_deg,
            lift_height_m=float((tip[2] - raw._fork_tip_z0[0]).item()),
            vx_mps=float(root_lin_vel[0].item()),
            vy_mps=float(root_lin_vel[1].item()),
            yaw_rate_radps=float(root_ang_vel[2].item()),
            lift_joint_m=float(raw._joint_pos[0, raw._lift_id].item()),
            pallet_disp_xy_m=float(pallet_disp.item()),
            insert_depth_m=float(insert_depth.item()),
            hold_counter=float(raw._hold_counter[0].item()),
            push_free=push_free,
            done_reason=done_reason,
        )
        return asdict(state)
