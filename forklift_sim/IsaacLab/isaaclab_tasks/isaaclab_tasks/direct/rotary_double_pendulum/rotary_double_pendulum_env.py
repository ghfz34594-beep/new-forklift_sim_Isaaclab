"""
二阶倒立摆（Acrobot / Double Pendulum）环境

结构：
- 电机悬浮在空中（z=1.2m）
- 驱动关节（joint1）绕 X 轴旋转
- 第一段摆杆连接到驱动关节
- 第二段摆杆通过被动关节（joint2）连接到第一段摆杆

坐标系：
- joint_pos = 0: 摆杆自然下垂（朝下）
- joint_pos = π: 摆杆竖直向上（目标状态）
- 所有关节绕 X 轴旋转，摆杆在 YZ 平面内摆动
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .rotary_double_pendulum_cfg import RotaryDoublePendulumEnvCfg


class RotaryDoublePendulumEnv(DirectRLEnv):
    cfg: RotaryDoublePendulumEnvCfg

    def __init__(self, cfg: RotaryDoublePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._pend1_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum1_dof_name)
        self._pend2_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum2_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["double_pendulum"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions, joint_ids=self._pend1_dof_idx)

    def _get_observations(self) -> dict:
        pend1 = self.joint_pos[:, self._pend1_dof_idx[0]]
        pend2 = self.joint_pos[:, self._pend2_dof_idx[0]]

        pend1_vel = self.joint_vel[:, self._pend1_dof_idx[0]]
        pend2_vel = self.joint_vel[:, self._pend2_dof_idx[0]]

        # 观测：两段摆杆的 sin/cos 角度 + 归一化角速度（6维）
        obs = torch.cat(
            (
                torch.sin(pend1).unsqueeze(dim=1),
                torch.cos(pend1).unsqueeze(dim=1),
                torch.sin(pend2).unsqueeze(dim=1),
                torch.cos(pend2).unsqueeze(dim=1),
                (pend1_vel * 0.1).unsqueeze(dim=1),
                (pend2_vel * 0.1).unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pend1_pos = self.joint_pos[:, self._pend1_dof_idx[0]]
        pend2_pos = self.joint_pos[:, self._pend2_dof_idx[0]]
        pend1_vel = self.joint_vel[:, self._pend1_dof_idx[0]]
        pend2_vel = self.joint_vel[:, self._pend2_dof_idx[0]]
        
        return compute_rewards(
            pend1_pos,
            pend2_pos,
            pend1_vel,
            pend2_vel,
            self.actions,
            self.reset_terminated,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 不设置角度终止条件，让策略自由探索
        out_of_bounds = torch.zeros_like(time_out)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 初始状态：摆杆自然下垂（角度 ≈ 0），加随机扰动
        pend1_noise = sample_uniform(
            self.cfg.initial_pendulum1_angle_range[0],
            self.cfg.initial_pendulum1_angle_range[1],
            (len(env_ids), 1),
            joint_pos.device,
        )
        pend2_noise = sample_uniform(
            self.cfg.initial_pendulum2_angle_range[0],
            self.cfg.initial_pendulum2_angle_range[1],
            (len(env_ids), 1),
            joint_pos.device,
        )

        joint_pos[:, self._pend1_dof_idx] = pend1_noise  # 下垂 + 扰动
        joint_pos[:, self._pend2_dof_idx] = pend2_noise  # 下垂 + 扰动

        # 给一些初始速度帮助探索
        vel_noise1 = sample_uniform(-0.5, 0.5, (len(env_ids), 1), joint_vel.device)
        vel_noise2 = sample_uniform(-0.3, 0.3, (len(env_ids), 1), joint_vel.device)
        joint_vel[:, self._pend1_dof_idx] = vel_noise1
        joint_vel[:, self._pend2_dof_idx] = vel_noise2

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    pend1_pos: torch.Tensor,
    pend2_pos: torch.Tensor,
    pend1_vel: torch.Tensor,
    pend2_vel: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    """
    奖励设计：
    
    坐标系约定：
    - joint_pos = 0: 摆杆自然下垂（朝下）
    - joint_pos = π: 摆杆竖直向上（目标）
    
    奖励组成：
    1. 高度奖励：摆杆末端越高越好（-cos(θ) 在 θ=π 时最大）
    2. 平衡奖励：接近竖直向上时给予额外奖励
    3. 动作惩罚：节省能量
    """
    # 第一段摆杆末端高度
    # cos(0) = 1（朝下，最低），cos(π) = -1（朝上，最高）
    # 所以 -cos(θ) 在朝上时最大
    height1 = -torch.cos(pend1_pos)
    
    # 第二段摆杆末端的绝对角度 = pend1_pos + pend2_pos
    # 同样，-cos(θ1 + θ2) 在朝上时最大
    height2 = -torch.cos(pend1_pos + pend2_pos)
    
    # 总高度奖励：两段摆杆末端高度的加权和
    # 范围：[-2, 2]，向上时最大
    height_reward = height1 + height2
    
    # 归一化到 [0, 1] 范围
    normalized_height = (height_reward + 2.0) / 4.0
    rew_height = 2.0 * normalized_height
    
    # 平衡奖励：当两段摆杆都接近竖直向上时给予额外奖励
    # height1 > 0.8 表示 cos(θ1) < -0.8，即 θ1 接近 π
    near_upright = (height1 > 0.8) & (height2 > 0.8)
    rew_balance = torch.where(
        near_upright,
        5.0 * (height1 + height2 - 1.6),
        torch.zeros_like(height1)
    )
    
    # 接近平衡时的速度惩罚（希望稳定）
    rew_vel = torch.where(
        near_upright,
        -0.1 * (torch.abs(pend1_vel) + torch.abs(pend2_vel)),
        torch.zeros_like(pend1_vel)
    )
    
    # 动作惩罚（节省能量）
    rew_action = -0.001 * torch.sum(torch.square(actions), dim=-1)
    
    # 存活奖励
    rew_alive = 0.1 * (1.0 - reset_terminated.float())
    
    total_reward = rew_height + rew_balance + rew_vel + rew_action + rew_alive
    
    return total_reward
