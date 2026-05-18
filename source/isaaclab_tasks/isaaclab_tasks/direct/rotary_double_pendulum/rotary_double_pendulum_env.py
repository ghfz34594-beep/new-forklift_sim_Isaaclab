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

        # 目标时间：6秒内达到稳定
        # 环境步长 = 1/120 秒，所以 6 秒 = 720 步
        self.target_steps = 720

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 根据 replicate_physics 决定克隆方式：
        # - replicate_physics=True (训练模式): copy_from_source=False (高性能)
        # - replicate_physics=False (Play模式): copy_from_source=True (正确渲染)
        copy_from_source = not self.cfg.scene.replicate_physics
        self.scene.clone_environments(copy_from_source=copy_from_source)
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
            self.episode_length_buf,
            self.target_steps,
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
    episode_step: torch.Tensor,
    target_steps: int,
):
    """
    奖励设计（快速稳定版 - 目标：6秒内达到稳定）

    坐标系约定：
    - joint_pos = 0: 摆杆自然下垂（朝下）
    - joint_pos = π: 摆杆竖直向上（目标）

    关键设计：
    1. 时间衰减奖励：越早达到高位，奖励越高
    2. 早期稳定大奖励：6秒内稳定给额外奖励
    3. 高位稳定奖励：鼓励保持平衡
    """
    # ============================================================
    # 1. 高度计算
    # ============================================================
    height1 = -torch.cos(pend1_pos)
    height2 = -torch.cos(pend1_pos + pend2_pos)
    total_height = height1 + height2
    normalized_height = (total_height + 2.0) / 4.0

    # ============================================================
    # 2. 时间衰减因子（关键：鼓励快速达到目标）
    # ============================================================
    # 6秒(720步)后衰减到 0.2，之后保持 0.2
    time_progress = episode_step.float() / float(target_steps)
    time_factor = torch.clamp(1.0 - 0.8 * time_progress, min=0.2)

    # ============================================================
    # 3. 基础高度奖励（带时间衰减）
    # ============================================================
    rew_height = 1.0 * normalized_height * time_factor

    # ============================================================
    # 4. 稳定平衡奖励（带时间衰减）
    # ============================================================
    upright1 = (1.0 - torch.cos(pend1_pos)) / 2.0
    upright2 = (1.0 - torch.cos(pend1_pos + pend2_pos)) / 2.0
    upright_score = upright1 * upright2

    # 稳定奖励乘以时间因子
    rew_upright = 4.0 * upright_score * time_factor

    # ============================================================
    # 5. 早期稳定大奖励（6秒内达到稳定）
    # ============================================================
    total_vel = torch.abs(pend1_vel) + torch.abs(pend2_vel)
    
    # 稳定条件：高度 > 0.85 且 速度 < 2.0
    is_stable = (normalized_height > 0.85) & (total_vel < 2.0)
    is_early = episode_step < target_steps

    # 早期稳定奖励：越早越大（最高 15.0）
    early_bonus_value = 15.0 * (1.0 - time_progress)
    early_stable_bonus = torch.where(
        is_stable & is_early,
        early_bonus_value,
        torch.zeros_like(normalized_height)
    )

    # ============================================================
    # 6. 高位额外奖励
    # ============================================================
    high_bonus = torch.where(
        normalized_height > 0.9,
        8.0 * (normalized_height - 0.9) / 0.1,  # 最高 8.0
        torch.zeros_like(normalized_height)
    )

    # ============================================================
    # 7. 条件速度惩罚（高位时鼓励稳定）
    # ============================================================
    rew_vel = torch.where(
        normalized_height > 0.75,
        -0.01 * total_vel,  # 高位时惩罚速度
        torch.zeros_like(total_vel)
    )

    # ============================================================
    # 8. 动作惩罚
    # ============================================================
    rew_action = -0.0005 * torch.sum(torch.square(actions), dim=-1)

    # ============================================================
    # 9. 存活奖励
    # ============================================================
    rew_alive = 0.05 * (1.0 - reset_terminated.float())

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = (
        rew_height +          # 基础高度（时间衰减）
        rew_upright +         # 稳定平衡（时间衰减）
        early_stable_bonus +  # 早期稳定大奖励
        high_bonus +          # 高位奖励
        rew_vel +             # 速度惩罚
        rew_action +          # 动作惩罚
        rew_alive             # 存活奖励
    )

    return total_reward
