from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
URDF_PATH = os.path.join(ASSET_DIR, "rotary_double_pendulum.urdf")


ROTARY_DOUBLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=True,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        # 禁用 instanceable，确保每个环境有独立的视觉表示
        make_instanceable=False,
        # 禁用 fixed joint 合并，保持原始关节结构
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,  # 初始自然下垂
            "joint2": 0.0,  # 初始自然下垂
        },
    ),
    actuators={
        "motor_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.1,
        ),
    },
)


@configclass
class RotaryDoublePendulumEnvCfg(DirectRLEnvCfg):
    """二阶倒立摆环境配置"""
    
    # env
    decimation = 2
    episode_length_s = 10.0  # 缩短到 10 秒，鼓励快速稳定
    action_scale = 100.0
    action_space = 1
    observation_space = 6
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    # viewer (渲染多个环境时，拉远相机视角)
    viewer: ViewerCfg = ViewerCfg(
        eye=(12.0, 12.0, 8.0),
        lookat=(2.5, 2.5, 0.0),
    )

    # robot
    robot_cfg: ArticulationCfg = ROTARY_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    pendulum1_dof_name = "joint1"
    pendulum2_dof_name = "joint2"

    # scene - 训练模式（高性能）
    # 注意：make_instanceable=False 和 merge_fixed_joints=False 已确保正确渲染
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,   # 训练模式：高性能
        clone_in_fabric=True,     # 训练模式：高性能
    )

    # reset - 初始摆杆自然下垂（角度 = 0），加随机扰动
    initial_pendulum1_angle_range = (-0.3, 0.3)
    initial_pendulum2_angle_range = (-0.3, 0.3)


@configclass
class RotaryDoublePendulumEnvCfg_PLAY(RotaryDoublePendulumEnvCfg):
    """Play 模式配置 - 优化渲染效果
    
    与训练配置的区别：
    - 减少环境数量（16 vs 4096）
    - 禁用 replicate_physics 和 clone_in_fabric 以确保正确渲染
    """
    
    def __post_init__(self):
        super().__post_init__()
        # 减少环境数量，便于观察
        self.scene.num_envs = 16
        # 禁用物理复制，确保每个环境独立渲染
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
