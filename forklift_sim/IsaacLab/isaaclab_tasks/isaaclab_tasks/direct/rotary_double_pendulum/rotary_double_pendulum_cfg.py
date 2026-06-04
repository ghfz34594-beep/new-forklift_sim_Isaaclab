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
    episode_length_s = 20.0
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,  # 增加并行环境数以充分利用GPU
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # reset - 初始摆杆自然下垂（角度 = 0），加随机扰动
    initial_pendulum1_angle_range = (-0.3, 0.3)
    initial_pendulum2_angle_range = (-0.3, 0.3)
