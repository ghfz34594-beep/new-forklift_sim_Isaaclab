#!/usr/bin/env python3
"""
二阶倒立摆模型查看脚本

启动 Isaac Sim 并加载模型，让用户可以查看初始结构。
按空格键暂停/继续模拟，按 ESC 退出。
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# 命令行参数
parser = argparse.ArgumentParser(description="View Double Pendulum Model")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入其他模块（必须在 AppLauncher 之后）
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


def main():
    """主函数"""
    # 模型路径（优先使用当前项目的 patch 版本）
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "../isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/rotary_double_pendulum/assets/rotary_double_pendulum.urdf",
    )
    if not os.path.exists(urdf_path):
        isaaclab_dir = Path(os.environ.get("ISAACLAB_PATH", "/data/jianshi/projects/forklift_sim/IsaacLab"))
        urdf_path = str(
            isaaclab_dir
            / "source/isaaclab_tasks/isaaclab_tasks/direct/rotary_double_pendulum/assets/rotary_double_pendulum.urdf"
        )

    print(f"[INFO] Loading URDF from: {urdf_path}")

    # 仿真配置
    sim_cfg = SimulationCfg(dt=1 / 120, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 0.0, 2.0], [0.0, 0.0, 0.5])

    # 创建地面
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # 机器人配置
    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
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
        prim_path="/World/Robot",
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

    # 创建机器人
    robot = Articulation(robot_cfg)

    # 添加灯光
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 重置仿真
    sim.reset()
    robot.reset()

    print("\n" + "=" * 60)
    print("二阶倒立摆模型查看器")
    print("=" * 60)
    print("\n模型结构（电机悬浮在 z=1.2m）：")
    print("  - 蓝色圆柱：电机（悬浮固定）")
    print("  - 绿色摆杆：第一段摆杆（驱动关节，绕X轴旋转）")
    print("  - 橙色摆杆：第二段摆杆（被动关节）")
    print("  - 红色小球：末端标识")
    print("  - 银色圆柱：轴套")
    print("\n控制：")
    print("  - 关闭窗口或按 Ctrl+C 退出")
    print("  - 可以用鼠标旋转/缩放视角")
    print("=" * 60 + "\n")

    # 获取关节信息
    joint1_idx, _ = robot.find_joints("joint1")
    joint2_idx, _ = robot.find_joints("joint2")

    print(f"[INFO] Joint1 (驱动关节) index: {joint1_idx}")
    print(f"[INFO] Joint2 (被动关节) index: {joint2_idx}")

    # 运行仿真循环
    step_count = 0
    while simulation_app.is_running():
        # 不施加任何力矩，让摆杆自然下垂/摆动
        robot.set_joint_effort_target(
            torch.zeros(1, 1, device=sim.device),
            joint_ids=joint1_idx,
        )

        # 更新机器人数据
        robot.write_data_to_sim()

        # 执行仿真步进
        sim.step()

        # 更新机器人状态
        robot.update(sim.cfg.dt)

        step_count += 1

        # 每隔一段时间打印状态
        if step_count % 240 == 0:
            joint_pos = robot.data.joint_pos[0]
            joint_vel = robot.data.joint_vel[0]
            print(
                f"[Step {step_count}] Joint1: pos={joint_pos[joint1_idx[0]]:.3f} rad, "
                f"vel={joint_vel[joint1_idx[0]]:.3f} rad/s | "
                f"Joint2: pos={joint_pos[joint2_idx[0]]:.3f} rad, "
                f"vel={joint_vel[joint2_idx[0]]:.3f} rad/s"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        simulation_app.close()
