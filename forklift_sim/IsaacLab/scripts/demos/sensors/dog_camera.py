# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load the UNI dog asset and attach a front camera to its base link.

Run from the IsaacLab root:

    ./isaaclab.sh -p scripts/demos/sensors/dog_camera.py --enable_cameras
"""

import argparse
import sys
from pathlib import Path


ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
for extension_path in ISAACLAB_ROOT.joinpath("source").iterdir():
    if extension_path.is_dir():
        sys.path.append(str(extension_path))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attach a camera to the UNI dog asset.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


UNI_DOG_USD_PATH = "/home/uniubi/xuanyuan/dog/uni_0428.usd"
LEFT_CAMERA_POS = (0.23286, -0.00665, 0.01343)
STEREO_BASELINE_M = 0.06
RIGHT_CAMERA_POS = (LEFT_CAMERA_POS[0], LEFT_CAMERA_POS[1] - STEREO_BASELINE_M, LEFT_CAMERA_POS[2])
CAMERA_ROT = (0.5, -0.5, 0.5, -0.5)
CAMERA_SPAWN_CFG = sim_utils.PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 100.0),
)


UNI_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UNI_DOG_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    articulation_root_prim_path="/BASE_LINK",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_ABAD_JOINT": 0.0,
            ".*_HIP_JOINT": 0.75,
            ".*_KNEE_JOINT": -1.55,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_(ABAD|HIP|KNEE)_JOINT"],
            effort_limit_sim=30.0,
            velocity_limit_sim=20.0,
            stiffness=None,
            damping=None,
        )
    },
    soft_joint_pos_limit_factor=0.9,
)


@configclass
class DogCameraSceneCfg(InteractiveSceneCfg):
    """Scene with the UNI dog and a camera attached to BASE_LINK."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = UNI_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK/left_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=CAMERA_SPAWN_CFG,
        offset=CameraCfg.OffsetCfg(
            pos=LEFT_CAMERA_POS,
            rot=CAMERA_ROT,
            convention="ros",
        ),
    )

    right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK/right_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=CAMERA_SPAWN_CFG,
        offset=CameraCfg.OffsetCfg(
            pos=RIGHT_CAMERA_POS,
            rot=CAMERA_ROT,
            convention="ros",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Keep the robot at its default pose while the camera can be adjusted in the UI."""

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting UNI dog state.")

        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        if count % 100 == 0:
            left_rgb = scene["left_camera"].data.output["rgb"]
            left_depth = scene["left_camera"].data.output["distance_to_image_plane"]
            right_rgb = scene["right_camera"].data.output["rgb"]
            right_depth = scene["right_camera"].data.output["distance_to_image_plane"]
            print(f"[INFO]: left_camera rgb={tuple(left_rgb.shape)}, depth={tuple(left_depth.shape)}")
            print(f"[INFO]: right_camera rgb={tuple(right_rgb.shape)}, depth={tuple(right_depth.shape)}")


def main():
    """Run the camera adjustment scene."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.6, 1.4, 0.9], target=[0.0, 0.0, 0.25])

    scene_cfg = DogCameraSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")
    print("[INFO]: Left camera path:  /World/envs/env_0/Robot/BASE_LINK/left_cam")
    print("[INFO]: Right camera path: /World/envs/env_0/Robot/BASE_LINK/right_cam")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
