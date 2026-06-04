#!/usr/bin/env python3
"""Runtime sanity dump for the 21D geometry-edge observation.

Boots IsaacLab with a tiny num_envs, teleports robot to a set of known poses,
forces an obs evaluation, and prints the 12D edge_obs side-by-side with the
math-only static reference values from scripts/verify_geo_edge_projection.py.

Pass criteria:
  - All 12 numbers per pose match within ~0.02 absolute tolerance
  - Edge ordering matches: edge_feat[0:6] = edge0 (-X end), edge_feat[6:12] = edge1 (+X end)
  - is_near flips at appropriate transitions

Usage:
    source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
    cd /data/jianshi/projects/forklift_sim/IsaacLab
    bash isaaclab.sh -p /data/jianshi/projects/forklift_sim_exp9/scripts/debug_geo_edge_obs.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = REPO_ROOT.parent / "forklift_sim" / "IsaacLab"
sys.path.insert(0, str(ISAACLAB_ROOT / "source"))
task_patch_path = (
    REPO_ROOT
    / "forklift_pallet_insert_lift_project"
    / "isaaclab_patch"
    / "source"
    / "isaaclab_tasks"
)
sys.path.insert(0, str(task_patch_path))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Runtime dump for 21D geometry-edge obs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if getattr(args, "enable_cameras", False):
    args.enable_cameras = False  # 21D path doesn't need rendering

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch  # noqa: E402

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import (  # noqa: E402
    ForkliftPalletInsertLiftEnv,
)
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import (  # noqa: E402
    ForkliftPalletInsertLiftGeoEdgeEnvCfg,
)


# 4 已知位姿 (root_x, root_y, yaw_deg)，配合 pallet 在原点
POSES = [
    ("far_field   x=-3.5", -3.5, 0.0, 0.0),
    ("mid_field   x=-2.5", -2.5, 0.0, 0.0),
    ("near_field  x=-1.5", -1.5, 0.0, 0.0),
    ("very_near   x=-1.20", -1.20, 0.0, 0.0),
]


# 静态自检的预期 (对应 scripts/verify_geo_edge_projection.py 输出)
# 每行: edge0_-Y_u, edge0_-Y_v, edge0_+Y_u, edge0_+Y_v, edge1_-Y_u, edge1_-Y_v, edge1_+Y_u, edge1_+Y_v
EXPECTED_UV = {
    "far_field   x=-3.5": [
        ( 0.176,  0.013),  # edge0 -Y endpoint
        (-0.176,  0.013),  # edge0 +Y endpoint
        ( 0.096, -0.115),  # edge1 -Y endpoint
        (-0.096, -0.115),  # edge1 +Y endpoint
    ],
    "mid_field   x=-2.5": [
        ( 0.285,  0.189),
        (-0.285,  0.189),
        ( 0.122, -0.074),
        (-0.122, -0.074),
    ],
    "near_field  x=-1.5": [
        ( 0.753,  0.941),
        (-0.753,  0.941),
        ( 0.166, -0.003),
        (-0.166, -0.003),
    ],
    "very_near   x=-1.20": [
        # edge0 端点静态自检显示越界（uv≈±1.48, +2.12），edge_obs 中应填 0
        (0.0, 0.0),
        (0.0, 0.0),
        ( 0.186,  0.029),
        (-0.186,  0.029),
    ],
}


def build_env_cfg(num_envs: int) -> ForkliftPalletInsertLiftGeoEdgeEnvCfg:
    cfg = ForkliftPalletInsertLiftGeoEdgeEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 3600.0
    cfg.wait_for_textures = False
    return cfg


def teleport(env: ForkliftPalletInsertLiftEnv, poses):
    device = env.device
    n = len(poses)
    env_ids = torch.arange(n, device=device, dtype=torch.long)

    # multi-env: pose 必须叠加 env_origins，否则 4 个 env 的 robot 与 pallet 不在同一小世界
    env_origins = env.scene.env_origins  # (N, 3) world frame
    positions = []
    quats = []
    for i, (_, rx, ry, yaw_deg) in enumerate(poses):
        yaw_rad = math.radians(yaw_deg)
        eo = env_origins[i].cpu().numpy()
        positions.append([float(eo[0]) + rx, float(eo[1]) + ry, 0.03])
        half = yaw_rad * 0.5
        quats.append([math.cos(half), 0.0, 0.0, math.sin(half)])

    # 必须先初始化 sim（首次需要），但只 reset 一次。后续 teleport 不再 sim.reset()
    # 否则 env.cfg.robot_cfg.init_state.pos 会覆盖我们写入的 pose。
    if not getattr(env, "_geoedge_dbg_init", False):
        env.sim.reset()
        env._geoedge_dbg_init = True

    pos_t = torch.tensor(positions, device=device, dtype=torch.float32)
    quat_t = torch.tensor(quats, device=device, dtype=torch.float32)
    env._write_root_pose(env.robot, pos_t, quat_t, env_ids)

    zeros3 = torch.zeros((n, 3), device=device)
    env._write_root_vel(env.robot, zeros3, zeros3, env_ids)

    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)

    env.scene.write_data_to_sim()
    env.scene.update(env.cfg.sim.dt)
    env.actions[env_ids] = 0.0
    if hasattr(env, "previous_actions"):
        env.previous_actions[env_ids] = 0.0
    env.episode_length_buf[env_ids] = 0


def main():
    out_path = Path("/tmp/geoedge_dump_result.txt")
    out_lines: list[str] = []

    def log(msg: str = ""):
        print(msg, flush=True)
        out_lines.append(msg)

    cfg = build_env_cfg(num_envs=len(POSES))
    log("[step] env constructing ...")
    try:
        env = ForkliftPalletInsertLiftEnv(cfg)
    except Exception as ex:  # noqa: BLE001
        log(f"[FATAL] env construction failed: {type(ex).__name__}: {ex}")
        out_path.write_text("\n".join(out_lines))
        raise
    log("[step] env constructed OK")
    try:
        try:
            teleport(env, POSES)
            log("[step] teleport done")
        except Exception as ex:  # noqa: BLE001
            log(f"[FATAL] teleport failed: {type(ex).__name__}: {ex}")
            raise

        # warm up sim a few steps so kinematic state is settled
        for _ in range(2):
            env.sim.step(render=False)
            env.scene.update(env.cfg.sim.dt)
        log("[step] warmed up sim")

        try:
            obs_dict = env._get_observations()
        except Exception as ex:  # noqa: BLE001
            log(f"[FATAL] _get_observations failed: {type(ex).__name__}: {ex}")
            raise
        obs = obs_dict["policy"]  # (N, 21)
        log(f"\n[obs.shape] = {tuple(obs.shape)}")

        # 打印 env 真实姿态以核对 multi-env origin offset
        rp = env.robot.data.root_pos_w.cpu().numpy()
        pp = env.pallet.data.root_pos_w.cpu().numpy()
        log("\n[real_world_pose]")
        for i, (tag, *_rest) in enumerate(POSES):
            log(
                f"  env[{i}] {tag}: robot_w=({rp[i,0]:+.3f},{rp[i,1]:+.3f},{rp[i,2]:+.3f}) "
                f" pallet_w=({pp[i,0]:+.3f},{pp[i,1]:+.3f},{pp[i,2]:+.3f}) "
                f" rel=({rp[i,0]-pp[i,0]:+.3f},{rp[i,1]-pp[i,1]:+.3f},{rp[i,2]-pp[i,2]:+.3f})"
            )

        # 打印 env_origins，确认 multi-env spacing 的影响
        if hasattr(env.scene, "env_origins"):
            eo = env.scene.env_origins.cpu().numpy()
            log("\n[env_origins]")
            for i in range(eo.shape[0]):
                log(f"  env[{i}] origin=({eo[i,0]:+.3f},{eo[i,1]:+.3f},{eo[i,2]:+.3f})")

        for idx, (tag, rx, _ry, _yaw) in enumerate(POSES):
            edge = obs[idx, :12].cpu().numpy()
            proprio = obs[idx, 12:].cpu().numpy()
            # edge layout: [edge0(u1,v1,u2,v2,vis,near), edge1(u1,v1,u2,v2,vis,near)]
            e0 = edge[:6]
            e1 = edge[6:]

            log(f"\n=== {tag}  (root_x={rx}) ===")
            log(
                f"  edge0 (-X side):  u1v1=({e0[0]:+.3f},{e0[1]:+.3f})"
                f"  u2v2=({e0[2]:+.3f},{e0[3]:+.3f})"
                f"  vis={e0[4]:.0f}  near={e0[5]:.0f}"
            )
            log(
                f"  edge1 (+X side):  u1v1=({e1[0]:+.3f},{e1[1]:+.3f})"
                f"  u2v2=({e1[2]:+.3f},{e1[3]:+.3f})"
                f"  vis={e1[4]:.0f}  near={e1[5]:.0f}"
            )
            exp = EXPECTED_UV[tag]
            log(f"  expected (static math, 4 endpoints):")
            log(f"    edge0 endpoints (-Y, +Y): {exp[0]}, {exp[1]}")
            log(f"    edge1 endpoints (-Y, +Y): {exp[2]}, {exp[3]}")

            err = []
            err.append(abs(e0[0] - exp[0][0]) + abs(e0[1] - exp[0][1]))  # edge0 -Y
            err.append(abs(e0[2] - exp[1][0]) + abs(e0[3] - exp[1][1]))  # edge0 +Y
            err.append(abs(e1[0] - exp[2][0]) + abs(e1[1] - exp[2][1]))  # edge1 -Y
            err.append(abs(e1[2] - exp[3][0]) + abs(e1[3] - exp[3][1]))  # edge1 +Y
            log(f"  L1 errs (4 endpoints): {[f'{e:.3f}' for e in err]}  max={max(err):.3f}")

            log(f"  proprio (9): {[f'{x:+.3f}' for x in proprio]}")

    finally:
        out_path.write_text("\n".join(out_lines))
        log(f"\n[step] dump written to {out_path}")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
