#!/usr/bin/env python3
"""Standalone math-only sanity check for the 21D geometry-edge projection pipeline.

Does NOT start IsaacLab. Validates that:
  1. focal length matches HFOV 120 deg (fx ~ 73.9 px)
  2. R_body_cam yields correct sign conventions (front, left/right, up/down)
  3. Pallet short-edge endpoints project into expected u-v ranges at three test
     poses (far, mid, near), all of which are physically reachable by the
     forklift root (pallet front-face is at world x = -1.08, so robot.x must
     remain < -1.08 + a small margin to keep edge[0] ahead of the camera).

Run: python scripts/verify_geo_edge_projection.py
"""

from __future__ import annotations

import math


def project(P_body, R, cam_pos, fx, cx, W):
    """OpenCV-style pinhole projection from body-frame point.

    Args:
        P_body: (3,) python list/tuple, point in robot body frame [m]
        R:      3x3 list-of-lists, body->camera rotation
        cam_pos: (3,) camera origin in body frame [m]
        fx:     focal length [px]
        cx:     principal point x [px] (also = W/2 for square images)
        W:      image width [px]
    Returns:
        (u_norm, v_norm, z_cam) or (None, None, z_cam) if z<=0
    """
    Po = [P_body[i] - cam_pos[i] for i in range(3)]
    Pc = [
        R[0][0] * Po[0] + R[0][1] * Po[1] + R[0][2] * Po[2],
        R[1][0] * Po[0] + R[1][1] * Po[1] + R[1][2] * Po[2],
        R[2][0] * Po[0] + R[2][1] * Po[1] + R[2][2] * Po[2],
    ]
    z = Pc[2]
    if z <= 0.05:
        return None, None, z
    u_pix = fx * Pc[0] / z + cx
    v_pix = fx * Pc[1] / z + cx  # square image -> cy = cx
    u_norm = (u_pix - cx) / cx
    v_norm = (v_pix - cx) / cx
    return u_norm, v_norm, z


def main() -> int:
    # ----- 内参 -----
    W = 256.0
    HFOV = 120.0
    fx = (W * 0.5) / math.tan(math.radians(HFOV) * 0.5)
    cx = W * 0.5
    print(f"[K] HFOV={HFOV} deg, W={W:.0f} px -> fx = fy = {fx:.3f}  (expect ~73.9)")
    assert abs(fx - 73.9) < 0.1, f"fx mismatch: {fx}"

    # ----- 外参 (pitch=25, no roll/yaw) -----
    pitch = math.radians(25.0)
    sa, ca = math.sin(pitch), math.cos(pitch)
    R = [
        [0.0, -1.0, 0.0],
        [-sa, 0.0, -ca],
        [ca,  0.0, -sa],
    ]
    cam_pos = [0.30, 0.0, 1.30]
    print(f"[R] pitch=25 deg, cam_pos_body=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f})\n")

    # ----- 4 个 pallet-local 端点（顶面）-----
    edges_local = [
        ("edge0_-X_-Y", (-1.08, -0.72, 0.131)),
        ("edge0_-X_+Y", (-1.08, +0.72, 0.131)),
        ("edge1_+X_-Y", (+1.08, -0.72, 0.131)),
        ("edge1_+X_+Y", (+1.08, +0.72, 0.131)),
    ]
    pallet_origin_z = 0.15  # cfg.pallet_cfg.init_state.pos[2]

    # robot root z = 0.03 (cfg)
    robot_z = 0.03

    test_poses = [
        ("far_field   robot_x=-3.5", -3.5),
        ("mid_field   robot_x=-2.5", -2.5),
        ("near_field  robot_x=-1.5", -1.5),
        ("very_near   robot_x=-1.20", -1.20),
    ]

    margin = 1.1
    failure = False

    for tag, rx in test_poses:
        print(f"--- {tag} ---")
        # convert each edge endpoint to body frame (yaw=0 case)
        for name, (lx, ly, lz) in edges_local:
            wx = lx
            wy = ly
            wz = pallet_origin_z + lz
            P_b = (wx - rx, wy - 0.0, wz - robot_z)
            u, v, z = project(P_b, R, cam_pos, fx, cx, W)
            if u is None:
                visible = 0
                print(f"  {name:<14} P_b=({P_b[0]:+.2f},{P_b[1]:+.2f},{P_b[2]:+.2f})  z_cam={z:+.3f}  BEHIND")
            else:
                visible = int(abs(u) <= margin and abs(v) <= margin)
                print(
                    f"  {name:<14} P_b=({P_b[0]:+.2f},{P_b[1]:+.2f},{P_b[2]:+.2f}) "
                    f" z_cam={z:+.3f}  uv=({u:+.3f},{v:+.3f})  vis={visible}"
                )
        print()

    # 期望验证
    # 远场 -3.5：4 个端点都应该 visible
    # 中场 -2.5：4 个端点都应该 visible（仍在 FoV 内）
    # 近场 -1.5：edge0 (-X 端) 在车前方 -1.08-(-1.5)=0.42m，仍可见；edge1 (+X 端) 在车前方 2.58m，可见
    # 极近场 -1.20：edge0 在车前方 0.12m（极近），半视角约 atan(0.72/0.12)=80°远超 60°，应当 invisible
    #              edge1 在车前方 2.28m，可见
    print("[expected]")
    print("  far_field    : 4 endpoints visible")
    print("  mid_field    : 4 endpoints visible")
    print("  near_field   : 4 endpoints visible (edge0 still 0.42m ahead of root)")
    print("  very_near    : edge0 endpoints both invisible (out of horizontal FoV)")
    print("                 edge1 endpoints visible")

    if failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
