"""
ForkliftController — 线程安全的叉车命令/状态中枢。

仿真主线程（Isaac Sim）每帧调用 apply_to_articulation() 读取命令并写回状态。
Web 服务线程通过 set_command() / get_state() 与之通信。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ForkliftCommand:
    """归一化控制命令，所有分量均在 [-1, 1]。"""
    drive: float = 0.0    # 前进(+) / 后退(-)
    steer: float = 0.0    # 左转(+) / 右转(-)
    lift: float = 0.0     # 上升(+) / 下降(-)


@dataclass
class ForkliftState:
    """叉车当前状态（由仿真线程写入）。"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0
    lift_height_m: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# 关节缩放常量（与 env_cfg.py 一致）
WHEEL_SPEED_MAX: float = 20.0   # rad/s
STEER_ANGLE_MAX: float = 0.6    # rad
LIFT_SPEED_MAX: float = 0.25    # m/s


class ForkliftController:
    """
    线程安全的叉车控制器。

    Web 线程调用 set_command()，仿真线程调用 apply_to_articulation()。
    两者通过 threading.Lock 隔离。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cmd = ForkliftCommand()
        self._state = ForkliftState()
        self._ready = False   # 仿真场景是否完成初始化

        # 关节索引缓存（由 main.py 在场景初始化后填充）
        self._front_wheel_indices: Optional[list[int]] = None
        self._back_wheel_indices: Optional[list[int]] = None
        self._rotator_indices: Optional[list[int]] = None
        self._lift_index: Optional[int] = None

        # ArticulationView 引用（由 main.py 填充）
        self._articulation = None

    # ------------------------------------------------------------------
    # 公共接口：Web 线程调用
    # ------------------------------------------------------------------

    def set_command(self, drive: float, steer: float, lift: float) -> None:
        """设置叉车控制命令（归一化 [-1, 1]）。"""
        drive = max(-1.0, min(1.0, drive))
        steer = max(-1.0, min(1.0, steer))
        lift = max(-1.0, min(1.0, lift))
        with self._lock:
            self._cmd = ForkliftCommand(drive=drive, steer=steer, lift=lift)

    def get_state(self) -> ForkliftState:
        """获取当前叉车状态快照。"""
        with self._lock:
            return ForkliftState(
                x=self._state.x,
                y=self._state.y,
                z=self._state.z,
                yaw_deg=self._state.yaw_deg,
                lift_height_m=self._state.lift_height_m,
                vx=self._state.vx,
                vy=self._state.vy,
                vz=self._state.vz,
                timestamp=self._state.timestamp,
            )

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    # ------------------------------------------------------------------
    # 仿真线程调用
    # ------------------------------------------------------------------

    def register_articulation(
        self,
        articulation,
        front_wheel_indices: list[int],
        back_wheel_indices: list[int],
        rotator_indices: list[int],
        lift_index: int,
    ) -> None:
        """在仿真场景初始化完成后，由 main.py 注册 ArticulationView 及关节索引。"""
        with self._lock:
            self._articulation = articulation
            self._front_wheel_indices = front_wheel_indices
            self._back_wheel_indices = back_wheel_indices
            self._rotator_indices = rotator_indices
            self._lift_index = lift_index
            self._ready = True

    def apply_to_articulation(self) -> None:
        """
        每个仿真帧调用一次：
        1. 将当前命令转换为关节目标并写入 ArticulationView。
        2. 从 ArticulationView 读取当前状态并更新 _state。
        """
        with self._lock:
            if not self._ready or self._articulation is None:
                return

            art = self._articulation
            cmd = self._cmd

            import numpy as np
            import math

            num_dof = art.num_dof

            # ---------- 计算关节目标 ----------
            drive_vel = cmd.drive * WHEEL_SPEED_MAX    # rad/s
            steer_pos = cmd.steer * STEER_ANGLE_MAX    # rad
            lift_vel  = cmd.lift  * LIFT_SPEED_MAX     # m/s

            # 速度目标数组（shape: [1, num_dof]），默认 0
            vel_targets = np.zeros((1, num_dof), dtype=np.float32)
            for idx in self._front_wheel_indices:
                vel_targets[0, idx] = drive_vel
            for idx in self._back_wheel_indices:
                vel_targets[0, idx] = drive_vel
            vel_targets[0, self._lift_index] = lift_vel
            art.set_joint_velocity_targets(vel_targets)

            # 位置目标数组（shape: [1, num_dof]），默认 0
            pos_targets = np.zeros((1, num_dof), dtype=np.float32)
            for idx in self._rotator_indices:
                pos_targets[0, idx] = steer_pos
            art.set_joint_position_targets(pos_targets)

            # ---------- 读取状态 ----------
            try:
                pos, rot = art.get_world_poses()
                joint_pos = art.get_joint_positions()

                if pos is not None and pos.shape[0] > 0:
                    self._state.x = float(pos[0, 0])
                    self._state.y = float(pos[0, 1])
                    self._state.z = float(pos[0, 2])

                if rot is not None and rot.shape[0] > 0:
                    # Isaac Sim ArticulationView 四元数格式为 (w, x, y, z)
                    w = float(rot[0, 0])
                    x = float(rot[0, 1])
                    y = float(rot[0, 2])
                    z = float(rot[0, 3])
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    self._state.yaw_deg = math.degrees(yaw)

                # 线速度：优先使用 get_linear_velocities，回退到 get_velocities
                try:
                    lin_vel = art.get_linear_velocities()
                    if lin_vel is not None and lin_vel.shape[0] > 0:
                        self._state.vx = float(lin_vel[0, 0])
                        self._state.vy = float(lin_vel[0, 1])
                        self._state.vz = float(lin_vel[0, 2])
                except AttributeError:
                    vel = art.get_velocities()
                    if vel is not None and vel.shape[0] > 0:
                        self._state.vx = float(vel[0, 0])
                        self._state.vy = float(vel[0, 1])
                        self._state.vz = float(vel[0, 2])

                if joint_pos is not None and joint_pos.shape[0] > 0:
                    self._state.lift_height_m = float(joint_pos[0, self._lift_index])

                self._state.timestamp = time.time()
            except Exception:
                pass
