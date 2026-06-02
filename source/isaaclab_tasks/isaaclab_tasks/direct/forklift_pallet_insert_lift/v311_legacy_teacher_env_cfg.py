"""Forklift Pallet Insert+Lift 环境配置（DirectRLEnvCfg）。

此文件集中定义仿真环境的所有可调参数：环境步长、场景复制、资产配置、
奖励系数、任务 KPI 等。训练时由 `gym.register()` 中的
`env_cfg_entry_point` 自动加载并实例化。
"""

from __future__ import annotations

import os
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _find_repo_root() -> str | None:
    """Resolve the shared project root from either source tree layout."""
    here = os.path.abspath(os.path.dirname(__file__))
    for levels_up in (6, 7):
        root = os.path.abspath(os.path.join(here, *([".."] * levels_up)))
        if os.path.isfile(os.path.join(root, "assets", "forklift_c.usd")):
            return root
    return None


def _prefer_local_usd(local_name: str, remote_path: str) -> str:
    """Use a checked-in local USD when available, otherwise fall back to Nucleus."""
    repo_root = _find_repo_root()
    if repo_root is None:
        return remote_path
    local_path = os.path.join(repo_root, "assets", local_name)
    return local_path if os.path.isfile(local_path) else remote_path


_DEFAULT_FORKLIFT_USD_PATH = _prefer_local_usd(
    "forklift_c.usd",
    f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd",
)


@configclass
class ForkliftPalletInsertLiftEnvCfg(DirectRLEnvCfg):
    """Configuration for the Forklift Pallet Insert+Lift environment (direct workflow).

    S1.0h: 修复仿真环境 + 对齐学习闭环 + 奖励泵漏洞修复。
    S1.0h 环境修正：
      - 托盘缩放 4.0x → 1.8x（匹配 isaac_sim_asset_import.md 文档）
      - pallet_depth_m 4.8 → 2.16（1.2m × 1.8）
      - 托盘质量 30kg → 45kg，初始高度 0.30 → 0.15
      - 添加碰撞接触参数 collision_props（contactOffset/restOffset）
      - 修复 _pallet_front_x 符号 bug（+ → -，指向 pocket 开口而非远端面）
    S1.0h 奖励修正：
      - r_approach 软门控（0.2 + 0.8*w_ready），打破对齐-接近死锁
      - 对齐阈值随距离插值（远处松、近处紧）
      - yaw 目标分两段（远处对准指向方向，近处对准托盘朝向）
      - r_forward 绑定到朝向因子 w_yaw
      - r_insert / r_lift 势函数 shaping（带 gamma=0.99），修复奖励泵漏洞
      - reset 时正确初始化势能缓存
    """

    # ===== 环境基础参数 =====
    decimation = 4
    episode_length_s = 36.0  # was 12.0; ~1080 steps for expert to complete insertion+lift

    # actions: [drive, steer]（驾驶、转向，Stage 1 纯 approach 策略）
    action_space = 2

    # 默认走视频端到端训练：
    # - actor 观测是 image + proprio
    # - critic 观测是 15 维低维 privileged state
    observation_space = {"image": [3, 64, 64], "proprio": 8}
    state_space = 15

    # ===== Video-e2e defaults =====
    use_camera: bool = True
    # Toyota-style policy path: two side cameras + 5D proprio
    # [v_x, v_y, yaw_rate, previous_drive, previous_steer].
    # The base task keeps this off for backwards compatibility; the
    # ForkliftPalletApproachToyotaDualCameraEnvCfg subclass enables it.
    use_dual_cameras: bool = False
    use_asymmetric_critic: bool = True
    stage_1_mode: bool = True
    # Stage 1 课程只训练“接近 + 对齐 + 插入”，默认不要求举升进入 success。
    # 这样可以避免“动作层锁 lift，但成功判定仍要求 lift”导致 success 永远为 0。
    stage1_success_without_lift: bool = True
    # 禁用等待贴图加载，避免因 PNG 解析报错导致仿真启动挂起
    wait_for_textures: bool = False

    # Stage 1 初始随机化范围。
    # Exp9.0: 对齐 master 的原始分布，便于和“无参考轨迹”基准直接对照：
    # - x ∈ [-2.5, -1.0]
    # - y ∈ [-0.6, 0.6]
    # - yaw ∈ [-0.25, 0.25] rad ≈ ±14.3239 deg
    stage1_init_x_min_m: float = -4.0
    stage1_init_x_max_m: float = -3.0
    stage1_init_y_min_m: float = -0.6
    stage1_init_y_max_m: float = 0.6
    stage1_init_yaw_deg_min: float = -14.32394487827058
    stage1_init_yaw_deg_max: float = 14.32394487827058
    stage1_near_hard_curriculum_enable: bool = False
    stage1_near_hard_curriculum_frac: float = 0.0
    stage1_near_hard_curriculum_start_step: int = 0
    stage1_near_hard_curriculum_ramp_steps: int = 1
    stage1_near_hard_x_min_m: float = -3.35
    stage1_near_hard_x_max_m: float = -3.00
    stage1_near_hard_y_abs_min_m: float = 0.30
    stage1_near_hard_y_abs_max_m: float = 0.60
    stage1_near_hard_yaw_abs_min_deg: float = 8.0
    stage1_near_hard_yaw_abs_max_deg: float = 14.32394487827058
    stage1_near_hard_lateral_frac: float = 0.5
    stage1_near_hard_positive_y_frac: float = 0.5
    stage1_near_hard_positive_yaw_frac: float = 0.5
    stage1_near_hard_opposite_yaw_frac: float = 0.0

    # 相机参数：
    # - 训练默认 256x256，进一步提升视觉特征提取精度
    # - camera_eval.py 会显式覆盖到 320x320 做可视化验收
    camera_width: int = 256
    camera_height: int = 256
    camera_hfov_deg: float = 90.0
    camera_mount_body: str = "body"
    # forklift_c.usd 使用 cm 单位；TiledCamera offset 也需使用挂载 prim 的局部单位。
    # 这里的值等价于前方 1.3m、上方 2.5m。
    camera_pos_local: tuple[float, float, float] = (130.0, 0.0, 250.0)
    # 在 world 约定下，pitch=+45° 表示相机向下俯视 45°。
    camera_rpy_local_deg: tuple[float, float, float] = (0.0, 75.0, 0.0)

    # easy8 + privileged 维度（供 obs 组装使用）
    easy8_dim: int = 8
    toyota_proprio_dim: int = 5
    privileged_dim: int = 15

    # 跟随挂载相机（strict: 必须挂在 Robot/<mount_body> 下，不做 world fallback）
    # rot 会在 env._setup_scene() 中根据 camera_rpy_local_deg 运行时覆盖。
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot/{camera_mount_body}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=camera_pos_local,
            rot=(0.9238795, 0.0, 0.3826834, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 40.0),
        ),
        width=camera_width,
        height=camera_height,
    )

    # Toyota-style dual side cameras.  forklift_c.usd uses cm for local offsets,
    # so these values are centimeter offsets under Robot/<camera_mount_body>.
    dual_camera_width: int = 224
    dual_camera_height: int = 224
    dual_camera_hfov_deg: float = 60.0
    dual_camera_left_pos_local: tuple[float, float, float] = (120.0, 55.0, 150.0)
    dual_camera_right_pos_local: tuple[float, float, float] = (120.0, -55.0, 150.0)
    dual_camera_left_rpy_local_deg: tuple[float, float, float] = (0.0, 68.0, -8.0)
    dual_camera_right_rpy_local_deg: tuple[float, float, float] = (0.0, 68.0, 8.0)

    tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot/{camera_mount_body}/CameraLeft",
        offset=TiledCameraCfg.OffsetCfg(
            pos=dual_camera_left_pos_local,
            rot=(0.8290376, 0.0400333, 0.5563076, -0.0596425),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 40.0),
        ),
        width=dual_camera_width,
        height=dual_camera_height,
    )
    tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot/{camera_mount_body}/CameraRight",
        offset=TiledCameraCfg.OffsetCfg(
            pos=dual_camera_right_pos_local,
            rot=(0.8290376, -0.0400333, 0.5563076, 0.0596425),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 40.0),
        ),
        width=dual_camera_width,
        height=dual_camera_height,
    )

    # Toyota-style robustness knobs.  These are lightweight, deterministic
    # implementation hooks for the paper-aligned path; visual material
    # randomization remains a later renderer-specific extension.
    toyota_action_noise_std: float = 0.0
    toyota_velocity_obs_noise_std: float = 0.0
    stage1_use_triangular_visible_init: bool = False
    stage1_tri_x_near_m: float = -2.4
    stage1_tri_x_far_m: float = -4.0
    stage1_tri_y_half_width_near_m: float = 0.20
    stage1_tri_y_half_width_far_m: float = 0.70
    stage1_tri_yaw_deg_near: float = 6.0
    stage1_tri_yaw_deg_far: float = 16.0

    # Rule lift sequence parameters used by the non-web API and rollout scripts.
    scripted_lift_target_m: float = 0.35
    scripted_lift_drive_reverse: float = -0.35
    scripted_lift_lift_action: float = 1.0
    scripted_lift_lower_action: float = -0.7
    scripted_lift_steps: int = 90
    scripted_reverse_steps: int = 75
    scripted_lower_steps: int = 90
    loading_decision_stop_delay_s: float = 3.0

    # ===== 仿真参数 =====
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ===== 场景复制与并行环境 =====
    # clone_in_fabric=False: 修复 body_pos_w 全部等于 root_pos_w 的问题
    # 原因：Fabric clone 失败导致 body link 位置不追踪（见诊断报告）
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=6.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )

    # ===== 视觉多 env 隔离房间 =====
    # Isaac camera renders the shared USD world instead of clipping by env_id.
    # These walls are cloned under each /World/envs/env_N/Room to occlude
    # neighboring environments for RGB / ResNet training.
    vision_room_enable: bool = False
    vision_room_collision_enable: bool = True
    vision_room_length_m: float = 10.0
    vision_room_width_m: float = 8.0
    vision_room_height_m: float = 3.0
    vision_room_wall_thickness_m: float = 0.15
    vision_room_center_x_m: float = -1.5
    vision_room_center_y_m: float = 0.0
    vision_room_color: tuple[float, float, float] = (0.55, 0.58, 0.60)

    # ===== 资产路径 =====
    forklift_usd_path: str = _DEFAULT_FORKLIFT_USD_PATH
    pallet_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"

    # ===== 托盘几何参数（欧标托盘，按 1.8x 缩放）=====
    # 原始深度 1.2m × 1.8 = 2.16m（参考 docs/learning_guiding/isaac_sim_asset_import.md）
    pallet_depth_m: float = 2.16

    # ===== KPI（成功判定指标）=====
    # S1.0k sanity check 发现：convexDecomposition 碰撞体阻止叉齿插入超过 ~1.03m，
    # 原 2/3 (1.44m) 阈值物理不可达。降低到 0.40 (0.864m)，留 15% 安全余量。
    # 详见 docs/diagnostic_reports/success_sanity_check_2026-02-10.md
    insert_fraction: float = 0.40
    lift_delta_m: float = 0.3      # S1.0T: 0.25→1.0, 防作弊计划降低门槛 1.0→0.3
    # S1.0M: hold_time_s 从 1.0 降到 0.33（hold_steps: 30→~10）。
    # sanity check A2 显示即使理论成功位姿，hold_counter 最高 4/30 即断，
    # 物理抖动使 30 步连续保持几乎不可能。课程阶段先出 success 再收紧。
    hold_time_s: float = 0.33
    # S1.0M: 放宽成功对齐阈值（课程学习起点）。
    # 当前 lateral≈0.25m（需 ≤0.03m，差 8 倍），yaw≈7.5°（需 ≤3.0°，差 2.5 倍）。
    # 先让 success 出现，后续逐步收紧。
    max_lateral_err_m: float = 0.15
    max_yaw_err_deg: float = 8.0        # Stage1 logic smoke 1: relax hold yaw gate 5→8
    # S1.0N: hold counter 全维度 Schmitt trigger（抗物理抖动）
    hysteresis_ratio: float = 1.2       # 对齐 exit 阈值 = entry × 1.2
    insert_exit_epsilon: float = 0.02   # 插入深度 exit 容差（与 insert_depth 同单位）
    lift_exit_epsilon: float = 0.08     # S1.0T: 0.02→0.08 等比放大 (m)
    # S1.0O-C2: hold counter 衰减（越界不清零，改为 *= decay）
    hold_counter_decay: float = 0.8
    
    # 实验 0：push-free 判定阈值
    push_free_disp_thresh_m: float = 0.05

    # ---- model_1999-style potential approach reward (default-off) ----
    # This is a clean reward path for the Toyota 2D approach teacher.  It reuses
    # the proven 15D baseline's stage potential structure, but drops lift from
    # the policy objective so the 2D action space and success definition agree.
    use_potential_approach_reward: bool = False
    k_pallet_push_pen: float = 1.0
    pallet_push_insert_gate: float = 0.15
    pallet_push_insert_ramp: float = 0.15
    pallet_push_deadband_m: float = 0.05
    pallet_push_penalty_full_episode: bool = False
    k_pallet_push_pen_quadratic: float = 0.0
    push_free_terminal_gate_enable: bool = False
    push_free_terminal_bonus: float = 0.0
    dirty_success_reward_scale: float = 0.0
    dirty_success_terminal_penalty: float = 0.0
    use_explicit_progress_reward: bool = False
    k_approach_progress: float = 0.0
    k_insert_progress: float = 0.0
    approach_progress_insert_stop_frac: float = 0.08
    approach_progress_align_floor: float = 0.20
    insert_progress_center_sigma_m: float = 0.12
    insert_progress_tip_sigma_m: float = 0.14
    insert_progress_yaw_sigma_deg: float = 7.0
    insert_progress_push_sigma_m: float = 0.08
    # ---- Exp8.3 clean insert / hold: post-insert reward gate ----
    clean_insert_reward_gate_enable: bool = True
    clean_insert_gate_start_frac: float = 0.25   # 开始对 post-insert 正奖励做 clean gate
    clean_insert_gate_ramp_frac: float = 0.15    # 0.25 -> 0.40 逐步切到 clean gate 主导
    clean_insert_gate_floor: float = 0.15        # 回到 B0：dirty insert 时保留少量正奖励，避免直接压死 insertion
    clean_insert_center_sigma_m: float = 0.10    # 回到 B0：fork center 横向 clean 尺度
    clean_insert_yaw_sigma_deg: float = 6.0      # 回到 B0：clean insert 偏航尺度
    clean_insert_tip_sigma_m: float = 0.10       # 回到 B0：fork tip 横向 clean 尺度
    clean_insert_use_push_gate: bool = True
    clean_insert_push_sigma_m: float = 0.10      # 回到 B0：托盘位移越大，post-insert 正奖励衰减越强
    clean_insert_gate_r_cd: bool = False         # 回到 B0：不额外 gate r_cd
    clean_insert_gate_r_cpsi: bool = False       # 回到 B0：不额外 gate r_cpsi
    clean_insert_dirty_penalty_enable: bool = False
    clean_insert_dirty_penalty_weight: float = 8.0
    clean_insert_push_free_bonus_enable: bool = True
    clean_insert_push_free_bonus_weight: float = 1.0
    # Default-off Stage A repair knob: when enabled, training success reward and
    # termination require the same push-free condition used by eval diagnostics.
    push_free_training_success_enable: bool = False
    push_free_training_use_max_disp: bool = False
    push_free_dirty_success_penalty_weight: float = 10.0
    # Default-off clean-first repair: make the eval push-free / dirty-insert
    # categories visible before final success, instead of rewarding dirty
    # geometry until the terminal success gate.
    push_free_rg_gate_enable: bool = False
    dirty_insert_early_penalty_enable: bool = False
    dirty_insert_early_penalty_weight: float = 0.0
    dirty_insert_early_start_frac: float = 0.02
    dirty_insert_early_ramp_frac: float = 0.18
    dirty_insert_early_push_start_m: float = 0.05
    dirty_insert_early_push_ramp_m: float = 0.20
    dirty_insert_early_use_progress_delta: bool = False
    dirty_insert_early_progress_delta_scale_frac: float = 0.05
    dirty_insert_early_gate_positive_rewards: bool = False
    dirty_insert_early_gate_floor: float = 0.25
    push_free_insert_progress_reward_enable: bool = False
    push_free_insert_progress_reward_weight: float = 0.0
    push_free_insert_progress_start_frac: float = 0.0
    push_free_insert_progress_end_frac: float = 0.45
    eval_dirty_insert_penalty_enable: bool = False
    eval_dirty_insert_once_penalty_weight: float = 0.0
    eval_dirty_insert_persistent_penalty_weight: float = 0.0
    eval_dirty_insert_progress_start_frac: float = 0.02
    eval_dirty_insert_progress_ramp_frac: float = 0.18
    eval_dirty_insert_disp_margin_m: float = 0.0
    eval_dirty_insert_gate_positive_rewards: bool = False
    eval_dirty_insert_gate_floor: float = 0.25
    eval_dirty_preinsert_penalty_enable: bool = False
    eval_dirty_preinsert_penalty_weight: float = 0.0
    eval_dirty_preinsert_disp_scale_m: float = 0.25
    eval_dirty_preinsert_contact_dist_m: float = 0.30
    eval_dirty_preinsert_contact_ramp_m: float = 0.45
    eval_dirty_preinsert_insert_frac_max: float = 0.45
    eval_dirty_preinsert_gate_positive_rewards: bool = False
    eval_dirty_preinsert_gate_floor: float = 0.80
    eval_dirty_max_reward_gate_enable: bool = False
    eval_dirty_max_reward_gate_floor: float = 0.35
    preinsert_push_penalty_enable: bool = False
    preinsert_push_penalty_weight: float = 4.0
    preinsert_push_penalty_start_m: float = 0.03
    preinsert_push_penalty_scale_m: float = 0.20
    preinsert_push_termination_enable: bool = False
    preinsert_push_termination_m: float = 0.25
    preinsert_push_termination_min_steps: int = 8
    preinsert_push_termination_penalty_weight: float = 80.0
    dirty_push_termination_enable: bool = False
    dirty_push_termination_m: float = 0.08
    dirty_push_termination_min_steps: int = 8
    dirty_push_termination_penalty_weight: float = 120.0
    preinsert_contact_clean_gate_enable: bool = False
    preinsert_contact_dist_m: float = 0.08
    preinsert_contact_dist_ramp_m: float = 0.18
    preinsert_contact_insert_frac_max: float = 0.25
    preinsert_contact_gate_floor: float = 0.0
    preinsert_contact_center_sigma_m: float = 0.10
    preinsert_contact_yaw_sigma_deg: float = 5.0
    preinsert_contact_tip_sigma_m: float = 0.10
    preinsert_contact_gate_r_d: bool = True
    preinsert_contact_gate_r_cd: bool = True
    preinsert_contact_gate_r_cpsi: bool = True
    preinsert_contact_drive_penalty_weight: float = 6.0
    preinsert_contact_drive_vel_scale_mps: float = 0.40
    align_before_contact_enable: bool = False
    align_before_contact_reward_weight: float = 8.0
    align_before_contact_center_sigma_m: float = 0.08
    align_before_contact_yaw_sigma_deg: float = 4.0
    align_before_contact_tip_sigma_m: float = 0.08
    align_before_contact_hold_weight: float = 4.0
    align_before_contact_disable_insert_success: bool = False
    align_before_contact_disable_insert_bonus: bool = False
    clean_insert_unlock_enable: bool = False
    clean_insert_unlock_align_gate_m: float = 0.09
    clean_insert_unlock_tip_gate_m: float = 0.12
    clean_insert_unlock_yaw_gate_deg: float = 5.0
    clean_insert_unlock_push_gate_m: float = 0.05

    # ---- Steering curriculum v2: pre-insert correction shaping ----
    # 在真正插入前，继续奖励“横向 / 偏航 / 前向距离变好”，
    # 但 v2 更强调横向/偏航纠偏，弱化“继续往前顶”的驱动。
    preinsert_align_reward_enable: bool = True
    preinsert_active_dist_max_m: float = 0.80
    preinsert_active_dist_ramp_m: float = 0.20
    preinsert_insert_frac_max: float = 0.20
    preinsert_y_err_delta_weight: float = 1.0
    preinsert_yaw_err_delta_weight: float = 1.0
    preinsert_dist_front_delta_weight: float = 0.30
    preinsert_retreat_penalty_weight: float = 0.50
    preinsert_delta_clip_y_m: float = 0.02
    preinsert_delta_clip_yaw_deg: float = 1.5
    preinsert_delta_clip_dist_m: float = 0.05
    preinsert_recovery_enable: bool = False
    preinsert_recovery_dist_trigger_m: float = 0.70
    preinsert_recovery_dist_target_min_m: float = 0.90
    preinsert_recovery_dist_target_max_m: float = 1.10
    preinsert_recovery_center_trigger_m: float = 0.40
    preinsert_recovery_yaw_trigger_deg: float = 10.0
    preinsert_recovery_retreat_reward_weight: float = 2.0
    preinsert_recovery_safe_align_weight: float = 1.0
    preinsert_recovery_forward_block_weight: float = 1.0
    preinsert_recovery_gate_positive_rewards: bool = True
    preinsert_recovery_stateful_enable: bool = False
    preinsert_recovery_release_center_m: float = 0.18
    preinsert_recovery_release_yaw_deg: float = 6.0
    preinsert_recovery_target_base_m: float = 1.05
    preinsert_recovery_target_extra_m: float = 0.40
    preinsert_recovery_target_max_m: float = 1.45
    preinsert_recovery_target_low_tol_m: float = 0.05
    preinsert_recovery_target_high_tol_m: float = 0.15
    preinsert_recovery_reentry_margin_m: float = 0.10
    preinsert_recovery_reentry_dist_cap_m: float = 1.35
    preinsert_recovery_target_band_weight: float = 1.0
    preinsert_recovery_dist_shortfall_weight: float = 0.0
    preinsert_recovery_dist_shortfall_scale_m: float = 0.25
    preinsert_recovery_dist_shortfall_cap: float = 2.0
    preinsert_recovery_dist_ready_bonus_weight: float = 0.0
    preinsert_recovery_release_bonus_weight: float = 0.5
    # Stage A recovery repair knobs. The legacy recovery distance uses fork tip
    # to pallet-front distance; "fork_center" and "root" expose more turn room
    # when the tip is already near the pallet mouth.
    preinsert_recovery_dist_ref: str = "tip"  # tip | fork_center | root
    preinsert_recovery_trigger_dist_ref: str = "tip"  # tip | fork_center | root | same
    preinsert_recovery_gate_success_rewards: bool = False
    preinsert_recovery_gate_success_done: bool = False
    preinsert_recovery_clear_on_insert_progress: bool = True
    preinsert_recovery_terminal_penalty_weight: float = 0.0
    preinsert_recovery_reverse_action_reward_weight: float = 0.0
    preinsert_recovery_forward_action_penalty_weight: float = 0.0
    preinsert_recovery_action_shortfall_gate_enable: bool = False
    preinsert_recovery_action_shortfall_floor: float = 0.0
    preinsert_recovery_narrow_near_hard_enable: bool = False
    preinsert_recovery_narrow_dist_m: float = 0.70
    preinsert_recovery_narrow_center_m: float = 0.22
    preinsert_recovery_narrow_tip_m: float = 0.18
    preinsert_recovery_narrow_yaw_deg: float = 8.0
    preinsert_recovery_narrow_initial_near_hard_only: bool = False
    preinsert_recovery_reverse_steer_align_weight: float = 0.0
    preinsert_recovery_reverse_steer_align_min_yaw_deg: float = 3.0
    preinsert_recovery_reverse_steer_align_yaw_scale_deg: float = 10.0
    preinsert_recovery_reverse_steer_align_center_scale_m: float = 0.35
    preinsert_recovery_reverse_steer_align_tip_scale_m: float = 0.25
    preinsert_recovery_reverse_steer_align_improve_floor: float = 0.0
    # Stage A near-hard action guard. When enabled, the environment can stop a
    # policy from driving deeper into the pallet mouth while the fork is still
    # visibly misaligned and the tip has too little clearance to turn.
    preinsert_action_guard_enable: bool = False
    preinsert_action_guard_stateful_enable: bool = True
    preinsert_action_guard_initial_near_hard_only: bool = False
    preinsert_action_guard_trigger_dist_m: float = 0.65
    preinsert_action_guard_release_dist_m: float = 0.85
    preinsert_action_guard_center_m: float = 0.24
    preinsert_action_guard_tip_m: float = 0.16
    preinsert_action_guard_yaw_deg: float = 7.0
    preinsert_action_guard_insert_frac_max: float = 0.25
    preinsert_action_guard_max_forward_action: float = 0.0
    preinsert_action_guard_force_reverse: bool = True
    preinsert_action_guard_reverse_action: float = 0.30
    preinsert_action_guard_steer_scale: float = 1.0
    preinsert_action_guard_min_abs_steer: float = 0.0
    preinsert_action_guard_once_per_episode: bool = False
    preinsert_action_guard_max_steps: int = 0
    preinsert_action_guard_steer_to_reduce_error: bool = False
    preinsert_action_guard_steer_action: float = 0.35
    preinsert_action_guard_center_steer_weight: float = 1.0
    preinsert_action_guard_yaw_steer_weight: float = 0.6
    # Stage A structural repair: forward progress is only rewarded when the
    # fork is already geometrically aligned enough. This is default-off so old
    # checkpoints and baselines keep their original reward surface.
    preinsert_forward_align_gate_enable: bool = False
    preinsert_forward_gate_floor: float = 0.0
    preinsert_forward_center_sigma_m: float = 0.10
    preinsert_forward_yaw_sigma_deg: float = 6.0
    preinsert_forward_tip_sigma_m: float = 0.10
    preinsert_forward_misaligned_penalty_weight: float = 0.0
    # Stage A structural repair: explicitly reward increasing insertion depth
    # only when geometry is clean enough. Default-off to preserve baselines.
    clean_insert_progress_reward_enable: bool = False
    clean_insert_progress_reward_weight: float = 0.0
    clean_insert_progress_reward_scale_frac: float = 0.05
    clean_insert_progress_reward_start_frac: float = 0.0
    clean_insert_progress_reward_end_frac: float = 0.45
    clean_insert_progress_reward_gate_floor: float = 0.0
    clean_insert_progress_reward_use_push_gate: bool = True
    clean_insert_progress_center_sigma_m: float = 0.10
    clean_insert_progress_yaw_sigma_deg: float = 6.0
    clean_insert_progress_tip_sigma_m: float = 0.10
    clean_insert_progress_push_sigma_m: float = 0.08

    # ---- O2/O3: post-insert lateral/tip/yaw dense shaping ----
    postinsert_align_enable: bool = False
    postinsert_align_weight: float = 3.0
    postinsert_center_sigma_m: float = 0.20
    postinsert_tip_sigma_m: float = 0.15
    postinsert_center_weight: float = 1.0
    postinsert_tip_weight: float = 1.0
    postinsert_yaw_sigma_deg: float = 10.0
    postinsert_yaw_weight: float = 0.5

    # ---- 防作弊与终局优化 ----
    max_insert_z_err: float = 0.4       # 最大允许的货叉与托盘高度差（防止隔空飞越作弊）
    rew_stay_still: float = 0.5         # 成功后保持静止的奖励

    # ===== 动作范围（[-1, 1] 的归一化动作会乘以下列缩放）=====
    wheel_speed_rad_s: float = 20.0
    steer_angle_rad: float = 0.6
    lift_speed_m_s: float = 0.5

    # ---------- S1.0k 奖励参数（三阶段势函数 + 严格几何） ----------
    # PPO 折扣因子（势函数 shaping 用）
    # S1.0L: 纯差分 shaping，避免 gamma<1 在慢变化势函数上形成负常数底噪。
    gamma: float = 1.0

    # ---- 实验 3.1: 参考轨迹走廊 (Trajectory-lite) ----
    # `use_reference_trajectory=false` 时：
    # - reset 不再生成参考轨迹
    # - reward 不再计算 r_cd / r_cpsi
    # - traj/* 日志置零，便于做“无参考轨迹”基准
    use_reference_trajectory: bool = True
    # 参考轨迹生成模型：
    # - root_path_first: vehicle/root cubic + final straight
    # - rs_exact: exact Reeds-Shepp over vehicle/root pose, then map to fork-center
    # - rs_forward_preferred: exact RS candidate set + forward-preferred selection
    # 默认保留 root_path_first，RS 通过 override 开启；当前 near-field 审计显示
    # 在 ±0.15m / ±6deg 课程上直接切成 shortest-RS 还不够稳定。
    traj_model: str = "root_path_first"
    traj_pre_dist_m: float = 1.05      # v3: 将 p_pre 适当前移，确保 s_start < s_pre
    traj_vehicle_curve_min_span_m: float = 0.35
    traj_vehicle_final_straight_min_m: float = 0.10
    # Ackermann proxy:
    # wheelbase ~= 1.6m, max physical steer ~= 0.6rad -> R_min ~= wheelbase / tan(0.6) ~= 2.34m
    traj_rs_min_turn_radius_m: float = 2.34
    traj_rs_sample_step_m: float = 0.05
    traj_rs_fail_fallback_to_root_path_first: bool = True
    traj_rs_forward_preferred_max_candidates: int = 8
    traj_rs_forward_preferred_max_extra_length_m: float = 1.50
    traj_rs_forward_preferred_max_reverse_frac: float = 0.35
    traj_rs_forward_preferred_max_direction_switches: int = 1
    traj_rs_forward_preferred_require_final_forward: bool = True
    traj_rs_forward_preferred_reverse_weight: float = 3.0
    traj_rs_forward_preferred_switch_weight: float = 0.8
    traj_rs_forward_preferred_terminal_reverse_penalty: float = 2.0
    # Exp8.3 第一轮主矩阵：
    # - front: 轨迹终点停在托盘前沿中心（B0′ 基线）
    # - success_center: 轨迹 terminal geometry package 平移到 success 等效 fork_center 深度（G1）
    exp83_traj_goal_mode: str = "front"
    # Exp9.0: 主引导奖励、arrival 奖励与 out_of_bounds 默认统一到 success 几何，
    # 避免“平时追一个点，最终 success 看另一个点”的目标错位。
    # - front_center: 保留 legacy target_center
    # - success_center: 统一到 success 等效 fork_center 深度
    exp83_target_center_family_mode: str = "success_center"
    # Exp8.3 runtime U0（真实 env 路径）：
    # - enable=true 时，在 reset 后立即验证 traj 起终点、d_traj 与 yaw 对齐
    # - 推荐仅在 sanity run 中开启，不常驻正式长训
    exp83_runtime_u0_enable: bool = False
    exp83_runtime_u0_fail_fast: bool = True
    exp83_runtime_u0_eps_pos_m: float = 1e-3
    exp83_runtime_u0_eps_yaw_deg: float = 15.0
    # Exp8.3 runtime U0.5/U1（真实 env 路径）：
    # - enable=true 时，在 reset 后立即验证 r_d / rg / out_of_bounds 的 target_center family 接线
    # - 重点覆盖 family_center / alternate_center / traj_goal / out_of_bounds 四类探针
    exp83_runtime_u1_enable: bool = False
    exp83_runtime_u1_fail_fast: bool = True
    exp83_runtime_u1_eps_m: float = 1e-3
    exp83_runtime_u1_probe_margin_m: float = 0.02
    traj_ctrl_start_m: float = 0.8     # Bézier 起点切线长度 (m)
    traj_ctrl_goal_m: float = 1.0      # Bézier 终点切线长度 (m)
    traj_num_samples: int = 21         # 轨迹离散点数
    sigma_traj_d: float = 0.35         # 轨迹走廊宽度 (m)
    sigma_traj_yaw_deg: float = 15.0   # 轨迹切线偏航尺度 (deg)
    k_traj_center: float = 4.0         # 走廊居中奖励强度
    k_traj_progress: float = 6.0       # 沿轨迹推进奖励强度

    # ---- 实验 3.2: 近场 commit 奖励 ----
    d_commit_open: float = 1.0         # 开启 commit 奖励的距离门槛
    sigma_commit_tip: float = 0.12     # 叉尖横向对齐要求
    sigma_commit_yaw: float = 10.0     # 偏航对齐要求
    delta_front_clip: float = 0.05     # 单步前进距离限幅
    delta_insert_clip: float = 0.03    # 单步插入深度限幅
    k_commit_front: float = 20.0       # 鼓励接近奖励强度
    k_commit_insert: float = 30.0      # 鼓励插入奖励强度

    # Stage 1: 距离带 + 粗对齐
    # S1.0L: 距离参考点改为 base（root），插入深度仍使用 tip。
    stage_distance_ref: str = "base"
    d1_min: float = 2.0      # 距离带下界 (m)
    d1_max: float = 3.0      # 距离带上界 (m)
    e_band_scale: float = 0.5  # 距离带误差归一化尺度
    # S1.0M: 收紧对齐尺度，增大 alignment 在 E1/E2 中的权重。
    y_scale1: float = 0.15   # S1.0M: 0.25→0.15，lateral 在 E1 中权重 1.0→1.67
    yaw_scale1: float = 10.0  # S1.0M: 15→10，yaw 在 E1 中权重 0.5→0.75
    # 实验 3.1: 用 phi_traj 替代 phi1，将 k_phi1 置 0
    k_phi1: float = 0.0      # Stage1 势函数强度

    # Stage 2: 微调接近（从距离带推到口前）
    d2_scale: float = 1.0    # Stage2 前向距离尺度 (m)
    y_scale2: float = 0.08   # S1.0M: 0.12→0.08，lateral 在 E2 中权重 2.08→3.13
    yaw_scale2: float = 5.0  # S1.0M: 8→5，yaw 在 E2 中权重 0.94→1.50
    k_phi2: float = 10.0     # Stage2 势函数强度
    # S1.0M: 放宽 w_align2 门控。原 y_gate2=0.25 在 y≈0.25m 时 w_align2≈0.37（丢弃 63%）。
    y_gate2: float = 0.60    # S1.0M: 0.25→0.40，信号保留翻倍
    yaw_gate2: float = 20.0  # S1.0M: 15→20

    # Stage 3: 插入深化
    # S1.0L: 推迟 Stage3 接管，避免早期压制对齐 shaping。
    ins_start: float = 0.10  # 插入接管起始阈值 (归一化)
    ins_ramp: float = 0.15   # 插入接管缓坡宽度
    # S1.0L: 放宽 Stage3 对齐门控，避免 gate 过紧导致插入信号过弱。
    y_gate3: float = 0.18    # Stage3 严对齐门控 (m)
    yaw_gate3: float = 12.0  # Stage3 严对齐门控 (deg)
    k_ins: float = 30.0      # 插入势函数强度 (从 18.0 增加到 30.0，强力引导最后插入)
    # S1.0L: 默认不再用 (1-w3) 压制 phi1/phi2，保留开关便于回滚。
    suppress_preinsert_phi_with_w3: bool = False

    # 举升
    # S1.0M: insert_gate_norm 从 0.60 降到 0.35。
    # 物理最大 insert_norm ≈ 0.477（convexDecomposition 碰撞限制），
    # 原 0.60 永远不可达 → w_lift_base 恒为 0，phi_lift 恒为 0，
    # pen_premature 惩罚一切举升 → success 永远不可能。
    insert_gate_norm: float = 0.35  # 允许举升的插入深度门槛
    insert_ramp_norm: float = 0.08  # 举升门控缓坡（0.35~0.43 打开）
    k_lift: float = 20.0     # 举升势函数强度
    k_pre: float = 5.0       # S1.0M: 10→5，降低空举惩罚避免打爆探索
    # S1.0O-A3: premature lift 惩罚分段温和化
    premature_hard_thresh: float = 0.05    # insert_norm < 此值时全额惩罚
    premature_soft_thresh: float = 0.20    # insert_norm >= 此值时惩罚 → 0
    # S1.0O-A3: lift 进度 delta 势函数
    k_lift_progress: float = 1.2           # lift delta shaping 权重
    sigma_lift: float = 0.15               # lift 误差尺度 (m)

    # 常驻惩罚
    rew_action_l2: float = -0.01
    rew_time_penalty: float = -0.003
    k_dist_cont: float = 0.02  # S1.0k: 0.03→0.02，避免压死探索

    # 成功奖励
    rew_success: float = 100.0
    rew_success_time: float = 30.0

    # 超时终局惩罚
    rew_timeout: float = -10.0

    # 里程碑奖励（一次性触发）
    rew_milestone_approach: float = 1.0
    rew_milestone_coarse_align: float = 2.0
    rew_milestone_insert_10: float = 5.0
    rew_milestone_insert_30: float = 10.0
    # S1.0M: 新增对齐里程碑，给对齐一条"强奖励通路"
    rew_milestone_fine_align: float = 5.0      # y<0.10m & yaw<5°
    rew_milestone_precise_align: float = 8.0   # y<0.05m & yaw<3°
    # S1.0N: gate_align 里程碑（entry 条件本身，绑定 approach flag 防早触发）
    rew_milestone_gate_align: float = 2.5      # y<0.15m & yaw<8° & approach 已触发

    # S1.0O-B1: 增大 sigma + k，让 lateral 0.2~0.4m 区间梯度更强（S1.0N: 0.1/0.15/8.0）
    k_hold_align: float = 1.0          # delta shaping 权重 (从 0.3 增加到 1.0，强力鼓励保持对齐)
    hold_align_sigma_y: float = 0.25   # 横向尺度 (m)
    hold_align_sigma_yaw: float = 8.0  # 偏航尺度 (deg) — A3B1C2_v2: 12→8 收紧 yaw 梯度

    # ---- S1.0Q: 死区惩罚 (A1) ----
    dead_zone_insert_thresh: float = 0.30   # 死区插入阈值（归一化）
    dead_zone_lat_thresh: float = 0.20      # 死区横向阈值 (m)
    k_dead_zone: float = 0.5               # 死区惩罚权重
    dead_zone_pen_clamp: float = 0.05      # 每步最大惩罚绝对值

    # ---- S1.0Q: 撤退鼓励 (A2/A2v2) ----
    k_retreat: float = 0.0                  # B0=0（不激活）; A2v2 改为 2.0
    retreat_reward_clamp: float = 0.1       # 每步最大撤退奖励
    retreat_window_size: int = 8            # A2v2: 撤退窗口长度（步数）

    # ---- S1.0Q: Milestone 死区衰减 ----
    # S1.0S: 锁定 A1 最佳配置为默认值（s1.0q 验证 milestone_dead_zone_scale=0.0 最优）
    milestone_dead_zone_scale: float = 0.0  # S1.0S default: A1 验证最优

    # ---- S1.0Q: 插入进度门控 (B1) ----
    # S1.0S: 锁定 B1a' 最佳配置为默认值（s1.0q 验证 ins_floor=0.2 最优）
    ins_floor: float = 0.2                  # S1.0S default: B1a' 验证最优
    ins_lat_gate_sigma: float = 1e6         # B0=无穷大（不生效）; B1b 改为 0.20

    # ---- S1.0Q: 横向精调 (C1) ----
    k_lat_fine: float = 0.0                 # Freeze current best default; keep lateral fine shaping as optional candidate
    lat_fine_sigma: float = 0.15            # 横向高斯尺度 (m)
    lat_fine_ins_thresh: float = 0.05       # 激活门槛（归一化 insert_norm）

    # ---- S1.0Q: 观测分辨率 (C2) ----
    y_err_fine_scale: float = 0.20          # 精细横向归一化尺度 (m)
    yaw_err_fine_scale_deg: float = 8.0     # 精细偏航归一化尺度 (°)

    # ---- S1.0S Phase-0.5: 初始位姿鲁棒性 ----
    # 方案 B: y_err_obs 归一化尺度（原硬编码 0.5，|y|>0.5m 时观测饱和）
    # 扩大到 0.8 消除 Y=[-0.6,+0.6] 范围内的观测饱和
    y_err_obs_scale: float = 0.8            # 默认 0.5（原行为）; Phase-0.5 P1/P3 改为 0.8

    # 失败早停
    early_stop_d_xy_max: float = 5.0
    early_stop_d_xy_steps: int = 60
    early_stop_stall_phi_eps: float = 0.001
    early_stop_stall_steps: int = 60
    early_stop_stall_action_eps: float = 0.05
    rew_early_stop_fly: float = -2.0
    rew_early_stop_stall: float = -1.0

    # ---- S1.0Q Batch-3: Dead-zone stuck detector ----
    dz_stuck_ins_eps: float = 0.005         # insert_norm 变化阈值
    dz_stuck_lat_eps: float = 0.005         # y_err 变化阈值 (m)
    dz_stuck_steps: int = 99999             # 默认不激活; Exp4 改为 30
    rew_early_stop_dz_stuck: float = -2.0   # 卡死 penalty
    # ---- S1.0Q Batch-4: stuck detector 消融控制 ----
    dz_stuck_early_done: bool = True        # False = penOnly（只给 penalty 不终止）

    # ---- S1.0S Phase-2: 举升里程碑 ----
    rew_milestone_lift_10cm: float = 3.0    # 首次 lift_height >= 0.10m
    rew_milestone_lift_20cm: float = 5.0    # 首次 lift_height >= 0.20m

    # ---- S1.0T: 高举升里程碑 ----
    rew_milestone_lift_50cm: float = 6.0    # 首次 lift_height >= 0.50m
    rew_milestone_lift_75cm: float = 8.0    # 首次 lift_height >= 0.75m

    # ---- S1.0T: 观测归一化 ----
    lift_pos_scale: float = 1.0             # obs 中 lift_pos /= scale（防止高举升 OOD）

    # ---- S1.0S Phase-R: 远场大横偏修正奖励 ----
    k_far_lat: float = 0.0                  # 默认不激活; Phase-R 改为 2.0
    far_lat_y_thresh: float = 0.4           # y_err 激活阈值 (m)
    far_lat_ins_thresh: float = 0.1         # insert_norm 激活阈值（仅远场生效）

    # ---- S1.0S Phase-3: 全局进展停滞检测器 ----
    global_stall_phi_eps: float = 0.01      # phi_total 变化阈值
    global_stall_steps: int = 120         # 默认不激活; Phase-3 改为 120（约 4 秒）
    rew_global_stall: float = -1.5          # 全局停滞 penalty

    # ---- S1.0U: Fork-tip alignment gate ----
    # Adds tip_y_err (fork tip lateral error) to w_align2/w_align3 gates,
    # distance-weighted so far-field training is unaffected.
    fork_reach_m: float = 1.87              # root to fork tip forward distance (m)
    tip_y_gate2: float = 0.50              # Stage2 fork tip lateral gate sigma (m)
    tip_y_gate3: float = 0.20              # Stage3 fork tip lateral gate sigma (m)
    tip_y_weight_dist_thresh: float = 2.5  # tip_y_err starts blending in below this dist (m)

    # ---- S1.0U: Collision proximity penalty ----
    k_tip_proximity_pen: float = 1.5       # proximity penalty weight
    tip_prox_dist_thresh: float = 2.2      # only active when dist < this (m)
    tip_prox_lat_thresh: float = 0.15      # tip_y_err dead-zone (m), no penalty below

    # ---- S1.0U: Success/hold tip alignment consistency ----
    tip_align_entry_m: float = 0.12        # tip_y_err <= this to enter hold (near-field)
    tip_align_exit_m: float = 0.16         # Schmitt exit: tip_y_err > this breaks hold
    tip_align_near_dist: float = 2.2       # tip constraint only active below this dist
    # Exp9.0 diagnostic only: fixed pre-hold reachable bands for logging.
    # These do not change hold/success logic. They measure how often policy reaches
    # "strict 0.12 hold"之外的更宽 corridor, independent of the training-time tip gate.
    prehold_reachable_strict_tip_ref_m: float = 0.12
    prehold_reachable_tip_band_m: float = 0.17
    prehold_reachable_tip_band_companion_m: float = 0.175

    # ---- GeoEdge strict-success training assists (default-off) ----
    # These knobs are intended for the 21D GeometryEdgeObs staged pipeline.
    # Strict diagnostics/eval remain pinned to strict_tip_align_entry_m even
    # when hold_gate_curriculum_enable relaxes the training-time tip gate.
    strict_tip_align_entry_m: float = 0.12
    hold_gate_curriculum_enable: bool = False
    hold_gate_curriculum_start_m: float = 0.175
    hold_gate_curriculum_end_m: float = 0.12
    hold_gate_curriculum_steps: int = 1_000_000
    lift_progress_reward_enable: bool = False
    lift_progress_reward_weight: float = 8.0
    lift_progress_reward_scale_m: float = 0.30
    premature_lift_penalty_enable: bool = False
    premature_lift_penalty_weight: float = 2.0
    premature_lift_penalty_deadband_m: float = 0.02
    hold_counter_progress_reward_enable: bool = False
    hold_counter_progress_reward_weight: float = 4.0
    post_lift_stability_penalty_enable: bool = False
    post_lift_stability_penalty_weight: float = 0.25
    post_lift_stability_min_lift_m: float = 0.05

    # ---- 实验 B: 去过度设计版论文原生 Reward ----
    # 正向奖励权重 (Positive Reward R+)
    alpha_1: float = 5.0     # r_d 权重 (距离托盘)
    alpha_2: float = 5.0     # r_cd 权重 (距离参考轨迹)
    alpha_3: float = 5.0     # r_cpsi 权重 (偏航角对齐参考轨迹)
    alpha_4: float = 50.0    # rg 权重 (到达托盘特殊奖励)
    alpha_lift: float = 0.0  # 举升奖励权重（纯 approach 设为 0）

    # 负向惩罚权重 (Penalty Reward R-)
    alpha_5: float = 0.5     # rp 权重 (推盘惩罚) - 大幅降低，允许初期试错
    alpha_6: float = 1.0     # rv 权重 (超速惩罚)
    alpha_7: float = 0.1     # ra 权重 (动作突变惩罚) - 降低，释放转向能力
    alpha_bound: float = 0.5 # r_bound 权重 (动作绝对值惩罚，模拟论文中的 L_bound)
    alpha_8: float = 5.0     # rini 权重 (初始停滞惩罚) - 降回5.0，减少过度恐慌
    alpha_9: float = 50.0    # r_out 权重 (越界逃跑惩罚)
    
    # 论文公式中的阈值
    paper_pallet_vel_thresh: float = 0.01  # 托盘移动速度阈值 (m/s)
    paper_fork_vel_thresh: float = 0.07    # 叉车超速阈值 (m/s)
    paper_ini_vel_thresh: float = 0.05     # 初始停滞速度阈值 (m/s)
    paper_ini_dist_thresh: float = 0.3     # 初始停滞距离阈值 (m)
    paper_rg_dist_thresh: float = 0.28     # 触发 rg 的距离阈值 (m) - 实验5.8：回归物理现实，从不可达的0.25m退回到0.28m
    paper_out_of_bounds_dist: float = 5.0  # 越界逃跑距离阈值 (m) - 修复：之前为3.0m，导致出生在远处的Agent直接被判定越界秒杀
    paper_eps: float = 0.01                # 防止除零的极小值
    paper_reward_max: float = 20.0         # 1/x 奖励形式的最大截断值，释放原生引力同时防爆

    # termination thresholds
    max_roll_pitch_rad: float = 0.45  # ~25 deg
    max_time_s: float = episode_length_s

    # ===== 叉车配置（forklift_c 关节命名）=====
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DEFAULT_FORKLIFT_USD_PATH,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # 强制将整个叉车涂成亮红色，解决 headless 下看不见的问题
                metallic=0.5,
                roughness=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=20.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=3000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-3.5, 0.0, 0.03),  # S1.0h: 缩放 1.8x 后调整初始距离
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "left_front_wheel_joint": 0.0,
                "right_front_wheel_joint": 0.0,
                "left_rotator_joint": 0.0,
                "right_rotator_joint": 0.0,
                "left_back_wheel_joint": 0.0,
                "right_back_wheel_joint": 0.0,
                "lift_joint": 0.0,
            },
        ),
        actuators={
            # rolling joints (velocity targets)
            "front_wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_front_wheel_joint", "right_front_wheel_joint"],
                velocity_limit=40.0,
                effort_limit=200.0,
                stiffness=0.0,
                damping=100.0,
            ),
            "back_wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_back_wheel_joint", "right_back_wheel_joint"],
                velocity_limit=40.0,
                effort_limit=200.0,
                stiffness=0.0,
                damping=50.0,
            ),
            # steering joints (position targets)
            "rotators": ImplicitActuatorCfg(
                joint_names_expr=["left_rotator_joint", "right_rotator_joint"],
                velocity_limit=10.0,
                effort_limit=300.0,
                stiffness=4000.0,
                damping=400.0,
            ),
            # lift joint (position control — logs32 验证的唯一可行 drive 方式)
            # Isaac Lab 的 ImplicitActuator 会将 stiffness/damping 写入 PhysX 关节驱动，
            # 覆盖 USD DriveAPI 原始值。
            # logs32 证明：stiffness=200000 + set_joint_position_target 可产生
            # force ≈ 200000*pos_error ≈ 3200N（足以克服重力），lift 稳定升至 0.23m。
            # 纯速度控制（stiffness=0）经 31 次实验全部失败。
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit=1.0,
                effort_limit=50000.0,  # 50kN，与 USD DriveAPI maxForce 一致
                stiffness=200000.0,    # 位置控制（logs32 验证值）
                damping=10000.0,       # 阻尼（logs32 验证值）
            ),
        },
    )

    # ===== 托盘配置（动态刚体）=====
    # S1.0h 修改说明（参考 docs/learning_guiding/isaac_sim_asset_import.md）：
    # 1. 从 kinematic 改为动态刚体，使托盘可以被叉车推动和举起
    # 2. scale=1.8 使托盘放大到与叉车货叉兼容的尺寸
    #    - 原始托盘插入孔宽度 ~228mm，货叉宽度 ~394mm
    #    - 放大 1.8x 后插入孔宽度 ~410mm，足够容纳货叉
    # 3. 添加 collision_props 设置接触参数（参考 collision_mesh_guide 第 6.1 节）
    # 注意：如果修改托盘缩放比例，需同步更新：
    # - pallet_depth_m（按深度比例缩放）
    # - init_state.pos[2]（初始高度）
    # - 以及 env.py 中基于 pallet_depth_m 的插入判定逻辑
    pallet_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.environ.get('ISAACLAB_PATH', '/home/uniubi/projects/forklift_sim/IsaacLab')}/../assets/pallet_com_shifted.usd",
            scale=(1.8, 1.8, 1.8),  # 托盘统一缩放（修改后需同步更新相关几何参数）
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # 动态刚体，可被推动/举起
                disable_gravity=False,    # 受重力影响，落在地面上
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),  # S1.0h: 空托盘约 45 kg
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,   # 碰撞检测提前量（防止微小穿透）
                rest_offset=0.005,     # 静止时最小间隙
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 初始高度需与缩放比例匹配，避免托盘悬空或穿地
            pos=(0.0, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ===== 地面 =====
    # 使用偏亮的蓝灰色地面，和深色货叉形成更明显的视觉对比，先降低视频策略的识别难度。
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg(color=(0.62, 0.70, 0.78))


@configclass
class ForkliftPalletInsertLiftGeoEdgeEnvCfg(ForkliftPalletInsertLiftEnvCfg):
    """21D 几何边缘观测变体（Phase 1A v2 复刻）。

    观测：12 维 edge_obs（2 条 pocket 短边 × 6 特征） + 9 维 proprio = 21 维。
    投影管线：world → robot body → camera (OpenCV 约定 + pitch) → pinhole → 归一化 → FoV 裁切。
    严格按照 docs/0427-now/geometry_edge_obs_full_summary.md 中 v2 参数实现：
      - HFOV 120°（focal ≈ 73.9 px）
      - camera_pos_local = (0.30, 0.0, 1.30) m（车体前向 0.3m，离地 1.30m）
      - camera_pitch = +25°（nose down，ROS 约定）

    与 15D 基线相比：
      - 单一 21D flat tensor，不开图像分支也不开 asymmetric critic
      - 启用完整 3 维动作（drive/steer/lift），关闭 stage_1_mode 与 stage1_success_without_lift
    """

    # 覆盖父类，启用完整任务（含举升）
    stage_1_mode: bool = False
    stage1_success_without_lift: bool = False
    use_camera: bool = False
    use_asymmetric_critic: bool = False

    # 完整 3 维动作：drive, steer, lift
    action_space: int = 3

    # 21D 几何观测开关（env.py 据此切换 obs 路径）
    enable_geo_edge_obs: bool = True

    # ----- 虚拟相机内参（pinhole）-----
    geo_camera_width: int = 256
    geo_camera_height: int = 256
    geo_camera_hfov_deg: float = 120.0
    # FoV 裁切容差（10% 边缘外仍判可见）
    geo_camera_fov_margin: float = 1.1

    # ----- 虚拟相机外参（相对叉车 root body 的固定安装）-----
    # body frame: X forward, Y left, Z up（Isaac 约定）
    geo_camera_pos_local_m: tuple[float, float, float] = (0.30, 0.0, 1.30)
    # ROS 约定：正 pitch = nose down
    geo_camera_pitch_deg: float = 25.0
    geo_camera_roll_deg: float = 0.0
    geo_camera_yaw_deg: float = 0.0

    # ----- 托盘短边端点（pallet local frame，顶面）-----
    # depth (X) = 1.2 × 1.8 = 2.16  → half = 1.08
    # width (Y) = 0.8 × 1.8 = 1.44  → half = 0.72
    # top  (Z) = 顶面相对中心约 +0.131 m（与 isaac_sim_asset_import 文档一致）
    geo_edge_half_depth_m: float = 1.08
    geo_edge_half_width_m: float = 0.72
    geo_edge_top_z_m: float = 0.131


@configclass
class ForkliftPalletApproachToyotaDualCameraEnvCfg(ForkliftPalletInsertLiftEnvCfg):
    """Toyota-style approach task: dual side cameras and 2D drive/steer action.

    This config follows the structure of the Toyota visual forklift paper:
      - approach policy controls only throttle/steering
      - lift is disabled during PPO and handled by a separate decision/script
      - actor observes left/right camera images plus velocity/yaw/previous action
      - critic keeps the privileged 15D state already used by this project
    """

    action_space: int = 2
    use_camera: bool = True
    use_dual_cameras: bool = True
    use_asymmetric_critic: bool = True
    stage_1_mode: bool = True
    stage1_success_without_lift: bool = True
    observation_space = {
        "image_left": [3, 224, 224],
        "image_right": [3, 224, 224],
        "proprio": 5,
    }

    camera_width: int = 224
    camera_height: int = 224
    dual_camera_width: int = 224
    dual_camera_height: int = 224
    dual_camera_hfov_deg: float = 60.0
    easy8_dim: int = 8
    toyota_proprio_dim: int = 5

    # Visible-range triangular initial distribution, matching the paper's
    # practical setup more closely than a single rectangular near-field band.
    stage1_use_triangular_visible_init: bool = True
    stage1_tri_x_near_m: float = -2.4
    stage1_tri_x_far_m: float = -4.0
    stage1_tri_y_half_width_near_m: float = 0.20
    stage1_tri_y_half_width_far_m: float = 0.70
    stage1_tri_yaw_deg_near: float = 6.0
    stage1_tri_yaw_deg_far: float = 16.0

    # Keep the first Toyota-aligned policy focused on push-free approach.
    clean_insert_reward_gate_enable: bool = True
    push_free_training_success_enable: bool = True
    push_free_training_use_max_disp: bool = True
    alpha_lift: float = 0.0

    # Lightweight sim2real noise hooks.  Start conservative; sweep later.
    toyota_action_noise_std: float = 0.02
    toyota_velocity_obs_noise_std: float = 0.01

    # RGB observations need render-level isolation when num_envs > 1.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=20.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )
    vision_room_enable: bool = True
    rerender_on_reset: bool = True


@configclass
class ForkliftPalletApproachToyotaDualCameraPushSafeEnvCfg(ForkliftPalletApproachToyotaDualCameraEnvCfg):
    """Push-safe Toyota approach task for BC warm start + PPO fine tuning."""

    # Conservative physical envelope to reduce saturated drive/steer behavior.
    wheel_speed_rad_s: float = 12.0
    steer_angle_rad: float = 0.45
    alpha_bound: float = 1.5
    alpha_7: float = 0.15

    # Make pallet pushing a first-class failure signal.
    preinsert_push_penalty_enable: bool = True
    preinsert_push_penalty_start_m: float = 0.02
    preinsert_push_penalty_scale_m: float = 0.20
    preinsert_push_penalty_weight: float = 4.0
    preinsert_push_termination_enable: bool = True
    preinsert_push_termination_m: float = 0.10
    preinsert_push_termination_min_steps: int = 8
    preinsert_push_termination_penalty_weight: float = 80.0
    dirty_push_termination_enable: bool = True
    dirty_push_termination_m: float = 0.18
    dirty_push_termination_min_steps: int = 8
    dirty_push_termination_penalty_weight: float = 120.0

    push_free_training_success_enable: bool = True
    push_free_training_use_max_disp: bool = True
    clean_insert_reward_gate_enable: bool = True
    clean_insert_use_push_gate: bool = True

    # Near-field safety guard: prevent driving deeper while visibly misaligned.
    preinsert_action_guard_enable: bool = True
    preinsert_action_guard_stateful_enable: bool = True
    preinsert_action_guard_trigger_dist_m: float = 0.65
    preinsert_action_guard_release_dist_m: float = 0.85
    preinsert_action_guard_center_m: float = 0.24
    preinsert_action_guard_tip_m: float = 0.16
    preinsert_action_guard_yaw_deg: float = 7.0
    preinsert_action_guard_insert_frac_max: float = 0.25
    preinsert_action_guard_max_forward_action: float = 0.0
    preinsert_action_guard_force_reverse: bool = False


@configclass
class ForkliftPalletApproachToyotaGeoEdgePushSafeTeacherEnvCfg(ForkliftPalletInsertLiftGeoEdgeEnvCfg):
    """Push-safe approach teacher with 21D geometric observations and 2D actions.

    This is the "方案B" teacher: train with many envs safely because no RGB
    camera is part of the policy observation.  The trained teacher is then used
    to generate clean single-env RGB/action pairs for the visual student.
    """

    # Toyota approach split: policy only controls drive / steer.
    stage_1_mode: bool = True
    stage1_success_without_lift: bool = True
    action_space: int = 2
    use_camera: bool = False
    use_dual_cameras: bool = False
    use_asymmetric_critic: bool = False
    enable_geo_edge_obs: bool = True
    geo_edge_record_cameras: bool = False

    observation_space = 21
    state_space = 0

    # Keep the same visible approach distribution as the Toyota visual task.
    stage1_use_triangular_visible_init: bool = True
    stage1_tri_x_near_m: float = -2.4
    stage1_tri_x_far_m: float = -4.0
    stage1_tri_y_half_width_near_m: float = 0.20
    stage1_tri_y_half_width_far_m: float = 0.70
    stage1_tri_yaw_deg_near: float = 6.0
    stage1_tri_yaw_deg_far: float = 16.0

    # Match PushSafe physical envelope and reward/termination behavior.
    wheel_speed_rad_s: float = 12.0
    steer_angle_rad: float = 0.45
    alpha_bound: float = 1.5
    alpha_7: float = 0.15
    alpha_lift: float = 0.0

    clean_insert_reward_gate_enable: bool = True
    clean_insert_use_push_gate: bool = True
    push_free_training_success_enable: bool = True
    push_free_training_use_max_disp: bool = True

    preinsert_push_penalty_enable: bool = True
    preinsert_push_penalty_start_m: float = 0.02
    preinsert_push_penalty_scale_m: float = 0.20
    preinsert_push_penalty_weight: float = 4.0
    preinsert_push_termination_enable: bool = True
    preinsert_push_termination_m: float = 0.10
    preinsert_push_termination_min_steps: int = 8
    preinsert_push_termination_penalty_weight: float = 80.0
    dirty_push_termination_enable: bool = True
    dirty_push_termination_m: float = 0.18
    dirty_push_termination_min_steps: int = 8
    dirty_push_termination_penalty_weight: float = 120.0

    preinsert_action_guard_enable: bool = True
    preinsert_action_guard_stateful_enable: bool = True
    preinsert_action_guard_trigger_dist_m: float = 0.65
    preinsert_action_guard_release_dist_m: float = 0.85
    preinsert_action_guard_center_m: float = 0.24
    preinsert_action_guard_tip_m: float = 0.16
    preinsert_action_guard_yaw_deg: float = 7.0
    preinsert_action_guard_insert_frac_max: float = 0.25
    preinsert_action_guard_max_forward_action: float = 0.0
    preinsert_action_guard_force_reverse: bool = False

    # Teacher should learn the clean state/action geometry first; randomization
    # is better added after it reaches the push-free target.
    toyota_action_noise_std: float = 0.0
    toyota_velocity_obs_noise_std: float = 0.0


@configclass
class ForkliftPalletApproachToyotaGeoEdgeRewardRefTeacherEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgePushSafeTeacherEnvCfg
):
    """Reward-reference Toyota approach teacher.

    This keeps the Toyota approach split and 21D geometry teacher interface,
    but softens the push-safe settings using the remote 15D baseline as reward
    design reference: shape clean insertion, regularize actions, and treat
    pushing mainly as a soft penalty with only extreme push as early failure.
    """

    # Keep the approach policy conservative but less saturated.
    wheel_speed_rad_s: float = 12.0
    steer_angle_rad: float = 0.45
    alpha_bound: float = 0.8
    alpha_7: float = 0.1
    alpha_lift: float = 0.0

    # Reward-reference push handling: soft first, terminate only outliers.
    preinsert_push_penalty_enable: bool = True
    preinsert_push_penalty_start_m: float = 0.05
    preinsert_push_penalty_scale_m: float = 0.20
    preinsert_push_penalty_weight: float = 1.0
    preinsert_push_termination_enable: bool = True
    preinsert_push_termination_m: float = 0.20
    preinsert_push_termination_min_steps: int = 8
    dirty_push_termination_enable: bool = True
    dirty_push_termination_m: float = 0.25
    dirty_push_termination_min_steps: int = 8

    # Make clean insertion progress visible before final success.
    clean_insert_reward_gate_enable: bool = True
    clean_insert_use_push_gate: bool = True
    clean_insert_progress_reward_enable: bool = True
    clean_insert_progress_reward_weight: float = 6.0
    push_free_insert_progress_reward_enable: bool = True
    push_free_insert_progress_reward_weight: float = 4.0
    push_free_training_success_enable: bool = True
    push_free_training_use_max_disp: bool = True

    # Guard blocks unsafe forward motion without scripted reverse recovery.
    preinsert_action_guard_enable: bool = True
    preinsert_action_guard_force_reverse: bool = False
    preinsert_action_guard_max_forward_action: float = 0.0


@configclass
class ForkliftPalletApproachToyotaGeoEdgePotentialTeacherEnvCfg(ForkliftPalletInsertLiftGeoEdgeEnvCfg):
    """Clean 2D approach teacher using the remote model_1999 potential reward.

    This is intentionally separate from PushSafe/RewardRefTeacher.  It keeps the
    Toyota split (drive/steer only, no lift in PPO) while using the successful
    15D baseline's stage-potential learning surface:
      phi_total = phi1 + phi2 + phi_insert

    Push handling starts as a soft penalty only.  Hard push termination, action
    guard, recovery, and RGB are left off so the task definition has one clear
    objective: learn aligned insertion first.
    """

    stage_1_mode: bool = True
    stage1_success_without_lift: bool = True
    action_space: int = 2
    use_camera: bool = False
    use_dual_cameras: bool = False
    use_asymmetric_critic: bool = False
    enable_geo_edge_obs: bool = True
    geo_edge_record_cameras: bool = False
    observation_space = 21
    state_space = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=6.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )

    # Start from the high-success baseline's reset envelope.  After this teacher
    # succeeds, add the Toyota visible-triangle distribution as a separate
    # curriculum step instead of mixing it into the first repair run.
    stage1_use_triangular_visible_init: bool = False
    stage1_init_x_min_m: float = -4.0
    stage1_init_x_max_m: float = -3.0
    stage1_init_y_min_m: float = -0.6
    stage1_init_y_max_m: float = 0.6
    stage1_init_yaw_deg_min: float = -14.32394487827058
    stage1_init_yaw_deg_max: float = 14.32394487827058

    use_potential_approach_reward: bool = True
    use_reference_trajectory: bool = False

    # Match the 15D main baseline's physical and PPO-facing reward scale first.
    wheel_speed_rad_s: float = 20.0
    steer_angle_rad: float = 0.6
    gamma: float = 1.0
    k_phi1: float = 6.0
    k_phi2: float = 10.0
    k_ins: float = 18.0
    k_lift: float = 0.0
    k_hold_align: float = 0.3
    max_yaw_err_deg: float = 5.0

    # Soft push penalty only in the first repair run.
    k_pallet_push_pen: float = 1.0
    pallet_push_insert_gate: float = 0.15
    pallet_push_insert_ramp: float = 0.15
    pallet_push_deadband_m: float = 0.05
    use_explicit_progress_reward: bool = True
    k_approach_progress: float = 5.0
    k_insert_progress: float = 35.0
    approach_progress_insert_stop_frac: float = 0.08
    approach_progress_align_floor: float = 0.20
    insert_progress_center_sigma_m: float = 0.12
    insert_progress_tip_sigma_m: float = 0.14
    insert_progress_yaw_sigma_deg: float = 7.0
    insert_progress_push_sigma_m: float = 0.08
    push_free_training_success_enable: bool = False
    push_free_training_use_max_disp: bool = False
    preinsert_push_termination_enable: bool = True
    preinsert_push_termination_m: float = 0.12
    preinsert_push_termination_min_steps: int = 8
    preinsert_push_termination_penalty_weight: float = 100.0
    dirty_push_termination_enable: bool = True
    dirty_push_termination_m: float = 0.14
    dirty_push_termination_min_steps: int = 8
    dirty_push_termination_penalty_weight: float = 140.0

    # Disable controls that previously created objective/action conflicts.
    preinsert_action_guard_enable: bool = False
    preinsert_recovery_enable: bool = False
    clean_insert_unlock_enable: bool = False
    clean_insert_reward_gate_enable: bool = False
    clean_insert_progress_reward_enable: bool = False
    push_free_insert_progress_reward_enable: bool = False
    preinsert_align_reward_enable: bool = False
    align_before_contact_enable: bool = False

    toyota_action_noise_std: float = 0.0
    toyota_velocity_obs_noise_std: float = 0.0


@configclass
class ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgePotentialTeacherEnvCfg
):
    """Fine-tune the potential teacher toward push-free insertion.

    This entry keeps the successful model_299 task geometry and action space,
    but removes the remaining loophole: terminal success and insertion progress
    no longer pay well when the pallet has moved beyond the push-free budget.
    """

    # Keep the model_299 insertion behavior intact and tighten only the
    # push-free part of the learning signal.
    k_pallet_push_pen: float = 1.2
    k_pallet_push_pen_quadratic: float = 1.0
    pallet_push_deadband_m: float = 0.03
    pallet_push_penalty_full_episode: bool = True
    insert_progress_push_sigma_m: float = 0.07
    push_free_terminal_gate_enable: bool = True
    push_free_terminal_bonus: float = 40.0
    dirty_success_reward_scale: float = 0.6
    dirty_success_terminal_penalty: float = 20.0
    push_free_training_success_enable: bool = True
    push_free_training_use_max_disp: bool = True

    # Still soft-first: no hard reset on mild push, so PPO can repair rather
    # than only learning to avoid termination.
    preinsert_push_termination_enable: bool = False
    dirty_push_termination_enable: bool = False


@configclass
class ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeScratchEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeEnvCfg
):
    """Scratch PushFree teacher using privileged geometry, not checkpoint warm-start.

    This keeps the model_1999/model_299 potential-reward learning surface and
    v2 push-free gates, but trains a fresh policy so it does not inherit the
    dirty-insert habit from model_299.
    """


@configclass
class ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeCurriculumEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgePotentialTeacherEnvCfg
):
    """From-scratch teacher with staged push-free tightening.

    The first part of the run keeps the model_1999/model_299 insertion learning
    surface intact.  After the policy has had enough steps to discover aligned
    insertion, the same run gradually turns on full-episode push penalty,
    tighter insert-progress push gating, clean terminal bonus, and dirty terminal
    penalty.  This is still a fresh policy: no checkpoint resume or warm-start.
    """

    potential_pushfree_curriculum_enable: bool = True
    potential_pushfree_curriculum_start_step: int = 20_000
    potential_pushfree_curriculum_ramp_steps: int = 20_000

    # Initial values are inherited from the high-success insertion teacher.
    # These final values are reached only after the curriculum ramp.
    potential_pushfree_curriculum_final_k_pallet_push_pen: float = 2.0
    potential_pushfree_curriculum_final_k_pallet_push_pen_quadratic: float = 2.0
    potential_pushfree_curriculum_final_pallet_push_deadband_m: float = 0.03
    potential_pushfree_curriculum_final_insert_progress_push_sigma_m: float = 0.06

    push_free_terminal_bonus: float = 70.0
    dirty_success_reward_scale: float = 0.0
    dirty_success_terminal_penalty: float = 30.0
    # Keep the first curriculum phase byte-for-byte equivalent to the proven
    # insertion teacher.  The potential-reward path turns on push-free terminal
    # gating from common_step_counter only after the configured start step.
    push_free_training_success_enable: bool = False
    push_free_training_use_max_disp: bool = False


@configclass
class ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg(ForkliftPalletInsertLiftGeoEdgeEnvCfg):
    """Fresh privileged teacher trained from scratch with progress-first rewards.

    This is intentionally not a checkpoint continuation and not a model_1999
    route clone.  The policy sees privileged forklift/pallet geometry, outputs
    only drive/steer, and first learns a high-success insertion teacher before
    any RGB, lift, hard push-free gate, or action guard is introduced.
    """

    stage_1_mode: bool = True
    stage1_success_without_lift: bool = True
    action_space: int = 2
    use_camera: bool = False
    use_dual_cameras: bool = False
    use_asymmetric_critic: bool = False
    enable_geo_edge_obs: bool = True
    geo_edge_record_cameras: bool = False
    observation_space = 21
    state_space = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=6.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )

    stage1_use_triangular_visible_init: bool = False
    stage1_init_x_min_m: float = -4.0
    stage1_init_x_max_m: float = -3.0
    stage1_init_y_min_m: float = -0.6
    stage1_init_y_max_m: float = 0.6
    stage1_init_yaw_deg_min: float = -14.32394487827058
    stage1_init_yaw_deg_max: float = 14.32394487827058

    use_progress_teacher_reward: bool = True
    use_potential_approach_reward: bool = False
    use_reference_trajectory: bool = False
    wheel_speed_rad_s: float = 20.0
    steer_angle_rad: float = 0.6
    k_lift: float = 0.0
    alpha_lift: float = 0.0
    max_yaw_err_deg: float = 8.0

    push_free_training_success_enable: bool = False
    push_free_training_use_max_disp: bool = False
    preinsert_push_termination_enable: bool = True
    preinsert_push_termination_m: float = 0.10
    preinsert_push_termination_min_steps: int = 8
    preinsert_push_termination_penalty_weight: float = 100.0
    dirty_push_termination_enable: bool = True
    dirty_push_termination_m: float = 0.14
    dirty_push_termination_min_steps: int = 8
    dirty_push_termination_penalty_weight: float = 140.0
    preinsert_action_guard_enable: bool = False
    preinsert_recovery_enable: bool = False
    clean_insert_unlock_enable: bool = False
    clean_insert_reward_gate_enable: bool = False
    clean_insert_progress_reward_enable: bool = False
    push_free_insert_progress_reward_enable: bool = False
    preinsert_align_reward_enable: bool = False
    align_before_contact_enable: bool = False
    lift_progress_reward_enable: bool = False
    premature_lift_penalty_enable: bool = False

    progress_teacher_approach_weight: float = 7.0
    progress_teacher_away_penalty_weight: float = 4.0
    progress_teacher_center_weight: float = 6.0
    progress_teacher_fork_center_weight: float = 22.0
    progress_teacher_fork_tip_weight: float = 32.0
    progress_teacher_yaw_weight: float = 0.35
    progress_teacher_insert_weight: float = 150.0
    progress_teacher_hold_weight: float = 35.0
    progress_teacher_align_potential_weight: float = 0.12
    progress_teacher_insert_potential_weight: float = 3.0
    progress_teacher_commit_progress_weight: float = 220.0
    progress_teacher_commit_forward_weight: float = 0.45
    progress_teacher_mouth_stall_penalty: float = 0.16
    progress_teacher_min_commit_forward_action: float = 0.22
    progress_teacher_action_l2: float = -0.012
    progress_teacher_time_penalty: float = -0.004
    progress_teacher_distance_penalty: float = 0.006
    progress_teacher_push_penalty: float = 5.0
    progress_teacher_push_deadband_m: float = 0.015
    progress_teacher_push_sigma_m: float = 0.065
    progress_teacher_clean_disp_m: float = 0.06
    progress_teacher_success_disp_m: float = 0.05
    progress_teacher_clean_gate_use_max_disp: bool = True
    progress_teacher_success_use_max_disp: bool = True
    progress_teacher_dirty_insert_penalty_weight: float = 16.0
    progress_teacher_dirty_insert_disp_m: float = 0.035
    progress_teacher_dirty_insert_min_norm: float = 0.18
    progress_teacher_dirty_insert_use_max_disp: bool = True
    progress_teacher_dirty_insert_termination_enable: bool = True
    progress_teacher_dirty_insert_termination_disp_m: float = 0.05
    progress_teacher_dirty_insert_termination_min_norm: float = 0.95
    progress_teacher_dirty_insert_termination_min_steps: int = 8
    progress_teacher_dirty_insert_termination_penalty_weight: float = 70.0
    progress_teacher_pushfree_curriculum_enable: bool = True
    progress_teacher_pushfree_curriculum_start_step: int = 2_500
    progress_teacher_pushfree_curriculum_ramp_steps: int = 9_000
    progress_teacher_push_sigma_start_m: float = 0.13
    progress_teacher_clean_disp_start_m: float = 0.12
    progress_teacher_success_disp_start_m: float = 0.10
    progress_teacher_dirty_insert_weight_start: float = 2.0
    progress_teacher_dirty_insert_disp_start_m: float = 0.065
    progress_teacher_push_penalty_start_weight: float = 1.8
    progress_teacher_push_deadband_start_m: float = 0.03
    progress_teacher_preinsert_termination_start_m: float = 0.18
    progress_teacher_dirty_termination_start_m: float = 0.22
    progress_teacher_near_dist_m: float = 1.05
    progress_teacher_near_ramp_m: float = 0.55
    progress_teacher_approach_stop_insert_norm: float = 0.10
    progress_teacher_center_gate_m: float = 0.21
    progress_teacher_yaw_gate_deg: float = 9.5
    progress_teacher_insert_gate_floor: float = 0.0
    progress_teacher_center_sigma_m: float = 0.30
    progress_teacher_yaw_sigma_deg: float = 12.0
    progress_teacher_tip_sigma_m: float = 0.25
    progress_teacher_commit_dist_m: float = 0.46
    progress_teacher_commit_center_sigma_m: float = 0.18
    progress_teacher_commit_tip_sigma_m: float = 0.16
    progress_teacher_commit_yaw_sigma_deg: float = 10.5
    progress_teacher_aligned_approach_progress_weight: float = 60.0
    progress_teacher_near_align_progress_weight: float = 0.0
    progress_teacher_misaligned_forward_penalty_enable: bool = True
    progress_teacher_misaligned_forward_penalty_weight: float = 1.2
    progress_teacher_misaligned_forward_near_m: float = 0.62
    progress_teacher_misaligned_forward_center_m: float = 0.14
    progress_teacher_misaligned_forward_tip_m: float = 0.14
    progress_teacher_misaligned_forward_yaw_deg: float = 7.0

    toyota_action_noise_std: float = 0.0
    toyota_velocity_obs_noise_std: float = 0.0


@configclass
class ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311LegacyNearLateralRecoveryEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg
):
    """v3.11 teacher with explicit near-distance high-lateral reset coverage."""

    stage1_near_hard_curriculum_enable: bool = True
    stage1_near_hard_curriculum_frac: float = 0.24
    stage1_near_hard_curriculum_start_step: int = 6_400
    stage1_near_hard_curriculum_ramp_steps: int = 19_200
    stage1_near_hard_x_min_m: float = -3.25
    stage1_near_hard_x_max_m: float = -3.00
    stage1_near_hard_y_abs_min_m: float = 0.40
    stage1_near_hard_y_abs_max_m: float = 0.60
    stage1_near_hard_yaw_abs_min_deg: float = 3.0
    stage1_near_hard_yaw_abs_max_deg: float = 14.32394487827058
    stage1_near_hard_lateral_frac: float = 1.0
    stage1_near_hard_positive_y_frac: float = 0.5
    stage1_near_hard_positive_yaw_frac: float = 0.5
    stage1_near_hard_opposite_yaw_frac: float = 0.75


@configclass
class ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311LegacyLateNearLateralRecoveryEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg
):
    """Preserve the v3.11 0-400 window, then repair near-lateral recovery."""

    stage1_near_hard_curriculum_enable: bool = True
    stage1_near_hard_curriculum_frac: float = 0.08
    stage1_near_hard_curriculum_start_step: int = 25_600
    stage1_near_hard_curriculum_ramp_steps: int = 12_800
    stage1_near_hard_x_min_m: float = -3.28
    stage1_near_hard_x_max_m: float = -3.00
    stage1_near_hard_y_abs_min_m: float = 0.40
    stage1_near_hard_y_abs_max_m: float = 0.60
    stage1_near_hard_yaw_abs_min_deg: float = 0.0
    stage1_near_hard_yaw_abs_max_deg: float = 14.32394487827058
    stage1_near_hard_lateral_frac: float = 1.0
    stage1_near_hard_positive_y_frac: float = 0.5
    stage1_near_hard_positive_yaw_frac: float = 0.5
    stage1_near_hard_opposite_yaw_frac: float = 0.50

    progress_teacher_near_align_progress_weight: float = 10.0
    progress_teacher_near_align_curriculum_enable: bool = True
    progress_teacher_near_align_curriculum_start_step: int = 25_600
    progress_teacher_near_align_curriculum_ramp_steps: int = 12_800
    progress_teacher_misaligned_forward_penalty_weight: float = 1.35


@configclass
class ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311LegacyLateNearLateralHoldEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg
):
    """Keep v3.11 learning past 400 before applying a gentle low-drift hold."""

    stage1_near_hard_curriculum_enable: bool = True
    stage1_near_hard_curriculum_frac: float = 0.04
    stage1_near_hard_curriculum_start_step: int = 32_000
    stage1_near_hard_curriculum_ramp_steps: int = 16_000
    stage1_near_hard_x_min_m: float = -3.28
    stage1_near_hard_x_max_m: float = -3.00
    stage1_near_hard_y_abs_min_m: float = 0.40
    stage1_near_hard_y_abs_max_m: float = 0.60
    stage1_near_hard_yaw_abs_min_deg: float = 0.0
    stage1_near_hard_yaw_abs_max_deg: float = 14.32394487827058
    stage1_near_hard_lateral_frac: float = 1.0
    stage1_near_hard_positive_y_frac: float = 0.5
    stage1_near_hard_positive_yaw_frac: float = 0.5
    stage1_near_hard_opposite_yaw_frac: float = 0.50

    progress_teacher_near_align_progress_weight: float = 4.0
    progress_teacher_near_align_curriculum_enable: bool = True
    progress_teacher_near_align_curriculum_start_step: int = 32_000
    progress_teacher_near_align_curriculum_ramp_steps: int = 16_000
    progress_teacher_misaligned_forward_penalty_weight: float = 1.25


@configclass
class ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311LegacyCleanQualityProbeEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg
):
    """Short warm-start probe for visually clean hard-lateral insertion.

    This is not a fresh teacher retrain. It starts from the accepted v3.11
    actor and tightens the push-free target from the old 5 cm eval scale toward
    a 3 cm visual-clean scale.
    """

    push_free_disp_thresh_m: float = 0.030

    stage1_near_hard_curriculum_enable: bool = True
    stage1_near_hard_curriculum_frac: float = 0.24
    stage1_near_hard_curriculum_start_step: int = 0
    stage1_near_hard_curriculum_ramp_steps: int = 2_560
    stage1_near_hard_x_min_m: float = -3.35
    stage1_near_hard_x_max_m: float = -3.00
    stage1_near_hard_y_abs_min_m: float = 0.40
    stage1_near_hard_y_abs_max_m: float = 0.60
    stage1_near_hard_yaw_abs_min_deg: float = 0.0
    stage1_near_hard_yaw_abs_max_deg: float = 14.32394487827058
    stage1_near_hard_lateral_frac: float = 1.0
    stage1_near_hard_positive_y_frac: float = 0.5
    stage1_near_hard_positive_yaw_frac: float = 0.5
    stage1_near_hard_opposite_yaw_frac: float = 0.50

    push_free_training_success_enable: bool = True
    push_free_training_use_max_disp: bool = True
    push_free_dirty_success_penalty_weight: float = 18.0

    progress_teacher_pushfree_curriculum_enable: bool = True
    progress_teacher_pushfree_curriculum_start_step: int = 0
    progress_teacher_pushfree_curriculum_ramp_steps: int = 5_120
    progress_teacher_push_sigma_start_m: float = 0.065
    progress_teacher_push_sigma_m: float = 0.035
    progress_teacher_clean_disp_start_m: float = 0.060
    progress_teacher_clean_disp_m: float = 0.035
    progress_teacher_success_disp_start_m: float = 0.050
    progress_teacher_success_disp_m: float = 0.030
    progress_teacher_push_deadband_start_m: float = 0.015
    progress_teacher_push_deadband_m: float = 0.006
    progress_teacher_push_penalty_start_weight: float = 5.0
    progress_teacher_push_penalty: float = 12.0
    progress_teacher_dirty_insert_weight_start: float = 16.0
    progress_teacher_dirty_insert_penalty_weight: float = 36.0
    progress_teacher_dirty_insert_disp_start_m: float = 0.035
    progress_teacher_dirty_insert_disp_m: float = 0.020
    progress_teacher_dirty_insert_min_norm: float = 0.12

    progress_teacher_near_align_progress_weight: float = 22.0
    progress_teacher_aligned_approach_progress_weight: float = 72.0
    progress_teacher_commit_progress_weight: float = 210.0
    progress_teacher_commit_forward_weight: float = 0.34
    progress_teacher_insert_weight: float = 150.0
    progress_teacher_hold_weight: float = 42.0
    progress_teacher_action_l2: float = -0.014

    progress_teacher_misaligned_forward_penalty_enable: bool = True
    progress_teacher_misaligned_forward_penalty_weight: float = 2.2
    progress_teacher_misaligned_forward_near_m: float = 0.84
    progress_teacher_misaligned_forward_center_m: float = 0.11
    progress_teacher_misaligned_forward_tip_m: float = 0.11
    progress_teacher_misaligned_forward_yaw_deg: float = 5.5

    preinsert_push_termination_enable: bool = True
    preinsert_push_termination_m: float = 0.080
    preinsert_push_termination_min_steps: int = 8
    preinsert_push_termination_penalty_weight: float = 120.0
    dirty_push_termination_enable: bool = True
    dirty_push_termination_m: float = 0.110
    dirty_push_termination_min_steps: int = 8
    dirty_push_termination_penalty_weight: float = 180.0
    progress_teacher_preinsert_termination_start_m: float = 0.120
    progress_teacher_dirty_termination_start_m: float = 0.160
    progress_teacher_dirty_insert_termination_enable: bool = True
    progress_teacher_dirty_insert_termination_disp_m: float = 0.055
    progress_teacher_dirty_insert_termination_min_norm: float = 0.60
    progress_teacher_dirty_insert_termination_min_steps: int = 8
    progress_teacher_dirty_insert_termination_penalty_weight: float = 110.0


@configclass
class ForkliftPalletApproachToyotaGeoEdgePushSafeTeacherCollectEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgePushSafeTeacherEnvCfg
):
    """Teacher rollout collection env: 21D teacher obs plus side RGB recording."""

    use_camera: bool = True
    use_dual_cameras: bool = True
    geo_edge_record_cameras: bool = True

    camera_width: int = 224
    camera_height: int = 224
    dual_camera_width: int = 224
    dual_camera_height: int = 224
    dual_camera_hfov_deg: float = 60.0


@configclass
class ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshEnvCfg(
    ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg
):
    """Fresh visual student matched to the accepted V311 legacy teacher MDP.

    This is not a DirectVisual/V40 continuation.  It keeps the accepted
    privileged teacher's reset, physics, and reward stack, but changes the
    policy observation to Toyota dual-camera RGB plus 5D proprio.  Privileged
    geometry is available only to the critic/reward path.
    """

    action_space: int = 2
    use_camera: bool = True
    use_dual_cameras: bool = True
    use_asymmetric_critic: bool = True
    enable_geo_edge_obs: bool = False
    geo_edge_record_cameras: bool = False
    observation_space = {
        "image_left": [3, 224, 224],
        "image_right": [3, 224, 224],
        "proprio": 5,
    }
    state_space = 15

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=20.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )

    dual_camera_width: int = 224
    dual_camera_height: int = 224
    dual_camera_hfov_deg: float = 60.0
    dual_camera_near_clip_m: float = 0.1
    dual_camera_far_clip_m: float = 8.0
    dual_camera_left_pos_local: tuple[float, float, float] = (120.0, 55.0, 150.0)
    dual_camera_right_pos_local: tuple[float, float, float] = (120.0, -55.0, 150.0)
    dual_camera_left_rpy_local_deg: tuple[float, float, float] = (0.0, 68.0, -8.0)
    dual_camera_right_rpy_local_deg: tuple[float, float, float] = (0.0, 68.0, 8.0)

    vision_room_enable: bool = True
    vision_room_ceiling_enable: bool = False
    vision_room_floor_enable: bool = True
    vision_room_color: tuple[float, float, float] = (0.92, 0.92, 0.88)
    rerender_on_reset: bool = False


@configclass
class ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshPushPenaltyW10EnvCfg(
    ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshEnvCfg
):
    """Single-factor visual reward experiment: increase only push penalty."""

    progress_teacher_push_penalty: float = 10.0


@configclass
class ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshNearAlignW10EnvCfg(
    ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshEnvCfg
):
    """Single-factor visual reward experiment: enable near-field align progress."""

    progress_teacher_near_align_progress_weight: float = 10.0


@configclass
class ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshDirtyInsertW36EnvCfg(
    ForkliftPalletApproachV311LegacyAcceptedTeacherVisualFreshEnvCfg
):
    """Single-factor visual reward experiment: increase only dirty-insert penalty."""

    progress_teacher_dirty_insert_penalty_weight: float = 36.0
