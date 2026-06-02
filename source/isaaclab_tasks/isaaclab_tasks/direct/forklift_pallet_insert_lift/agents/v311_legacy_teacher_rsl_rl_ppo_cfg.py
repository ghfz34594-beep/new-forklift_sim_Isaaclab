"""Forklift Pallet Insert+Lift 的 PPO 训练配置。

此文件提供 RSL-RL 的 runner / policy / algorithm 超参数配置，
由 `forklift_pallet_insert_lift/__init__.py` 中的 `gym.register()` 通过
`rsl_rl_cfg_entry_point` 引用。训练脚本会读取此配置来创建 PPO 训练器。
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForkliftVisionActorCriticCfg(RslRlPpoActorCriticCfg):
    """ActorCritic config with extra knobs for backbone pretraining transfer."""

    pretrained_backbone_path: str | None = None
    freeze_backbone: bool = False
    freeze_backbone_updates: int = 0
    imagenet_backbone_init: bool = True
    backbone_type: str = "mobilenet_v3_small"
    dual_camera: bool = False
    squash_actor_mean: bool = False
    actor_action_scale: tuple[float, ...] | None = None


@configclass
class ForkliftInsertLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Reasonable PPO defaults to get a working policy overnight.

    Tune these once the task is stable:
    - num_envs
    - num_steps_per_env
    - max_iterations
    """

    # runner：训练调度与基础运行参数
    seed = 42  # 随机种子，保证可复现
    device = "cuda:0"  # 训练设备
    num_steps_per_env = 64  # 每个环境每次 rollout 的步数
    obs_groups = {
        "policy": ["image", "proprio"],
        "critic": ["critic"],
    }
    max_iterations = 2000  # 最大训练迭代次数（iteration）
    save_interval = 50  # 保存模型与日志的间隔（iteration）
    experiment_name = "forklift_pallet_insert_lift"  # 训练实验名称（用于日志目录）

    # policy network：中等强度视觉 actor + 低维 critic
    policy = ForkliftVisionActorCriticCfg(
        class_name="rsl_rl.modules.VisionActorCritic",
        init_noise_std=0.4,
        noise_std_type="log",
        actor_obs_normalization=False,  # 图像已归一化，只保留 proprio encoder
        critic_obs_normalization=True,  # Critic 观测归一化
        actor_hidden_dims=[256, 256, 128],  # fusion actor head
        critic_hidden_dims=[256, 256, 128],  # Critic MLP 隐藏层
        activation="elu",  # 激活函数
        pretrained_backbone_path=None,  
        freeze_backbone=True,  # 强制冻结，遵循 RRL 范式
        freeze_backbone_updates=0,
        imagenet_backbone_init=True,
        backbone_type="resnet34",
    )

    # PPO algorithm：优化与损失相关超参数
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,  # 每次迭代的学习 epoch 数
        num_mini_batches=4,  # 每个 epoch 的小批次数
        learning_rate=3e-4,  # 学习率
        schedule="adaptive",  # 学习率调度策略
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE 参数
        entropy_coef=0.0005,  # StageB: 0.0015→0.0005，降低探索压力，精修模式
        desired_kl=0.008,  # StageB: 0.01→0.008，更保守更新步幅
        max_grad_norm=1.0,  # 梯度裁剪阈值
        value_loss_coef=1.0,  # 价值函数损失权重
        use_clipped_value_loss=True,  # 是否裁剪 value loss
        clip_param=0.2,  # PPO clip 系数
    )

    # optional: clip actions inside wrapper
    clip_actions = 1.0  # 动作裁剪范围（[-1, 1]）


@configclass
class ForkliftInsertLiftGeoEdgePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1A v2: 21D 几何边缘观测 PPO 配置（对称 ClampedActorCritic）。

    与基础 PPO 配置相比的差异：
      - actor / critic 都看同一份 21D flat tensor（对称 AC）
      - 不开图像分支（policy class 为 ClampedActorCritic 而非 VisionActorCritic）
      - actor_obs_normalization=True（与基线 15D 一致，21D 也走 running mean/std）
      - experiment_name 独立，便于日志分离
    """

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 64
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    max_iterations = 2000
    save_interval = 50
    experiment_name = "forklift_pallet_insert_lift_geo_edge"

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0005,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    clip_actions = 1.0


@configclass
class ForkliftToyotaDualCameraApproachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Toyota-style dual-camera approach PPO.

    Actor observes left/right ImageNet ResNet features plus 5D proprio and
    outputs only drive/steer.  Critic uses the existing privileged 15D state.
    """

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 64
    obs_groups = {
        "policy": ["image_left", "image_right", "proprio"],
        "critic": ["critic"],
    }
    max_iterations = 2000
    save_interval = 50
    experiment_name = "forklift_toyota_dual_camera_approach"

    policy = ForkliftVisionActorCriticCfg(
        class_name="rsl_rl.modules.VisionActorCritic",
        init_noise_std=0.35,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        pretrained_backbone_path=None,
        freeze_backbone=True,
        freeze_backbone_updates=0,
        imagenet_backbone_init=True,
        backbone_type="resnet34",
        dual_camera=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0005,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    clip_actions = 1.0


@configclass
class ForkliftToyotaDualCameraPushSafeApproachPPORunnerCfg(ForkliftToyotaDualCameraApproachPPORunnerCfg):
    """Push-safe Toyota dual-camera approach PPO.

    This keeps the same paper-aligned observation/action structure as the
    baseline Toyota task, but starts with lower exploration and a separate log
    namespace for BC-warm-start and push-safe PPO experiments.
    """

    experiment_name = "forklift_toyota_dual_camera_approach_pushsafe"

    policy = ForkliftVisionActorCriticCfg(
        class_name="rsl_rl.modules.VisionActorCritic",
        init_noise_std=0.18,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        pretrained_backbone_path=None,
        freeze_backbone=True,
        freeze_backbone_updates=0,
        imagenet_backbone_init=True,
        backbone_type="resnet34",
        dual_camera=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0001,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )


@configclass
class ForkliftToyotaGeoEdgePushSafeTeacherPPORunnerCfg(ForkliftInsertLiftGeoEdgePPORunnerCfg):
    """方案B teacher PPO: many-env geometric PushSafe approach policy."""

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "forklift_toyota_geoedge_pushsafe_teacher"

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.18,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0001,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    clip_actions = 1.0


@configclass
class ForkliftToyotaGeoEdgeRewardRefTeacherPPORunnerCfg(ForkliftToyotaGeoEdgePushSafeTeacherPPORunnerCfg):
    """Reward-reference teacher PPO for Toyota approach training."""

    experiment_name = "forklift_toyota_geoedge_rewardref_teacher"

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.25,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0003,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )


@configclass
class ForkliftV311LegacyAcceptedTeacherVisualFreshPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Fresh visual PPO on the accepted V311 legacy teacher MDP."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 16
    obs_groups = {
        "policy": ["image_left", "image_right", "proprio"],
        "critic": ["critic"],
    }
    max_iterations = 500
    save_interval = 50
    experiment_name = "v311_legacy_accepted_teacher_visual_fresh"
    run_name = "v311_legacy_visual_fresh"

    policy = ForkliftVisionActorCriticCfg(
        class_name="rsl_rl.modules.VisionActorCritic",
        init_noise_std=0.16,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        pretrained_backbone_path=None,
        freeze_backbone=False,
        freeze_backbone_updates=0,
        imagenet_backbone_init=True,
        backbone_type="resnet34",
        dual_camera=True,
        squash_actor_mean=True,
        actor_action_scale=(0.65, 0.55),
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0001,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    clip_actions = 1.0


@configclass
class ForkliftV311LegacyAcceptedTeacherVisualFreshPushPenaltyW10PPORunnerCfg(
    ForkliftV311LegacyAcceptedTeacherVisualFreshPPORunnerCfg
):
    """Single-factor reward run: same visual PPO, stronger push penalty."""

    experiment_name = "v311_legacy_visual_reward_single_factor"
    run_name = "push_penalty_w10"


@configclass
class ForkliftV311LegacyAcceptedTeacherVisualFreshNearAlignW10PPORunnerCfg(
    ForkliftV311LegacyAcceptedTeacherVisualFreshPPORunnerCfg
):
    """Single-factor reward run: same visual PPO, near-field align progress."""

    experiment_name = "v311_legacy_visual_reward_single_factor"
    run_name = "near_align_w10"


@configclass
class ForkliftV311LegacyAcceptedTeacherVisualFreshDirtyInsertW36PPORunnerCfg(
    ForkliftV311LegacyAcceptedTeacherVisualFreshPPORunnerCfg
):
    """Single-factor reward run: same visual PPO, stronger dirty-insert penalty."""

    experiment_name = "v311_legacy_visual_reward_single_factor"
    run_name = "dirty_insert_w36"


@configclass
class ForkliftV311LegacyAcceptedTeacherVisualFreshActionSmoothnessW8PPORunnerCfg(
    ForkliftV311LegacyAcceptedTeacherVisualFreshPPORunnerCfg
):
    """Single-factor reward run: same visual PPO, stronger action L2 pressure."""

    experiment_name = "v311_legacy_visual_reward_single_factor"
    run_name = "action_smoothness_w8"


@configclass
class ForkliftV311LegacyAcceptedTeacherVisualFreshSpeedPenaltyW5PPORunnerCfg(
    ForkliftV311LegacyAcceptedTeacherVisualFreshPPORunnerCfg
):
    """Single-factor reward run: same visual PPO, root-speed excess penalty."""

    experiment_name = "v311_legacy_visual_reward_single_factor"
    run_name = "speed_penalty_w5"


@configclass
class ForkliftToyotaGeoEdgePotentialTeacherPPORunnerCfg(ForkliftInsertLiftGeoEdgePPORunnerCfg):
    """Clean 2D approach teacher using model_1999-style potential shaping."""

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "forklift_toyota_geoedge_potential_teacher"

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0005,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    clip_actions = 1.0


@configclass
class ForkliftToyotaGeoEdgePotentialTeacherPushFreePPORunnerCfg(
    ForkliftToyotaGeoEdgePotentialTeacherPPORunnerCfg
):
    """Resume model_299 and fine-tune only the push-free reward gate."""

    max_iterations = 500
    save_interval = 25
    run_name = "pushfree_tighten_v2_from_model299"
    experiment_name = "forklift_toyota_geoedge_potential_teacher"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.12,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=2e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.002,
        max_grad_norm=0.5,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
    )


@configclass
class ForkliftToyotaGeoEdgePotentialTeacherPushFreeScratchPPORunnerCfg(
    ForkliftToyotaGeoEdgePotentialTeacherPPORunnerCfg
):
    """From-scratch PushFree teacher; do not resume or warm-start."""

    max_iterations = 2000
    save_interval = 50
    run_name = "pushfree_scratch_v1"
    experiment_name = "forklift_toyota_geoedge_potential_pushfree_scratch"
    resume = False
    load_run = ""
    load_checkpoint = ""

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.25,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )


@configclass
class ForkliftToyotaGeoEdgePotentialTeacherPushFreeCurriculumPPORunnerCfg(
    ForkliftToyotaGeoEdgePotentialTeacherPPORunnerCfg
):
    """Fresh insertion-first teacher with in-run push-free tightening."""

    max_iterations = 2000
    save_interval = 50
    run_name = "pushfree_curriculum_v2_clean_teacher_first"
    experiment_name = "forklift_toyota_geoedge_potential_pushfree_curriculum"
    resume = False
    load_run = ""
    load_checkpoint = ""

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg(ForkliftInsertLiftGeoEdgePPORunnerCfg):
    """Fresh privileged progress teacher; always starts from scratch."""

    seed = 42
    num_steps_per_env = 64
    max_iterations = 1200
    save_interval = 50
    run_name = "progress_teacher_scratch_curriculum_v311_late_dirty_event"
    experiment_name = "forklift_toyota_geoedge_progress_teacher"
    resume = False
    load_run = ""
    load_checkpoint = ""

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.65,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.001,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    clip_actions = 1.0


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactFreeze2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg
):
    """Run the reproduced v3.11 teacher path to 2000iter without post-window drift."""

    max_iterations = 2000
    save_interval = 50
    run_name = "progress_teacher_v311_legacy_exact_freeze_v8_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_exact_freeze_v8"
    resume = False
    load_run = ""
    load_checkpoint = ""

    late_phase_schedule = {
        "enable": True,
        "start_iter": 400,
        "ramp_iters": 1,
        "learning_rate": 0.0,
        "entropy_coef": 0.0,
        "desired_kl": 1.0e-6,
        "clip_param": 0.001,
        "max_grad_norm": 0.0,
        "num_learning_epochs": 1,
        "num_mini_batches": 16,
        "action_std": 0.05,
    }


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDrift2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg
):
    """Exact v3.11 early window with small non-zero post-400 PPO updates."""

    max_iterations = 2000
    save_interval = 50
    run_name = "progress_teacher_v311_legacy_exact_lowdrift_v9_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_exact_lowdrift_v9"
    resume = False
    load_run = ""
    load_checkpoint = ""

    late_phase_schedule = {
        "enable": True,
        "start_iter": 400,
        "ramp_iters": 1,
        "learning_rate": 1.0e-6,
        "entropy_coef": 0.0,
        "desired_kl": 1.0e-4,
        "clip_param": 0.01,
        "max_grad_norm": 0.05,
        "num_learning_epochs": 1,
        "num_mini_batches": 16,
        "action_std": 0.05,
        "schedule": "fixed",
        "freeze_normalization": True,
    }


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyNearLateralRecovery2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDrift2000PPORunnerCfg
):
    """Low-drift 2000iter teacher with near-lateral reset coverage from scratch."""

    run_name = "progress_teacher_v311_legacy_nearlat_recovery_v10_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_nearlat_recovery_v10"


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyLateNearLateralRecovery2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDrift2000PPORunnerCfg
):
    """v3.11 early window followed by low-drift near-lateral recovery refinement."""

    run_name = "progress_teacher_v311_legacy_late_nearlat_recovery_v11_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_late_nearlat_recovery_v11"


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyLateNearLateralHold2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDrift2000PPORunnerCfg
):
    """Let the v3.11 policy mature before gently holding the late checkpoint."""

    run_name = "progress_teacher_v311_legacy_late_nearlat_hold_v12_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_late_nearlat_hold_v12"

    late_phase_schedule = {
        "enable": True,
        "start_iter": 550,
        "ramp_iters": 1,
        "learning_rate": 5.0e-7,
        "entropy_coef": 0.0,
        "desired_kl": 5.0e-5,
        "clip_param": 0.005,
        "max_grad_norm": 0.02,
        "num_learning_epochs": 1,
        "num_mini_batches": 16,
        "action_std": 0.12,
        "schedule": "fixed",
        "freeze_normalization": True,
    }


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactFreezeActor450To2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDrift2000PPORunnerCfg
):
    """Resume from the verified model_450 window and keep the actor fixed to 2000."""

    run_name = "progress_teacher_v311_legacy_exact_freeze_actor450_v13_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_exact_freeze_actor450_v13"

    late_phase_schedule = {
        "enable": True,
        "start_iter": 450,
        "ramp_iters": 1,
        "learning_rate": 0.0,
        "entropy_coef": 0.0,
        "desired_kl": 0.0,
        "clip_param": 0.0,
        "max_grad_norm": 0.0,
        "num_learning_epochs": 1,
        "num_mini_batches": 16,
        "schedule": "fixed",
        "freeze_normalization": True,
        "freeze_actor": True,
    }


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyCleanQualityProbePPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg
):
    """Time-boxed clean-quality probe warm-started from the accepted teacher."""

    max_iterations = 300
    save_interval = 25
    run_name = "progress_teacher_v311_legacy_clean_quality_probe_seed42_300iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_clean_quality_probe"
    resume = False
    load_run = ""
    load_checkpoint = ""

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ClampedActorCritic",
        init_noise_std=0.08,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=2,
        num_mini_batches=8,
        learning_rate=8.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.0015,
        max_grad_norm=0.35,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.035,
    )


@configclass
class ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDriftFreezeActor450Scratch2000PPORunnerCfg(
    ForkliftToyotaGeoEdgeProgressTeacherV311LegacyExactLowDrift2000PPORunnerCfg
):
    """Train from scratch, preserve the successful v3.11 low-drift window after 450."""

    run_name = "progress_teacher_v311_legacy_exact_lowdrift_freeze_actor450_v14_seed42_1024env_2000iter"
    experiment_name = "forklift_toyota_geoedge_progress_teacher_v311_legacy_exact_lowdrift_freeze_actor450_v14"

    late_phase_schedule = {
        "enable": True,
        "start_iter": 400,
        "ramp_iters": 1,
        "learning_rate": 1.0e-6,
        "entropy_coef": 0.0,
        "desired_kl": 1.0e-4,
        "clip_param": 0.01,
        "max_grad_norm": 0.05,
        "num_learning_epochs": 1,
        "num_mini_batches": 16,
        "action_std": 0.05,
        "schedule": "fixed",
        "freeze_normalization": True,
        "freeze_actor": True,
        "freeze_actor_start_iter": 450,
    }
