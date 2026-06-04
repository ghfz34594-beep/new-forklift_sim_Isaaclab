"""Forklift Pallet Insert+Lift 任务（direct workflow）。

此文件是任务注册入口点，会在 Isaac Lab 训练脚本导入时被执行：
1) train.py 执行 `import isaaclab_tasks`
2) 触发 `isaaclab_tasks/direct/__init__.py`
3) 进一步导入本模块，执行 `gym.register(...)`

注册完成后，训练脚本可以通过以下 ID 创建环境：
- "Isaac-Forklift-PalletInsertLift-Direct-v0"
"""

import gymnasium as gym

from . import agents  # noqa: F401

# 将自定义 ActorCritic 注册到 rsl_rl.modules 命名空间，
# 使 OnPolicyRunner 的 eval(class_name) 能通过 "rsl_rl.modules.*" 找到它们。
from .clamped_actor_critic import ClampedActorCritic as _ClampedActorCritic
from .vision_actor_critic import VisionActorCritic as _VisionActorCritic
import rsl_rl.modules as _rsl_modules
_rsl_modules.ClampedActorCritic = _ClampedActorCritic
_rsl_modules.VisionActorCritic = _VisionActorCritic

# 注册 Gym 环境：id 是对外统一入口，entry_point 指向环境类，kwargs 指向配置入口
gym.register(
    id="Isaac-Forklift-PalletInsertLift-Direct-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletInsertLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftInsertLiftPPORunnerCfg",
    },
)

# Phase 1A v2: 21D 几何边缘观测变体（pocket 短边 2D 像素端点 + proprio）
gym.register(
    id="Isaac-Forklift-PalletInsertLift-GeometryEdgeObs-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletInsertLiftGeoEdgeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftInsertLiftGeoEdgePPORunnerCfg",
    },
)

# Clean local entry for reproducing the remote 15D main/baseline reward stack.
# This intentionally uses the copied reference env/env_cfg/runner modules rather
# than the current Toyota/PushSafe env.py path, so checkpoint comparisons are
# not polluted by later approach-only reward and action-guard changes.
gym.register(
    id="Isaac-Forklift-PalletInsertLift-15DMainBaselineCleanRepro-v0",
    entry_point=f"{__name__}.baseline15d_main_env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline15d_main_env_cfg:ForkliftPalletInsertLiftEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.baseline15d_main_rsl_rl_ppo_cfg:"
            "ForkliftInsertLift15DMainBaselinePPORunnerCfg"
        ),
    },
)

# Toyota paper aligned mainline: dual side cameras, drive/steer approach only,
# and a separate loading decision / scripted lift layer.
gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletApproachToyotaDualCameraEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaDualCameraApproachPPORunnerCfg"
        ),
    },
)

# Push-safe Toyota mainline: same dual-camera approach interface, but with
# API/BC-oriented data collection and explicit push failure handling.
gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletApproachToyotaDualCameraPushSafeEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaDualCameraPushSafeApproachPPORunnerCfg"
        ),
    },
)

# Clean-view dual-camera entry for student distillation.  Keep it separate from
# PushSafe so previous ambiguous-camera data/checkpoints cannot be mixed in.
gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaDualCameraCleanView-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletApproachToyotaDualCameraCleanViewEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaDualCameraPushSafeApproachPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaProgressStudentCleanView-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaProgressStudentCleanViewEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaDualCameraPushSafeApproachPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaProgressStudentWide100NoRoom-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaProgressStudentWide100NoRoomEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaDualCameraPushSafeApproachPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanView-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV3-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV3EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV31-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV31EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV31PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV32-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV32EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV32PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV33-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV33EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV33PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV34-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV34EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV34PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV35-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV35EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV35PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV36-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV36EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV36PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV37-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV37EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV37PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV38-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV38EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV38PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV39-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV39EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV39PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV40Direct-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachDirectVisualInsertionCleanViewV40DirectEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftDirectVisualInsertionCleanViewV40DirectPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletApproachToyotaDualCameraRoom60EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaDualCameraPushSafeApproachPPORunnerCfg"
        ),
    },
)

# 方案B teacher: no RGB in the policy observation, so it is safe to train with
# many envs.  The collect variant keeps the same 21D teacher policy obs while
# enabling side cameras for clean single-env RGB distillation data.
gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgePushSafeTeacher-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgePushSafeTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgePushSafeTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeRewardRefTeacher-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeRewardRefTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeRewardRefTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgePotentialTeacher-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgePotentialTeacherEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgePotentialTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgePotentialTeacherPushFree-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgePotentialTeacherPushFreePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgePotentialTeacherPushFreeScratch-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeScratchEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgePotentialTeacherPushFreeScratchPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgePotentialTeacherPushFreeCurriculum-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgePotentialTeacherPushFreeCurriculumEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgePotentialTeacherPushFreeCurriculumPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacher-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCurveGuidance-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCurveGuidanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherCurveGuidancePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherRecoveryFix-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherRecoveryFixEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherRecoveryFixPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311Recovery-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311RecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherV311RecoveryPPORunnerCfg"
        ),
    },
)



gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCleanStrict-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCleanStrictEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherCleanStrictPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCleanStrictNoForwardGate-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCleanStrictNoForwardGateEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherCleanStrictNoForwardGatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311PosYFinetune-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311PosYFinetuneEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherV311PosYFinetunePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311OppYawFinetune-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311OppYawFinetuneEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherV311OppYawFinetunePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311GuardedEval-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311GuardedEvalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311PosYSlowGuardEval-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311PosYSlowGuardEvalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311OneShotRescueEval-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherV311OneShotRescueEvalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgePushSafeTeacherCollect-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgePushSafeTeacherCollectEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgePushSafeTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollect-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCollectEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectWide100NoRoom-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCollectWide100NoRoomEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectCleanView-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCollectCleanViewEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectRoom60-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:ForkliftPalletApproachToyotaGeoEdgeProgressTeacherCollectRoom60EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftToyotaGeoEdgeProgressTeacherPPORunnerCfg"
        ),
    },
)
