"""
Rotary double pendulum (Furuta-like) balancing environment.
"""

import gymnasium as gym

from . import agents


##
# 训练版本 - 高性能配置
##
gym.register(
    id="Isaac-Rotary-Double-Pendulum-Direct-v0",
    entry_point=f"{__name__}.rotary_double_pendulum_env:RotaryDoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotary_double_pendulum_cfg:RotaryDoublePendulumEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

##
# Play 版本 - 优化渲染效果
##
gym.register(
    id="Isaac-Rotary-Double-Pendulum-Direct-Play-v0",
    entry_point=f"{__name__}.rotary_double_pendulum_env:RotaryDoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotary_double_pendulum_cfg:RotaryDoublePendulumEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
