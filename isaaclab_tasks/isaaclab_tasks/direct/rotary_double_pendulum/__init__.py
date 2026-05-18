"""
Rotary double pendulum (Furuta-like) balancing environment.
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Rotary-Double-Pendulum-Direct-v0",
    entry_point=f"{__name__}.rotary_double_pendulum_env:RotaryDoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotary_double_pendulum_cfg:RotaryDoublePendulumEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
