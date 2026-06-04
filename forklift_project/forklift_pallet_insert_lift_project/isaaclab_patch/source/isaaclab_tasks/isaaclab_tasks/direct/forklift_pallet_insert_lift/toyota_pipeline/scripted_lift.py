"""Rule-based lift / reverse / lower sequence.

The Toyota-style pipeline keeps hydraulic/lift behavior outside the PPO
approach policy.  This helper emits normalized [drive, steer, lift] commands
after the loading decision says the approach is good enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScriptedLiftSequence:
    lift_steps: int = 90
    reverse_steps: int = 75
    lower_steps: int = 90
    lift_action: float = 1.0
    reverse_drive: float = -0.35
    lower_action: float = -0.7

    @property
    def total_steps(self) -> int:
        return int(self.lift_steps + self.reverse_steps + self.lower_steps)

    def action_at(self, step: int, *, device: torch.device | str = "cpu", batch_size: int = 1) -> torch.Tensor:
        """Return normalized [drive, steer, lift] command for sequence step."""

        action = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        if step < self.lift_steps:
            action[:, 2] = float(self.lift_action)
        elif step < self.lift_steps + self.reverse_steps:
            action[:, 0] = float(self.reverse_drive)
        elif step < self.total_steps:
            action[:, 2] = float(self.lower_action)
        return torch.clamp(action, -1.0, 1.0)

    @classmethod
    def from_env_cfg(cls, cfg) -> "ScriptedLiftSequence":
        return cls(
            lift_steps=int(getattr(cfg, "scripted_lift_steps", 90)),
            reverse_steps=int(getattr(cfg, "scripted_reverse_steps", 75)),
            lower_steps=int(getattr(cfg, "scripted_lower_steps", 90)),
            lift_action=float(getattr(cfg, "scripted_lift_lift_action", 1.0)),
            reverse_drive=float(getattr(cfg, "scripted_lift_drive_reverse", -0.35)),
            lower_action=float(getattr(cfg, "scripted_lift_lower_action", -0.7)),
        )
