"""Utilities for recording Toyota dual-camera API rollouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.max() > 1.0:
        image = image / 255.0
    return torch.clamp(image, 0.0, 1.0)


class ToyotaRolloutRecorder:
    """Write per-step dual-camera observations, commands and state metrics."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_images: bool = True,
        image_every: int = 1,
        flush_every: int = 25,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_images = bool(save_images)
        self.image_every = max(1, int(image_every))
        self.flush_every = max(0, int(flush_every))
        self.metadata = dict(metadata or {})
        self.rows: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            (self.output_dir / "left").mkdir(exist_ok=True)
            (self.output_dir / "right").mkdir(exist_ok=True)

    def _write_outputs(self) -> None:
        csv_path = self.output_dir / "metadata.csv"
        if self.rows:
            fieldnames = list(self.rows[0].keys())
            seen = set(fieldnames)
            for row in self.rows[1:]:
                for key in row.keys():
                    if key not in seen:
                        fieldnames.append(key)
                        seen.add(key)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
        meta = {
            "steps": len(self.rows),
            "csv": str(csv_path),
            "save_images": self.save_images,
            "image_every": self.image_every,
        }
        meta.update(self.metadata)
        (self.output_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    def record_step(
        self,
        *,
        step: int,
        env_id: int | None = None,
        episode_id: int = 0,
        cameras: dict[str, torch.Tensor],
        state: dict[str, Any],
        command: tuple[float, float, float],
        effective_action: tuple[float, float, float] | None = None,
        prev_action: tuple[float, float, float] | None = None,
        extra_fields: dict[str, Any] | None = None,
        done: bool = False,
        done_reason: str | None = None,
    ) -> None:
        drive, steer, lift = (float(v) for v in command)
        if effective_action is None:
            effective_action = (max(-1.0, min(1.0, drive)), max(-1.0, min(1.0, steer)), max(-1.0, min(1.0, lift)))
        if prev_action is None:
            prev_action = (0.0, 0.0, 0.0)

        image_left_path = ""
        image_right_path = ""
        if self.save_images and step % self.image_every == 0 and "left" in cameras and "right" in cameras:
            image_left_path = f"left/{step:06d}.png"
            image_right_path = f"right/{step:06d}.png"
            save_image(_normalize_image(cameras["left"]), self.output_dir / image_left_path)
            save_image(_normalize_image(cameras["right"]), self.output_dir / image_right_path)

        row = {
            "step": int(step),
            "env_id": "" if env_id is None else int(env_id),
            "episode_id": int(episode_id),
            "image_left": image_left_path,
            "image_right": image_right_path,
            "cmd_drive": drive,
            "cmd_steer": steer,
            "cmd_lift": lift,
            "action_drive": float(effective_action[0]),
            "action_steer": float(effective_action[1]),
            "action_lift": float(effective_action[2]),
            "prev_drive": float(prev_action[0]),
            "prev_steer": float(prev_action[1]),
            "prev_lift": float(prev_action[2]),
            "vx_mps": float(state.get("vx_mps", 0.0)),
            "vy_mps": float(state.get("vy_mps", 0.0)),
            "yaw_rate_radps": float(state.get("yaw_rate_radps", 0.0)),
            "lift_height_m": float(state.get("lift_height_m", 0.0)),
            "lift_joint_m": float(state.get("lift_joint_m", 0.0)),
            "insert_depth_m": float(state.get("insert_depth_m", 0.0)),
            "pallet_disp_xy_m": float(state.get("pallet_disp_xy_m", 0.0)),
            "dist_front_m": float(state.get("dist_front_m", 0.0)),
            "center_lateral_err_m": float(state.get("center_lateral_err_m", 0.0)),
            "tip_lateral_err_m": float(state.get("tip_lateral_err_m", 0.0)),
            "yaw_err_deg": float(state.get("yaw_err_deg", 0.0)),
            "insert_norm": float(state.get("insert_norm", 0.0)),
            "push_free": bool(state.get("push_free", False)),
            "hold_counter": float(state.get("hold_counter", 0.0)),
            "done": bool(done),
            "done_reason": str(done_reason or state.get("done_reason", "running")),
        }
        if extra_fields:
            row.update(extra_fields)
        self.rows.append(row)
        if self.flush_every > 0 and len(self.rows) % self.flush_every == 0:
            self._write_outputs()

    def close(self, metadata: dict[str, Any] | None = None) -> None:
        if metadata:
            self.metadata.update(metadata)
        self._write_outputs()
