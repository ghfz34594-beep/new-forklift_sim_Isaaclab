"""Supervised loading-decision model for the Toyota-style pipeline.

The approach policy should stop near the pallet.  This classifier answers the
paper's second-stage question: are the forks inserted cleanly enough to lift?
It deliberately stays independent from PPO so it can be trained from simulated
success/failure snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..vision_backbone import ResNet18VisionBackbone, ResNet34VisionBackbone, freeze_module


class LoadingDecisionNet(nn.Module):
    """Binary classifier over dual-camera features and optional proprio.

    Inputs are already encoded image features.  With ResNet-34 this is usually
    left 512D + right 512D + 5D proprio.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        proprio_dim: int = 5,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(feature_dim) + int(proprio_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, image_features: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        if proprio is not None:
            x = torch.cat([image_features.float(), proprio.float()], dim=-1)
        else:
            x = image_features.float()
        return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def predict_lift(
        self,
        image_features: torch.Tensor,
        proprio: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        prob = torch.sigmoid(self.forward(image_features, proprio))
        return prob >= float(threshold)


@dataclass(frozen=True)
class DecisionMetrics:
    """Geometry metrics used to auto-label simulated decision snapshots."""

    inserted: bool
    clean_geometry: bool
    push_free: bool
    single_fork_or_dirty: bool


def decision_label_from_metrics(
    *,
    inserted: torch.Tensor,
    clean_geometry: torch.Tensor,
    push_free: torch.Tensor,
    dirty_insert: torch.Tensor,
) -> torch.Tensor:
    """Return binary lift labels for simulated snapshots.

    Positive labels require inserted + clean geometry + push-free behavior.
    Dirty/single-fork-like samples are explicitly negative, even if some loose
    success flag fires.
    """

    return (inserted.bool() & clean_geometry.bool() & push_free.bool() & (~dirty_insert.bool())).long()


class DualCameraLoadingDecisionModel(nn.Module):
    """Frozen ImageNet ResNet + binary decision head for raw dual RGB images."""

    def __init__(
        self,
        backbone_type: str = "resnet34",
        imagenet_init: bool = True,
        freeze_backbone: bool = True,
        proprio_dim: int = 5,
    ) -> None:
        super().__init__()
        if backbone_type == "resnet18":
            self.backbone = ResNet18VisionBackbone(imagenet_init=imagenet_init)
            feature_dim = 512
        elif backbone_type == "resnet34":
            self.backbone = ResNet34VisionBackbone(imagenet_init=imagenet_init)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported decision backbone_type: {backbone_type}")
        if freeze_backbone:
            freeze_module(self.backbone, True)
        self.head = LoadingDecisionNet(feature_dim=feature_dim * 2, proprio_dim=proprio_dim)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return torch.clamp(image, 0.0, 1.0)

    def forward(self, image_left: torch.Tensor, image_right: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        left_feat = self.backbone(self._normalize_image(image_left))
        right_feat = self.backbone(self._normalize_image(image_right))
        return self.head(torch.cat([left_feat, right_feat], dim=-1), proprio)
