#!/usr/bin/env python3
"""Train a small signed-geometry probe on CleanView RGB metadata.

This is a diagnostic, not a teacher-student training path.  It checks whether
the clean RGB stream carries enough information for left/right lateral and yaw
signs before spending long PPO runs on reward tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


SIGNED_LATERAL_COLUMNS = (
    "signed_lateral_err_m",
    "root_lateral_signed_m",
    "center_lateral_signed_m",
    "tip_lateral_signed_m",
)
SIGNED_YAW_COLUMNS = ("yaw_err_signed_deg", "signed_yaw_err_deg")


parser = argparse.ArgumentParser(description="CleanView signed lateral/yaw linear probe")
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_samples", type=int, default=20000)
parser.add_argument("--seed", type=int, default=20260527)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--min_accuracy", type=float, default=0.80)


def _find_metadata(dataset_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "metadata.csv",
        dataset_dir / "metadata.jsonl",
        dataset_dir / "index.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    csvs = sorted(dataset_dir.rglob("*.csv"))
    return csvs[0] if csvs else None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _first_existing(columns: tuple[str, ...], row: dict[str, Any]) -> str | None:
    for name in columns:
        if name in row and str(row.get(name, "")).strip() != "":
            return name
    return None


def _resolve_image(row: dict[str, Any], dataset_dir: Path, side: str) -> Path | None:
    keys = (
        f"image_{side}",
        f"{side}_image",
        f"image_{side}_path",
        f"{side}_image_path",
        f"camera_{side}",
        f"camera_{side}_path",
    )
    for key in keys:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = dataset_dir / path
        if path.is_file():
            return path
    return None


def _blocked(output_dir: Path, reason: str, metadata_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pass": False,
        "blocked": True,
        "reason": reason,
        "metadata_path": "" if metadata_path is None else str(metadata_path),
    }
    (output_dir / "signed_geometry_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        dataset_dir: Path,
        lateral_col: str,
        yaw_col: str,
        transform,
    ) -> None:
        self.samples: list[tuple[Path, Path, int, int]] = []
        self.transform = transform
        for row in rows:
            left = _resolve_image(row, dataset_dir, "left")
            right = _resolve_image(row, dataset_dir, "right")
            if left is None or right is None:
                continue
            lateral = float(row[lateral_col])
            yaw = float(row[yaw_col])
            if abs(lateral) < 1e-4 or abs(yaw) < 1e-4:
                continue
            self.samples.append((left, right, int(lateral >= 0.0), int(yaw >= 0.0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        left_path, right_path, lateral_sign, yaw_sign = self.samples[idx]
        left = self.transform(Image.open(left_path).convert("RGB"))
        right = self.transform(Image.open(right_path).convert("RGB"))
        return torch.cat([left, right], dim=0), torch.tensor([lateral_sign, yaw_sign], dtype=torch.long)


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.head = nn.Linear(1024, 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        left = images[:, :3]
        right = images[:, 3:]
        left_feat = self.backbone(left).flatten(1)
        right_feat = self.backbone(right).flatten(1)
        return self.head(torch.cat([left_feat, right_feat], dim=-1)).view(-1, 2, 2)


def main() -> None:
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_path = _find_metadata(dataset_dir)
    if metadata_path is None:
        _blocked(output_dir, f"no metadata csv/jsonl found under {dataset_dir}")
        return

    rows = _load_rows(metadata_path)
    if not rows:
        _blocked(output_dir, "metadata is empty", metadata_path)
        return
    lateral_col = _first_existing(SIGNED_LATERAL_COLUMNS, rows[0])
    yaw_col = _first_existing(SIGNED_YAW_COLUMNS, rows[0])
    if lateral_col is None or yaw_col is None:
        _blocked(
            output_dir,
            (
                "metadata lacks signed lateral/yaw columns; add one of "
                f"{SIGNED_LATERAL_COLUMNS} and one of {SIGNED_YAW_COLUMNS}"
            ),
            metadata_path,
        )
        return

    random.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = ProbeDataset(rows, dataset_dir, lateral_col, yaw_col, transform)
    if len(dataset) < 64:
        _blocked(output_dir, f"only {len(dataset)} usable signed dual-image samples", metadata_path)
        return

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    model = LinearProbe().to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _epoch in range(int(args.epochs)):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits[:, 0], labels[:, 0]) + criterion(logits[:, 1], labels[:, 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = torch.zeros(2, device=device)
    total = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = torch.argmax(model(images), dim=-1)
            correct += (pred == labels).float().sum(dim=0)
            total += int(labels.shape[0])
    acc = (correct / max(total, 1)).detach().cpu().numpy().astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pass": bool(float(acc[0]) >= args.min_accuracy and float(acc[1]) >= args.min_accuracy),
        "blocked": False,
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
        "num_samples": len(dataset),
        "num_val_samples": val_len,
        "lateral_col": lateral_col,
        "yaw_col": yaw_col,
        "lateral_sign_accuracy": float(acc[0]),
        "yaw_sign_accuracy": float(acc[1]),
        "min_accuracy": float(args.min_accuracy),
    }
    (output_dir / "signed_geometry_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
