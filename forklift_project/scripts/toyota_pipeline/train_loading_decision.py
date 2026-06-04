"""Train the supervised Toyota loading-decision classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from isaaclab_tasks.direct.forklift_pallet_insert_lift.toyota_pipeline import DualCameraLoadingDecisionModel


parser = argparse.ArgumentParser(description="Train loading decision classifier")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--backbone", choices=("resnet18", "resnet34"), default="resnet34")
parser.add_argument("--train_frac", type=float, default=0.8)
args = parser.parse_args()


def main() -> None:
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    dataset = TensorDataset(
        data["image_left"],
        data["image_right"],
        data["proprio"].float(),
        data["label"].float(),
    )
    train_n = int(len(dataset) * args.train_frac)
    val_n = len(dataset) - train_n
    train_set, val_set = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DualCameraLoadingDecisionModel(backbone_type=args.backbone).to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for left, right, proprio, label in train_loader:
            left = left.to(args.device)
            right = right.to(args.device)
            proprio = proprio.to(args.device)
            label = label.to(args.device)
            logits = model(left, right, proprio)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(label.numel())
            train_count += int(label.numel())

        model.eval()
        val_loss = 0.0
        val_count = 0
        tp = fp = tn = fn = 0
        with torch.inference_mode():
            for left, right, proprio, label in val_loader:
                left = left.to(args.device)
                right = right.to(args.device)
                proprio = proprio.to(args.device)
                label = label.to(args.device)
                logits = model(left, right, proprio)
                loss = F.binary_cross_entropy_with_logits(logits, label)
                pred = torch.sigmoid(logits) >= 0.5
                truth = label.bool()
                tp += int((pred & truth).sum().item())
                fp += int((pred & (~truth)).sum().item())
                tn += int(((~pred) & (~truth)).sum().item())
                fn += int(((~pred) & truth).sum().item())
                val_loss += float(loss.item()) * int(label.numel())
                val_count += int(label.numel())

        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        false_positive_rate = fp / max(fp + tn, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_loss / max(val_count, 1),
            "val_accuracy": acc,
            "false_positive_rate": false_positive_rate,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(history[-1], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
