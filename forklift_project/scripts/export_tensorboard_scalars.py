#!/usr/bin/env python3
"""Export TensorBoard scalar events to CSV, JSON summary, and curve PNGs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_GROUPS = {
    "teacher_success_curves.png": [
        "episode/success_rate_total",
        "episode/success_rate_ema",
        "phase/frac_success",
        "phase/frac_inserted",
        "phase/frac_clean_ok",
        "phase/frac_push_free",
    ],
    "teacher_reward_loss_curves.png": [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
        "Policy/mean_noise_std",
    ],
    "teacher_progress_diagnostics.png": [
        "progress_teacher/push_penalty",
        "progress_teacher/dirty_insert_penalty",
        "progress_teacher/success_disp_xy",
        "diag/pallet_disp_xy_mean",
        "err/insert_norm_mean",
        "err/lateral_mean",
        "err/yaw_deg_mean",
    ],
}


def _load_accumulator(run_dir: Path) -> event_accumulator.EventAccumulator:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under {run_dir}")

    acc = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
        },
    )
    acc.Reload()
    return acc


def _write_csv(acc: event_accumulator.EventAccumulator, tags: list[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "wall_time", "value"])
        for tag in tags:
            for event in acc.Scalars(tag):
                writer.writerow([tag, event.step, f"{event.wall_time:.6f}", f"{event.value:.10g}"])


def _series(acc: event_accumulator.EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
    events = acc.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def _plot_group(
    acc: event_accumulator.EventAccumulator,
    tags: list[str],
    available_tags: set[str],
    output_png: Path,
    title: str,
) -> bool:
    present = [tag for tag in tags if tag in available_tags]
    if not present:
        return False

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for tag in present:
        xs, ys = _series(acc, tag)
        ax.plot(xs, ys, linewidth=1.8, label=tag)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    return True


def _tag_summary(acc: event_accumulator.EventAccumulator, tags: list[str]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        values = [event.value for event in events]
        steps = [event.step for event in events]
        best_idx = max(range(len(values)), key=values.__getitem__)
        summary[tag] = {
            "points": len(events),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "max_step": steps[best_idx],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    acc = _load_accumulator(args.run_dir)
    tags = sorted(acc.Tags().get("scalars", []))
    if not tags:
        raise SystemExit(f"No scalar tags found under {args.run_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scalar_tags.txt").write_text("\n".join(tags) + "\n")
    _write_csv(acc, tags, args.output_dir / "teacher_scalars.csv")

    summary = {
        "run_dir": str(args.run_dir),
        "scalar_tag_count": len(tags),
        "tags": tags,
        "tag_summary": _tag_summary(acc, tags),
    }
    (args.output_dir / "teacher_scalar_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    available = set(tags)
    plots: list[str] = []
    for filename, group_tags in DEFAULT_GROUPS.items():
        output_png = args.output_dir / filename
        title = filename.removesuffix(".png").replace("_", " ").title()
        if _plot_group(acc, group_tags, available, output_png, title):
            plots.append(str(output_png))

    print(json.dumps({"tags": len(tags), "plots": plots}, indent=2))


if __name__ == "__main__":
    main()
