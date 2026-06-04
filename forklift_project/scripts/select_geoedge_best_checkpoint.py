#!/usr/bin/env python3
"""Select the best GeoEdge Stage-A checkpoint from eval outputs.

Ranking follows the hard95 plan:
1. overall strict success, descending
2. near-misalignment strict success, descending
3. timeout rate, ascending
4. dirty insert rate, ascending
5. push-free success, descending
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PALLET_DEPTH_M = 2.16
FORK_FORWARD_OFFSET_M = 1.5


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    data["_summary_path"] = str(path)
    return data


def _estimated_tip_front_dist(row: dict[str, str]) -> float:
    root_x = float(row["init_x_m"])
    yaw_rad = math.radians(float(row["init_yaw_deg"]))
    tip_x = root_x + FORK_FORWARD_OFFSET_M * math.cos(yaw_rad)
    pallet_front_x = -0.5 * PALLET_DEPTH_M
    return max(pallet_front_x - tip_x, 0.0)


def _rate(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(int(float(row[key])) for row in rows) / len(rows)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def _near_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if _estimated_tip_front_dist(row) < 0.7
        and (
            abs(float(row["init_y_m"])) >= 0.4
            or abs(float(row["init_yaw_deg"])) >= 10.0
        )
    ]


def _episode_csv_for(summary_path: Path, summary: dict[str, Any]) -> Path:
    label = str(summary["label"])
    return summary_path.with_name(f"{label}_episodes.csv")


def _score_row(summary_path: Path, summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    episode_csv = _episode_csv_for(summary_path, summary)
    if not episode_csv.exists():
        raise SystemExit(f"Missing episodes CSV for {summary_path}: {episode_csv}")
    with episode_csv.open(newline="") as f:
        episode_rows = list(csv.DictReader(f))

    near = _near_rows(episode_rows)
    row = {
        "label": summary.get("label", summary_path.stem.removesuffix("_summary")),
        "checkpoint": summary["checkpoint"],
        "summary_path": str(summary_path),
        "episodes_csv": str(episode_csv),
        "total_episodes": int(summary.get("total_episodes", 0)),
        "strict_success_rate": float(summary.get("strict_success_rate", 0.0)),
        "push_free_success_rate": float(summary.get("push_free_success_rate", 0.0)),
        "dirty_insert_rate": float(summary.get("dirty_insert_rate", 0.0)),
        "timeout_frac": float(summary.get("timeout_frac", 1.0)),
        "mean_max_pallet_disp_xy": float(summary.get("mean_max_pallet_disp_xy", 0.0)),
        "near_count": len(near),
        "near_strict_success_rate": _rate(near, "strict_success"),
        "near_push_free_success_rate": _rate(near, "push_free_success"),
        "near_dirty_insert_rate": _rate(near, "dirty_insert"),
        "near_timeout_rate": _rate(near, "timeout"),
        "near_mean_max_pallet_disp_xy": _mean(near, "max_pallet_disp_xy"),
    }
    row["passes_acceptance"] = (
        row["strict_success_rate"] >= args.min_strict
        and row["timeout_frac"] <= args.max_timeout
        and row["near_strict_success_rate"] >= args.min_near_strict
        and row["push_free_success_rate"] >= args.min_push_free
        and row["dirty_insert_rate"] <= args.max_dirty
    )
    return row


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["strict_success_rate"]),
        float(row["near_strict_success_rate"]),
        -float(row["timeout_frac"]),
        -float(row["dirty_insert_rate"]),
        float(row["push_free_success_rate"]),
    )


def _write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# GeoEdge Hard95 Best Checkpoint Selection",
        "",
        "| Rank | Pass | Label | Strict | Near Strict | Timeout | Dirty | Push-Free | Near N | Checkpoint |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {passed} | {label} | {strict:.4f} | {near:.4f} | {timeout:.4f} | {dirty:.4f} | {push:.4f} | {near_n} | `{ckpt}` |".format(
                rank=idx,
                passed="yes" if row["passes_acceptance"] else "no",
                label=row["label"],
                strict=row["strict_success_rate"],
                near=row["near_strict_success_rate"],
                timeout=row["timeout_frac"],
                dirty=row["dirty_insert_rate"],
                push=row["push_free_success_rate"],
                near_n=row["near_count"],
                ckpt=row["checkpoint"],
            )
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best GeoEdge hard95 checkpoint")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--min-strict", type=float, default=0.95)
    parser.add_argument("--max-timeout", type=float, default=0.06)
    parser.add_argument("--min-near-strict", type=float, default=0.80)
    parser.add_argument("--min-push-free", type=float, default=0.60)
    parser.add_argument("--max-dirty", type=float, default=0.30)
    args = parser.parse_args()

    summary_paths = sorted(args.eval_dir.glob("*_summary.json"))
    if not summary_paths:
        raise SystemExit(f"No summary JSON files found in {args.eval_dir}")

    rows = [_score_row(path, _load_json(path), args) for path in summary_paths]
    rows.sort(key=_sort_key, reverse=True)
    best = rows[0]
    result = {
        "best_checkpoint": best["checkpoint"],
        "best_label": best["label"],
        "best_passes_acceptance": bool(best["passes_acceptance"]),
        "ranking": rows,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(rows, args.output_md)

    print(best["checkpoint"])


if __name__ == "__main__":
    main()
