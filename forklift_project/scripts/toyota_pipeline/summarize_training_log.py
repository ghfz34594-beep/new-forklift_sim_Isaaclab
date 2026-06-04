"""Summarize RSL-RL training metrics from a train.log file."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "Mean reward",
    "Mean episode length",
    "Value function loss",
    "Surrogate loss",
    "Mean action noise std",
    "episode/success_rate_ema",
    "episode/success_rate_total",
    "phase/frac_inserted",
    "phase/frac_geom_success",
    "phase/frac_success",
    "phase/frac_push_free",
    "phase/frac_clean_ok",
    "phase/frac_dirty_insert",
    "err/insert_norm_mean",
    "err/center_lateral_mean",
    "err/tip_lateral_mean",
    "diag/eval_max_pallet_disp_xy_mean",
    "progress_teacher/r_insert",
    "progress_teacher/r_commit_progress",
    "progress_teacher/r_commit_forward",
    "progress_teacher/r_curve_guidance",
    "progress_teacher/close_noinsert_penalty",
    "progress_teacher/far_noinsert_penalty",
    "progress_teacher/misaligned_forward_penalty",
    "progress_teacher/steer_sign_reward",
    "progress_teacher/insert_delta",
    "progress_teacher/insert_gate",
    "progress_teacher/commit_gate",
    "progress_teacher/pushfree_curriculum",
]


ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
METRIC_RE = re.compile(
    r"^\s*([^:\n]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def _parse_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match:
            current = {
                "iteration": int(iter_match.group(1)),
                "total_iterations": int(iter_match.group(2)),
                "metrics": {},
            }
            records.append(current)
            continue
        if current is None:
            continue
        metric_match = METRIC_RE.match(line)
        if not metric_match:
            continue
        key = metric_match.group(1).strip()
        value = float(metric_match.group(2))
        current["metrics"][key] = value
    return records


def _metric_values(records: list[dict[str, Any]], metric: str) -> list[tuple[int, float]]:
    values = []
    for record in records:
        metrics = record["metrics"]
        if metric in metrics:
            values.append((int(record["iteration"]), float(metrics[metric])))
    return values


def _summarize_metric(values: list[tuple[int, float]]) -> dict[str, Any]:
    if not values:
        return {"present": False}
    finite = [(idx, value) for idx, value in values if math.isfinite(value)]
    if not finite:
        return {"present": True, "finite": False}
    first_iter, first = finite[0]
    last_iter, last = finite[-1]
    min_iter, min_value = min(finite, key=lambda item: item[1])
    max_iter, max_value = max(finite, key=lambda item: item[1])
    return {
        "present": True,
        "finite": True,
        "first": first,
        "first_iteration": first_iter,
        "last": last,
        "last_iteration": last_iter,
        "delta": last - first,
        "min": min_value,
        "min_iteration": min_iter,
        "max": max_value,
        "max_iteration": max_iter,
        "samples": [{"iteration": idx, "value": value} for idx, value in finite],
    }


def _closest_record(records: list[dict[str, Any]], target: int) -> dict[str, Any] | None:
    if not records:
        return None
    return min(records, key=lambda record: abs(int(record["iteration"]) - target))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RSL-RL train.log metrics.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", action="append", default=None, help="Metric to summarize. Repeatable.")
    parser.add_argument(
        "--checkpoint_iteration",
        action="append",
        type=int,
        default=[500, 1000],
        help="Iteration to sample in the summary. Repeatable.",
    )
    args = parser.parse_args()

    records = _parse_log(args.log)
    metrics = args.metric if args.metric else DEFAULT_METRICS
    metric_summary = {metric: _summarize_metric(_metric_values(records, metric)) for metric in metrics}
    checkpoint_records = {}
    for target in args.checkpoint_iteration:
        record = _closest_record(records, target)
        if record is not None:
            checkpoint_records[str(target)] = {
                "nearest_iteration": int(record["iteration"]),
                "metrics": {metric: record["metrics"].get(metric) for metric in metrics},
            }
    if records:
        checkpoint_records["final"] = {
            "nearest_iteration": int(records[-1]["iteration"]),
            "metrics": {metric: records[-1]["metrics"].get(metric) for metric in metrics},
        }

    summary = {
        "log": str(args.log.resolve()),
        "num_iteration_records": len(records),
        "first_iteration": int(records[0]["iteration"]) if records else None,
        "last_iteration": int(records[-1]["iteration"]) if records else None,
        "total_iterations_reported": int(records[-1]["total_iterations"]) if records else None,
        "metrics": metric_summary,
        "checkpoint_samples": checkpoint_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
