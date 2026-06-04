#!/usr/bin/env python3
"""Gate privileged teacher checkpoints before visual training is allowed."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable


PALLET_DEPTH_M = 2.16
FORK_FORWARD_OFFSET_M = 1.5


def _load_json(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No episode rows found: {path}")
    required = {
        "init_x_m",
        "init_y_m",
        "init_yaw_deg",
        "strict_success",
        "ever_inserted",
        "dirty_insert",
        "max_pallet_disp_xy",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"{path} missing required fields: {sorted(missing)}")
    return rows


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _label_from_summary(row: dict) -> str:
    return str(row.get("label") or Path(str(row["_path"])).stem)


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(int(float(row[key])) for row in rows) / len(rows)


def _estimated_tip_front_dist(row: dict) -> float:
    root_x = float(row["init_x_m"])
    yaw_rad = math.radians(float(row["init_yaw_deg"]))
    tip_x = root_x + FORK_FORWARD_OFFSET_M * math.cos(yaw_rad)
    pallet_front_x = -0.5 * PALLET_DEPTH_M
    return max(pallet_front_x - tip_x, 0.0)


def _bucket_stats(rows: list[dict], predicate: Callable[[dict], bool]) -> dict:
    bucket_rows = [row for row in rows if predicate(row)]
    return {
        "episodes": len(bucket_rows),
        "strict_success_rate": _rate(bucket_rows, "strict_success"),
        "dirty_insert_rate": _rate(bucket_rows, "dirty_insert"),
        "mean_max_pallet_disp_xy": (
            sum(float(row["max_pallet_disp_xy"]) for row in bucket_rows) / len(bucket_rows)
            if bucket_rows
            else 0.0
        ),
    }


def _clean_success_rate(rows: list[dict], max_disp_m: float) -> float:
    if not rows:
        return 0.0
    return (
        sum(
            int(float(row["strict_success"])) == 1
            and float(row["max_pallet_disp_xy"]) <= max_disp_m
            for row in rows
        )
        / len(rows)
    )


def _add_check(checks: list[dict], name: str, value: float | int, op: str, threshold: float | int) -> None:
    if value is None:
        passed = False
    elif op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    elif op == "==":
        passed = value == threshold
    else:
        raise ValueError(f"Unsupported check operator: {op}")
    checks.append(
        {
            "name": name,
            "value": value,
            "op": op,
            "threshold": threshold,
            "passed": bool(passed),
        }
    )


def _write_markdown(path: Path, report: dict) -> None:
    metrics = report["metrics"]
    lines = [
        "# Privileged Teacher Acceptance",
        "",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Summary JSON files: {metrics['num_summaries']}",
        f"- Episode CSV files: {metrics['num_episode_csvs']}",
        "",
        "## Gates",
        "",
        "| Gate | Value | Requirement | Result |",
        "|---|---:|---:|---|",
    ]
    for check in report["checks"]:
        lines.append(
            "| {name} | {value} | {op} {threshold} | {result} |".format(
                name=check["name"],
                value=_fmt(check["value"]),
                op=check["op"],
                threshold=_fmt(check["threshold"]),
                result="PASS" if check["passed"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| Label | Episodes | Strict Success | Dirty Insert | Push-No-Insert | Mean Max Disp | Reset Curriculum |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["checkpoint_metrics"]:
        reset_curriculum = []
        if row["stage1_near_hard_curriculum_enable"]:
            reset_curriculum.append("near-hard")
        if row["teacher_reference_reset_enable"]:
            reset_curriculum.append("teacher-ref")
        lines.append(
            "| {label} | {episodes} | {success} | {dirty} | {push} | {disp} | {reset} |".format(
                label=row["label"],
                episodes=row["total_episodes"],
                success=_fmt(row["strict_success_rate"]),
                dirty=_fmt(row["dirty_insert_rate"]),
                push=_fmt(row["push_no_insert_rate"]),
                disp=_fmt(row["mean_max_pallet_disp_xy"]),
                reset=", ".join(reset_curriculum) if reset_curriculum else "off",
            )
        )

    lines.extend(
        [
            "",
            "## Buckets",
            "",
            "| Episode CSV | Bucket | Episodes | Strict Success | Dirty Insert | Mean Max Disp |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["bucket_metrics"]:
        for bucket_name in ("init_y_pos_0p3_0p6", "near_lateral_abs_y_ge_0p4"):
            bucket = row[bucket_name]
            lines.append(
                "| {label} | {bucket_name} | {episodes} | {success} | {dirty} | {disp} |".format(
                    label=row["label"],
                    bucket_name=bucket_name,
                    episodes=bucket["episodes"],
                    success=_fmt(bucket["strict_success_rate"]),
                    dirty=_fmt(bucket["dirty_insert_rate"]),
                    disp=_fmt(bucket["mean_max_pallet_disp_xy"]),
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check privileged teacher acceptance gates")
    parser.add_argument("summaries", nargs="+", type=Path, help="Eval summary JSON files")
    parser.add_argument(
        "--episodes-csv",
        type=Path,
        action="append",
        default=[],
        help="Per-episode CSV from eval_geoedge_checkpoint.py. Repeat once per checkpoint/seed.",
    )
    parser.add_argument("--min-avg-success", type=float, default=0.95)
    parser.add_argument("--min-checkpoint-success", type=float, default=0.95)
    parser.add_argument("--min-init-y-pos-success", type=float, default=0.90)
    parser.add_argument("--min-near-lateral-success", type=float, default=0.85)
    parser.add_argument("--max-dirty-insert", type=float, default=0.02)
    parser.add_argument("--max-push-no-insert", type=float, default=0.12)
    parser.add_argument("--min-episodes-per-summary", type=int, default=2048)
    parser.add_argument("--max-avg-pallet-disp", type=float, default=None)
    parser.add_argument(
        "--clean-quality-disp-m",
        type=float,
        default=None,
        help="If set, require strict-success episodes to stay below this max pallet displacement.",
    )
    parser.add_argument("--min-clean-quality-success", type=float, default=0.0)
    parser.add_argument("--min-init-y-pos-clean-quality-success", type=float, default=0.0)
    parser.add_argument("--min-near-lateral-clean-quality-success", type=float, default=0.0)
    parser.add_argument(
        "--allow-training-reset-curriculum",
        action="store_true",
        help="Allow eval summaries with near-hard or teacher-reference reset curricula still enabled.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    summaries = [_load_json(path) for path in args.summaries]
    required_summary_keys = {
        "strict_success_rate",
        "dirty_insert_rate",
        "push_no_insert_rate",
        "mean_max_pallet_disp_xy",
        "total_episodes",
    }
    for row in summaries:
        missing = required_summary_keys - set(row.keys())
        if missing:
            raise SystemExit(f"{row['_path']} missing required summary fields: {sorted(missing)}")

    checkpoint_metrics = []
    for row in summaries:
        checkpoint_metrics.append(
            {
                "label": _label_from_summary(row),
                "path": row["_path"],
                "total_episodes": int(row["total_episodes"]),
                "strict_success_rate": float(row["strict_success_rate"]),
                "dirty_insert_rate": float(row["dirty_insert_rate"]),
                "push_no_insert_rate": float(row["push_no_insert_rate"]),
                "mean_max_pallet_disp_xy": float(row["mean_max_pallet_disp_xy"]),
                "stage1_near_hard_curriculum_enable": bool(
                    row.get("stage1_near_hard_curriculum_enable", False)
                ),
                "teacher_reference_reset_enable": bool(row.get("teacher_reference_reset_enable", False)),
            }
        )

    bucket_metrics = []
    for path in args.episodes_csv:
        rows = _load_rows(path)
        bucket_metrics.append(
            {
                "label": path.stem.removesuffix("_episodes"),
                "path": str(path),
                "total_episodes": len(rows),
                "clean_quality_success_rate": (
                    _clean_success_rate(rows, args.clean_quality_disp_m)
                    if args.clean_quality_disp_m is not None
                    else None
                ),
                "init_y_pos_0p3_0p6": _bucket_stats(
                    rows, lambda row: 0.3 <= float(row["init_y_m"]) <= 0.6
                ),
                "near_lateral_abs_y_ge_0p4": _bucket_stats(
                    rows,
                    lambda row: _estimated_tip_front_dist(row) < 0.7
                    and abs(float(row["init_y_m"])) >= 0.4,
                ),
            }
        )
        if args.clean_quality_disp_m is not None:
            bucket_metrics[-1]["init_y_pos_0p3_0p6"]["clean_quality_success_rate"] = _clean_success_rate(
                [row for row in rows if 0.3 <= float(row["init_y_m"]) <= 0.6],
                args.clean_quality_disp_m,
            )
            bucket_metrics[-1]["near_lateral_abs_y_ge_0p4"][
                "clean_quality_success_rate"
            ] = _clean_success_rate(
                [
                    row
                    for row in rows
                    if _estimated_tip_front_dist(row) < 0.7
                    and abs(float(row["init_y_m"])) >= 0.4
                ],
                args.clean_quality_disp_m,
            )

    checks: list[dict] = []
    avg_success = sum(row["strict_success_rate"] for row in checkpoint_metrics) / len(checkpoint_metrics)
    min_success = min(row["strict_success_rate"] for row in checkpoint_metrics)
    max_dirty = max(row["dirty_insert_rate"] for row in checkpoint_metrics)
    max_push_no_insert = max(row["push_no_insert_rate"] for row in checkpoint_metrics)
    min_episodes = min(row["total_episodes"] for row in checkpoint_metrics)
    max_avg_disp = max(row["mean_max_pallet_disp_xy"] for row in checkpoint_metrics)

    _add_check(checks, "average strict success", avg_success, ">=", args.min_avg_success)
    _add_check(checks, "worst checkpoint strict success", min_success, ">=", args.min_checkpoint_success)
    _add_check(checks, "max dirty insert", max_dirty, "<=", args.max_dirty_insert)
    _add_check(checks, "max push-no-insert", max_push_no_insert, "<=", args.max_push_no_insert)
    _add_check(checks, "min episodes per summary", min_episodes, ">=", args.min_episodes_per_summary)
    _add_check(checks, "episode csv count", len(args.episodes_csv), ">=", len(summaries))
    if args.max_avg_pallet_disp is not None:
        _add_check(checks, "max average pallet displacement", max_avg_disp, "<=", args.max_avg_pallet_disp)

    if bucket_metrics:
        min_init_y_pos_success = min(
            row["init_y_pos_0p3_0p6"]["strict_success_rate"] for row in bucket_metrics
        )
        min_init_y_pos_count = min(row["init_y_pos_0p3_0p6"]["episodes"] for row in bucket_metrics)
        min_near_lateral_success = min(
            row["near_lateral_abs_y_ge_0p4"]["strict_success_rate"] for row in bucket_metrics
        )
        min_near_lateral_count = min(row["near_lateral_abs_y_ge_0p4"]["episodes"] for row in bucket_metrics)
        if args.clean_quality_disp_m is not None:
            min_clean_quality_success = min(
                row["clean_quality_success_rate"] for row in bucket_metrics
            )
            min_init_y_pos_clean_quality_success = min(
                row["init_y_pos_0p3_0p6"]["clean_quality_success_rate"] for row in bucket_metrics
            )
            min_near_lateral_clean_quality_success = min(
                row["near_lateral_abs_y_ge_0p4"]["clean_quality_success_rate"]
                for row in bucket_metrics
            )
        else:
            min_clean_quality_success = None
            min_init_y_pos_clean_quality_success = None
            min_near_lateral_clean_quality_success = None
    else:
        min_init_y_pos_success = 0.0
        min_init_y_pos_count = 0
        min_near_lateral_success = 0.0
        min_near_lateral_count = 0
        min_clean_quality_success = None
        min_init_y_pos_clean_quality_success = None
        min_near_lateral_clean_quality_success = None

    _add_check(checks, "init_y [0.3,0.6] strict success", min_init_y_pos_success, ">=", args.min_init_y_pos_success)
    _add_check(checks, "init_y [0.3,0.6] bucket episodes", min_init_y_pos_count, ">=", 1)
    _add_check(checks, "near<0.7 and |y|>=0.4 strict success", min_near_lateral_success, ">=", args.min_near_lateral_success)
    _add_check(checks, "near<0.7 and |y|>=0.4 bucket episodes", min_near_lateral_count, ">=", 1)
    if args.clean_quality_disp_m is not None:
        _add_check(
            checks,
            f"strict clean-quality success <= {args.clean_quality_disp_m:.3f}m",
            min_clean_quality_success,
            ">=",
            args.min_clean_quality_success,
        )
        _add_check(
            checks,
            f"init_y [0.3,0.6] clean-quality success <= {args.clean_quality_disp_m:.3f}m",
            min_init_y_pos_clean_quality_success,
            ">=",
            args.min_init_y_pos_clean_quality_success,
        )
        _add_check(
            checks,
            f"near<0.7 and |y|>=0.4 clean-quality success <= {args.clean_quality_disp_m:.3f}m",
            min_near_lateral_clean_quality_success,
            ">=",
            args.min_near_lateral_clean_quality_success,
        )

    contaminated = [
        row["label"]
        for row in checkpoint_metrics
        if row["stage1_near_hard_curriculum_enable"] or row["teacher_reference_reset_enable"]
    ]
    if not args.allow_training_reset_curriculum:
        _add_check(checks, "eval reset curricula still enabled", len(contaminated), "==", 0)

    report = {
        "passed": all(check["passed"] for check in checks),
        "thresholds": {
            "min_avg_success": args.min_avg_success,
            "min_checkpoint_success": args.min_checkpoint_success,
            "min_init_y_pos_success": args.min_init_y_pos_success,
            "min_near_lateral_success": args.min_near_lateral_success,
            "max_dirty_insert": args.max_dirty_insert,
            "max_push_no_insert": args.max_push_no_insert,
            "min_episodes_per_summary": args.min_episodes_per_summary,
            "max_avg_pallet_disp": args.max_avg_pallet_disp,
            "clean_quality_disp_m": args.clean_quality_disp_m,
            "min_clean_quality_success": args.min_clean_quality_success,
            "min_init_y_pos_clean_quality_success": args.min_init_y_pos_clean_quality_success,
            "min_near_lateral_clean_quality_success": args.min_near_lateral_clean_quality_success,
        },
        "metrics": {
            "num_summaries": len(summaries),
            "num_episode_csvs": len(args.episodes_csv),
            "average_strict_success": avg_success,
            "worst_checkpoint_strict_success": min_success,
            "max_dirty_insert": max_dirty,
            "max_push_no_insert": max_push_no_insert,
            "min_episodes_per_summary": min_episodes,
            "max_average_pallet_displacement": max_avg_disp,
            "min_init_y_pos_success": min_init_y_pos_success,
            "min_near_lateral_success": min_near_lateral_success,
            "min_clean_quality_success": min_clean_quality_success,
            "min_init_y_pos_clean_quality_success": min_init_y_pos_clean_quality_success,
            "min_near_lateral_clean_quality_success": min_near_lateral_clean_quality_success,
            "contaminated_eval_labels": contaminated,
        },
        "checks": checks,
        "checkpoint_metrics": checkpoint_metrics,
        "bucket_metrics": bucket_metrics,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
    if args.output_md:
        _write_markdown(args.output_md, report)
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2))

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
