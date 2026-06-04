#!/usr/bin/env python3
"""Summarize GeoEdge strict eval JSON files against acceptance defaults."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PALLET_DEPTH_M = 2.16
FORK_FORWARD_OFFSET_M = 1.5


def _load(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _is_stage1_insert_eval(row: dict) -> bool:
    return bool(row.get("stage1_eval") or row.get("stage_1_mode") or row.get("stage1_success_without_lift"))


def _lowest_funnel_stage(row: dict) -> str:
    stages = [
        ("insert", float(row.get("ever_inserted_rate", 0.0))),
        ("strict_geometry", float(row.get("ever_strict_geom_rate", 0.0))),
        ("hold_counter", float(row.get("strict_success_rate", 0.0))),
        ("push_free", float(row.get("push_free_success_rate", 0.0))),
    ]
    if not _is_stage1_insert_eval(row):
        stages.insert(2, ("lift", float(row.get("ever_lifted_rate", 0.0))))
    return min(stages, key=lambda item: item[1])[0]


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(int(float(row[key])) for row in rows) / len(rows)


def _bin_label(value: float, bins: list[tuple[float, float]]) -> str:
    for idx, (lo, hi) in enumerate(bins):
        if lo <= value <= hi if idx == len(bins) - 1 else lo <= value < hi:
            return f"[{lo:g},{hi:g}{']' if idx == len(bins) - 1 else ')'}"
    return "out_of_range"


def _estimated_tip_front_dist(row: dict) -> float:
    """Estimate initial fork-tip distance to the pallet front edge.

    Eval CSV stores initial root x/y/yaw, not the runtime fork tip. This estimate
    matches the quick failure triage and is only used for coarse bucket analysis.
    """
    import math

    root_x = float(row["init_x_m"])
    yaw_rad = math.radians(float(row["init_yaw_deg"]))
    tip_x = root_x + FORK_FORWARD_OFFSET_M * math.cos(yaw_rad)
    pallet_front_x = -0.5 * PALLET_DEPTH_M
    return max(pallet_front_x - tip_x, 0.0)


def _summarize_near_misalignment(rows: list[dict]) -> list[str]:
    cases = [
        (
            "near<0.7 and (|y|>=0.4 or |yaw|>=10)",
            lambda row: _estimated_tip_front_dist(row) < 0.7
            and (abs(float(row["init_y_m"])) >= 0.4 or abs(float(row["init_yaw_deg"])) >= 10.0),
        ),
        (
            "near<0.7 and |y|>=0.4",
            lambda row: _estimated_tip_front_dist(row) < 0.7
            and abs(float(row["init_y_m"])) >= 0.4,
        ),
        (
            "near<0.7 and |yaw|>=10",
            lambda row: _estimated_tip_front_dist(row) < 0.7
            and abs(float(row["init_yaw_deg"])) >= 10.0,
        ),
        (
            "far>=0.9",
            lambda row: _estimated_tip_front_dist(row) >= 0.9,
        ),
    ]

    lines = [
        "## Near Misalignment Buckets",
        "",
        f"- Estimated with fork_forward_offset={FORK_FORWARD_OFFSET_M:g} m and pallet_depth={PALLET_DEPTH_M:g} m.",
        "",
        "| Bucket | Episodes | Strict Success | Push-Free Success | Dirty Insert | Timeout | Mean Max Disp | Mean Tip Front Dist |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, predicate in cases:
        bucket_rows = [row for row in rows if predicate(row)]
        mean_dist = (
            sum(_estimated_tip_front_dist(row) for row in bucket_rows) / len(bucket_rows)
            if bucket_rows
            else 0.0
        )
        lines.append(
            "| {label} | {episodes} | {strict} | {push_free} | {dirty} | {timeout} | {disp} | {dist} |".format(
                label=label,
                episodes=len(bucket_rows),
                strict=_fmt(_rate(bucket_rows, "strict_success")),
                push_free=_fmt(_rate(bucket_rows, "push_free_success")),
                dirty=_fmt(_rate(bucket_rows, "dirty_insert")),
                timeout=_fmt(_rate(bucket_rows, "timeout")),
                disp=_fmt(_mean(bucket_rows, "max_pallet_disp_xy")),
                dist=_fmt(mean_dist),
            )
        )
    return lines


def _summarize_episode_bins(episode_csv: Path) -> str:
    x_bins = [(-4.0, -3.7), (-3.7, -3.35), (-3.35, -3.0)]
    y_bins = [(-0.6, -0.3), (-0.3, 0.0), (0.0, 0.3), (0.3, 0.6)]
    yaw_bins = [(-14.3, -7.0), (-7.0, 0.0), (0.0, 7.0), (7.0, 14.3)]

    with episode_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No episode rows found: {episode_csv}")

    required = {"init_x_m", "init_y_m", "init_yaw_deg", "strict_success", "push_free_success", "dirty_insert", "timeout", "max_pallet_disp_xy"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"{episode_csv} missing required bin fields: {sorted(missing)}")

    def table_for(name: str, key: str, bins: list[tuple[float, float]]) -> list[str]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            label = _bin_label(float(row[key]), bins)
            grouped.setdefault(label, []).append(row)

        lines = [
            f"## {name} Buckets",
            "",
            "| Bucket | Episodes | Strict Success | Push-Free Success | Dirty Insert | Timeout | Mean Max Disp |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        ordered_labels = [_bin_label((lo + hi) * 0.5, bins) for lo, hi in bins]
        if "out_of_range" in grouped:
            ordered_labels.append("out_of_range")
        for label in ordered_labels:
            bucket_rows = grouped.get(label, [])
            lines.append(
                "| {label} | {episodes} | {strict} | {push_free} | {dirty} | {timeout} | {disp} |".format(
                    label=label,
                    episodes=len(bucket_rows),
                    strict=_fmt(_rate(bucket_rows, "strict_success")),
                    push_free=_fmt(_rate(bucket_rows, "push_free_success")),
                    dirty=_fmt(_rate(bucket_rows, "dirty_insert")),
                    timeout=_fmt(_rate(bucket_rows, "timeout")),
                    disp=_fmt(_mean(bucket_rows, "max_pallet_disp_xy")),
                )
            )
        return lines

    lines = [
        "# GeoEdge Reset Bucket Summary",
        "",
        f"- Episodes CSV: `{episode_csv}`",
        f"- Total episodes: {len(rows)}",
        f"- Overall strict success: {_fmt(_rate(rows, 'strict_success'))}",
        f"- Overall push-free success: {_fmt(_rate(rows, 'push_free_success'))}",
        f"- Overall dirty insert: {_fmt(_rate(rows, 'dirty_insert'))}",
        f"- Overall mean max displacement: {_fmt(_mean(rows, 'max_pallet_disp_xy'))} m",
        "",
    ]
    lines.extend(table_for("Initial X", "init_x_m", x_bins))
    lines.append("")
    lines.extend(table_for("Initial Y", "init_y_m", y_bins))
    lines.append("")
    lines.extend(table_for("Initial Yaw Deg", "init_yaw_deg", yaw_bins))
    lines.append("")
    lines.extend(_summarize_near_misalignment(rows))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GeoEdge strict eval acceptance")
    parser.add_argument("summaries", nargs="*", type=Path)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--min-avg-success", type=float, default=0.60)
    parser.add_argument("--min-seed-success", type=float, default=0.50)
    parser.add_argument("--max-avg-pallet-disp", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.episodes_csv is not None:
        report = _summarize_episode_bins(args.episodes_csv)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
        print(report)
        return

    rows = [_load(path) for path in args.summaries]
    if not rows:
        raise SystemExit("No summaries provided.")

    avg_success = sum(float(row.get("strict_success_rate", 0.0)) for row in rows) / len(rows)
    min_success = min(float(row.get("strict_success_rate", 0.0)) for row in rows)
    avg_disp = sum(float(row.get("mean_max_pallet_disp_xy", 0.0)) for row in rows) / len(rows)
    passed = (
        avg_success >= args.min_avg_success
        and min_success >= args.min_seed_success
        and avg_disp <= args.max_avg_pallet_disp
    )

    lines = [
        "# GeoEdge Strict Eval Summary",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Average strict success: {_fmt(avg_success)} (threshold >= {_fmt(args.min_avg_success)})",
        f"- Worst seed/checkpoint strict success: {_fmt(min_success)} (threshold >= {_fmt(args.min_seed_success)})",
        f"- Average max pallet displacement: {_fmt(avg_disp)} m (threshold <= {_fmt(args.max_avg_pallet_disp)})",
        "",
        "| Label | Episodes | Strict Success | Inserted | Strict Geometry | Lifted | Hold Entry | Push-Free Success | Mean Max Disp | Bottleneck |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| {label} | {episodes} | {success} | {inserted} | {geom} | {lifted} | {hold} | {push_free} | {disp} | {bottleneck} |".format(
                label=row.get("label", Path(row["_path"]).stem),
                episodes=int(row.get("total_episodes", 0)),
                success=_fmt(float(row.get("strict_success_rate", 0.0))),
                inserted=_fmt(float(row.get("ever_inserted_rate", 0.0))),
                geom=_fmt(float(row.get("ever_strict_geom_rate", 0.0))),
                lifted=_fmt(float(row.get("ever_lifted_rate", 0.0))),
                hold=_fmt(float(row.get("ever_hold_entry_rate", 0.0))),
                push_free=_fmt(float(row.get("push_free_success_rate", 0.0))),
                disp=_fmt(float(row.get("mean_max_pallet_disp_xy", 0.0))),
                bottleneck=_lowest_funnel_stage(row),
            )
        )

    if not passed:
        lines.extend(
            [
                "",
                "## Failure Report",
                "",
                "Use the lowest funnel stage per row to choose the next intervention. Stage-A insert-only evals ignore lift as a bottleneck.",
            ]
        )

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
