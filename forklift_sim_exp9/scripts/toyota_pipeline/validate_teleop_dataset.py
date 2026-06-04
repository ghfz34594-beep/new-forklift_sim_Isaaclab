"""Validate Toyota teleop approach sessions before BC training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


parser = argparse.ArgumentParser(description="Validate Toyota teleop sessions for BC warm start")
parser.add_argument("--dataset_dir", type=Path, required=True)
parser.add_argument("--min_sessions", type=int, default=20)
parser.add_argument("--min_clean_sessions", type=int, default=10)
parser.add_argument("--min_steps", type=int, default=80)
parser.add_argument("--min_insert_depth_m", type=float, default=0.45)
parser.add_argument("--push_free_disp_m", type=float, default=0.05)
parser.add_argument("--max_dirty_disp_m", type=float, default=0.20)
parser.add_argument(
    "--episode_mode",
    choices=("auto", "whole_session"),
    default="auto",
    help="In auto mode, evaluate each episode separately. Legacy CSVs without episode_id are split after done=True rows.",
)
parser.add_argument("--require_summary", action="store_true")
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _bool(row: dict[str, str], key: str, default: bool = False) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return default


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split_episodes(rows: list[dict[str, str]]) -> list[dict]:
    if not rows:
        return []
    if args.episode_mode == "whole_session":
        return [{"episode_id": 0, "rows": rows}]
    if "episode_id" in rows[0]:
        buckets: dict[int, list[dict[str, str]]] = {}
        order: list[int] = []
        for row in rows:
            try:
                episode_id = int(float(row.get("episode_id", 0)))
            except ValueError:
                episode_id = 0
            if episode_id not in buckets:
                buckets[episode_id] = []
                order.append(episode_id)
            buckets[episode_id].append(row)
        return [{"episode_id": episode_id, "rows": buckets[episode_id]} for episode_id in order]

    episodes = []
    current: list[dict[str, str]] = []
    episode_id = 0
    for row in rows:
        current.append(row)
        if _bool(row, "done"):
            episodes.append({"episode_id": episode_id, "rows": current})
            episode_id += 1
            current = []
    if current:
        episodes.append({"episode_id": episode_id, "rows": current})
    return episodes


def _episode_summary(episode_id: int, rows: list[dict[str, str]]) -> dict:
    max_insert = max((_float(row, "insert_depth_m") for row in rows), default=0.0)
    disp_values = [_float(row, "pallet_disp_xy_m") for row in rows]
    initial_disp = disp_values[0] if disp_values else 0.0
    max_disp = max(disp_values, default=0.0)
    max_relative_disp = max((value - initial_disp for value in disp_values), default=0.0)
    max_lift = max((_float(row, "lift_height_m") for row in rows), default=0.0)
    max_lift_joint = max((_float(row, "lift_joint_m") for row in rows), default=0.0)
    mean_abs_drive = sum(abs(_float(row, "action_drive")) for row in rows) / max(len(rows), 1)
    mean_abs_steer = sum(abs(_float(row, "action_steer")) for row in rows) / max(len(rows), 1)
    done_reasons = sorted({row.get("done_reason", "running") for row in rows})
    has_insert = max_insert >= float(args.min_insert_depth_m)
    clean = has_insert and max_relative_disp <= float(args.push_free_disp_m)
    dirty = max_relative_disp > float(args.max_dirty_disp_m)
    return {
        "episode_id": int(episode_id),
        "rows": len(rows),
        "start_step": int(float(rows[0].get("step", 0))) if rows else 0,
        "end_step": int(float(rows[-1].get("step", 0))) if rows else 0,
        "max_insert_depth_m": max_insert,
        "max_pallet_disp_xy_m": max_disp,
        "initial_pallet_disp_xy_m": initial_disp,
        "relative_max_pallet_disp_xy_m": max_relative_disp,
        "max_lift_height_m": max_lift,
        "max_lift_joint_m": max_lift_joint,
        "mean_abs_drive": mean_abs_drive,
        "mean_abs_steer": mean_abs_steer,
        "ever_done": any(_bool(row, "done") for row in rows),
        "done_reasons": done_reasons,
        "has_insert": has_insert,
        "clean_insert": clean,
        "dirty_push": dirty,
        "usable_for_bc": has_insert and len(rows) >= int(args.min_steps) and not dirty,
    }


def _session_summary(session_dir: Path) -> dict:
    csv_path = session_dir / "metadata.csv"
    summary_path = session_dir / "summary.json"
    left_dir = session_dir / "left"
    right_dir = session_dir / "right"
    errors: list[str] = []

    if not csv_path.is_file():
        errors.append("missing metadata.csv")
        rows: list[dict[str, str]] = []
    else:
        rows = _load_rows(csv_path)
    if args.require_summary and not summary_path.is_file():
        errors.append("missing summary.json")
    if not left_dir.is_dir() or not right_dir.is_dir():
        errors.append("missing left/right image directories")

    image_rows = [row for row in rows if row.get("image_left") and row.get("image_right")]
    missing_images = 0
    for row in image_rows:
        if not (session_dir / row["image_left"]).is_file() or not (session_dir / row["image_right"]).is_file():
            missing_images += 1
    if missing_images:
        errors.append(f"missing image files: {missing_images}")

    episodes = [_episode_summary(item["episode_id"], item["rows"]) for item in _split_episodes(rows)]
    successful_episodes = [item for item in episodes if item["has_insert"]]
    clean_episodes = [item for item in episodes if item["clean_insert"]]
    usable_episodes = [item for item in episodes if item["usable_for_bc"]]
    best_episode = max(episodes, key=lambda item: item["max_insert_depth_m"], default=None)
    final_episode = episodes[-1] if episodes else None
    max_insert = max((item["max_insert_depth_m"] for item in episodes), default=0.0)
    max_disp = max((item["max_pallet_disp_xy_m"] for item in episodes), default=0.0)
    max_lift = max((item["max_lift_height_m"] for item in episodes), default=0.0)
    max_lift_joint = max((item["max_lift_joint_m"] for item in episodes), default=0.0)
    mean_abs_drive = sum(abs(_float(row, "action_drive")) for row in rows) / max(len(rows), 1)
    mean_abs_steer = sum(abs(_float(row, "action_steer")) for row in rows) / max(len(rows), 1)
    ever_done = any(_bool(row, "done") for row in rows)
    done_reasons = sorted({row.get("done_reason", "running") for row in rows})
    has_insert = bool(successful_episodes)
    clean = bool(clean_episodes)
    dirty = all(item["dirty_push"] for item in episodes) if episodes else False
    usable_for_bc = bool(rows) and len(rows) >= int(args.min_steps) and bool(usable_episodes) and not errors

    if rows and len(rows) < int(args.min_steps):
        errors.append(f"too short: {len(rows)} < {args.min_steps}")
    if episodes and not usable_episodes:
        errors.append("no usable episode")

    return {
        "session": str(session_dir),
        "rows": len(rows),
        "image_rows": len(image_rows),
        "episodes": len(episodes),
        "successful_episodes": len(successful_episodes),
        "clean_episodes": len(clean_episodes),
        "usable_episodes": len(usable_episodes),
        "best_episode_id": best_episode["episode_id"] if best_episode else None,
        "final_episode_id": final_episode["episode_id"] if final_episode else None,
        "best_episode": best_episode,
        "final_episode": final_episode,
        "episodes_detail": episodes,
        "max_insert_depth_m": max_insert,
        "max_pallet_disp_xy_m": max_disp,
        "max_lift_height_m": max_lift,
        "max_lift_joint_m": max_lift_joint,
        "mean_abs_drive": mean_abs_drive,
        "mean_abs_steer": mean_abs_steer,
        "ever_done": ever_done,
        "done_reasons": done_reasons,
        "has_insert": has_insert,
        "clean_insert": clean,
        "dirty_push": dirty,
        "usable_for_bc": usable_for_bc and not dirty,
        "errors": errors,
    }


def main() -> None:
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    session_dirs = sorted(
        path for path in args.dataset_dir.iterdir() if path.is_dir() and ((path / "metadata.csv").is_file() or (path / "left").is_dir())
    )
    summaries = [_session_summary(path) for path in session_dirs]
    clean_count = sum(1 for item in summaries if item["clean_insert"])
    usable_count = sum(1 for item in summaries if item["usable_for_bc"])
    failed = [
        item for item in summaries
        if item["errors"] or not item["usable_for_bc"]
    ]
    aggregate = {
        "dataset_dir": str(args.dataset_dir),
        "sessions": len(summaries),
        "usable_sessions": usable_count,
        "clean_sessions": clean_count,
        "required_sessions": int(args.min_sessions),
        "required_clean_sessions": int(args.min_clean_sessions),
        "pass": (
            len(summaries) >= int(args.min_sessions)
            and clean_count >= int(args.min_clean_sessions)
            and usable_count >= int(args.min_sessions)
        ),
        "sessions_detail": summaries,
    }
    output = args.output or (args.dataset_dir / "validation_summary.json")
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: aggregate[k] for k in ("sessions", "usable_sessions", "clean_sessions", "pass")}, indent=2))
    if failed:
        print("[validate] sessions with issues:")
        for item in failed[:20]:
            print(f"  - {item['session']}: {', '.join(item['errors']) or 'not usable'}")
    if not aggregate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
