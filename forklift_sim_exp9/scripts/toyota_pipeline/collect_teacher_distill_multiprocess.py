"""Diagnostic one-env-per-process teacher-distillation collection.

This is not the primary visual-isolation solution.  It is kept as evidence that
the teacher can perform clean insertion when renderer cross-env contamination is
removed, while the main RGB data path should validate and use single-process
CleanView45 multi-env data collection after visual acceptance passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data/jianshi/projects/forklift_sim_exp9")
PIPELINE_DIR = PROJECT_ROOT / "scripts/toyota_pipeline"
DEFAULT_TEACHER = Path(
    "/data/jianshi/projects/forklift_sim/IsaacLab/logs/rsl_rl/"
    "forklift_toyota_geoedge_progress_teacher/"
    "2026-05-25_15-20-08_progress_teacher_scratch_curriculum_v311_late_dirty_event/model_399.pt"
)


parser = argparse.ArgumentParser(description="Collect teacher RGB distillation data via multi-process num_envs=1 workers")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--teacher_checkpoint", type=str, default=str(DEFAULT_TEACHER))
parser.add_argument("--run_wrapper", type=str, default=str(PIPELINE_DIR / "run_isaaclab_env.sh"))
parser.add_argument(
    "--collect_script",
    type=str,
    default=str(PIPELINE_DIR / "collect_teacher_approach_dataset.py"),
)
parser.add_argument(
    "--sanity_script",
    type=str,
    default=str(PIPELINE_DIR / "record_multi_env_camera_input.py"),
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectCleanView-v0",
)
parser.add_argument(
    "--sanity_task",
    type=str,
    default="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0",
)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--max_parallel", type=int, default=2)
parser.add_argument("--episodes_per_worker", type=int, default=40)
parser.add_argument("--attempts_per_worker", type=int, default=80)
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--image_every", type=int, default=1)
parser.add_argument("--flush_every", type=int, default=25)
parser.add_argument("--base_seed", type=int, default=20260526)
parser.add_argument("--seed_stride", type=int, default=1000)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--env_spacing", type=float, default=20.0)
parser.add_argument("--camera_far", type=float, default=8.0)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=100.0)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(150.0, 75.0, 140.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(150.0, -75.0, 140.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, -20.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, 20.0))
parser.add_argument("--sanity_steps", type=int, default=120)
parser.add_argument("--sanity_record_every", type=int, default=2)
parser.add_argument("--sanity_fps", type=int, default=20)
parser.add_argument("--red_component_gate", type=int, default=1)
parser.add_argument("--skip_sanity", action="store_true")
parser.add_argument("--skip_collect", action="store_true")
parser.add_argument(
    "--sanity_only",
    action="store_true",
    help="Only record worker sanity videos and merge summaries; do not collect teacher labels.",
)
parser.add_argument("--merge_only", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--continue_on_failure", action="store_true")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--no_headless", action="store_false", dest="headless")
parser.add_argument("--allow_red_gate_fail", action="store_true")


@dataclass
class WorkerResult:
    worker_id: int
    sanity_returncode: int | None = None
    collect_returncode: int | None = None
    skipped_existing: bool = False
    sanity_summary: dict[str, Any] | None = None
    collect_summary: dict[str, Any] | None = None
    error: str | None = None


def _append_bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _camera_args(args: argparse.Namespace) -> list[str]:
    return [
        "--env_spacing",
        str(float(args.env_spacing)),
        "--camera_far",
        str(float(args.camera_far)),
        "--dual_camera_hfov_deg",
        str(float(args.dual_camera_hfov_deg)),
        "--dual_camera_left_pos",
        *[str(float(v)) for v in args.dual_camera_left_pos],
        "--dual_camera_right_pos",
        *[str(float(v)) for v in args.dual_camera_right_pos],
        "--dual_camera_left_rpy_deg",
        *[str(float(v)) for v in args.dual_camera_left_rpy_deg],
        "--dual_camera_right_rpy_deg",
        *[str(float(v)) for v in args.dual_camera_right_rpy_deg],
        "--no_vision_room",
    ]


def _common_isaac_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--device", str(args.device), "--enable_cameras"]
    if args.headless:
        flags.append("--headless")
    return flags


def _worker_seed(args: argparse.Namespace, worker_id: int) -> int:
    return int(args.base_seed) + int(worker_id) * int(args.seed_stride)


def _sanity_cmd(args: argparse.Namespace, worker_id: int, sanity_dir: Path) -> list[str]:
    return [
        str(args.run_wrapper),
        "-p",
        str(args.sanity_script),
        "--task",
        str(args.sanity_task),
        "--num_envs",
        "1",
        "--env_id",
        "0",
        "--steps",
        str(int(args.sanity_steps)),
        "--record_every",
        str(int(args.sanity_record_every)),
        "--fps",
        str(int(args.sanity_fps)),
        "--output_dir",
        str(sanity_dir),
        "--seed",
        str(_worker_seed(args, worker_id)),
        *_camera_args(args),
        *_common_isaac_flags(args),
    ]


def _collect_cmd(args: argparse.Namespace, worker_id: int, worker_dir: Path) -> list[str]:
    return [
        str(args.run_wrapper),
        "-p",
        str(args.collect_script),
        "--task",
        str(args.task),
        "--checkpoint",
        str(args.teacher_checkpoint),
        "--output_dir",
        str(worker_dir),
        "--num_envs",
        "1",
        "--target_clean_episodes",
        str(int(args.episodes_per_worker)),
        "--episodes",
        str(int(args.attempts_per_worker)),
        "--max_steps",
        str(int(args.max_steps)),
        "--image_every",
        str(int(args.image_every)),
        "--flush_every",
        str(int(args.flush_every)),
        "--seed",
        str(_worker_seed(args, worker_id)),
        *_camera_args(args),
        "--relabel_teacher_actions",
        *_common_isaac_flags(args),
    ]


def _write_command(path: Path, cmd: list[str]) -> None:
    path.write_text(" ".join(_shell_quote(part) for part in cmd) + "\n", encoding="utf-8")


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_command(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(_shell_quote(part) for part in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return int(proc.wait())


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _red_gate_passed(summary: dict[str, Any] | None, gate: int) -> bool:
    if not summary:
        return False
    red_max = summary.get("red_component_max") or {}
    for camera in ("left", "right"):
        stats = red_max.get(camera) or {}
        if int(stats.get("large_red_components", 999)) > int(gate):
            return False
    return True


def _run_worker(args: argparse.Namespace, output_dir: Path, worker_id: int) -> WorkerResult:
    worker_dir = output_dir / f"worker_{worker_id:03d}"
    sanity_dir = output_dir / f"worker_{worker_id:03d}_sanity"
    result = WorkerResult(worker_id=worker_id)

    if args.resume and (worker_dir / "summary.json").is_file():
        summary = _load_json(worker_dir / "summary.json") or {}
        if int(summary.get("kept_episodes", 0)) >= int(args.episodes_per_worker):
            result.skipped_existing = True
            result.collect_summary = summary
            result.sanity_summary = _load_json(sanity_dir / "summary.json")
            return result

    worker_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sanity:
        cmd = _sanity_cmd(args, worker_id, sanity_dir)
        _write_command(sanity_dir / "command.sh", cmd)
        if args.dry_run:
            print("[dry-run] " + " ".join(_shell_quote(part) for part in cmd), flush=True)
            result.sanity_returncode = 0
        else:
            result.sanity_returncode = _run_command(cmd, sanity_dir / "run.log")
            result.sanity_summary = _load_json(sanity_dir / "summary.json")
            if result.sanity_returncode != 0:
                result.error = f"sanity command failed with rc={result.sanity_returncode}"
                return result
            if not args.allow_red_gate_fail and not _red_gate_passed(result.sanity_summary, int(args.red_component_gate)):
                result.error = f"red component gate failed for worker {worker_id}"
                return result

    if not args.skip_collect and not args.sanity_only:
        cmd = _collect_cmd(args, worker_id, worker_dir)
        _write_command(worker_dir / "command.sh", cmd)
        if args.dry_run:
            print("[dry-run] " + " ".join(_shell_quote(part) for part in cmd), flush=True)
            result.collect_returncode = 0
        else:
            result.collect_returncode = _run_command(cmd, worker_dir / "run.log")
            result.collect_summary = _load_json(worker_dir / "summary.json")
            if result.collect_returncode != 0:
                result.error = f"collect command failed with rc={result.collect_returncode}"
                return result

    return result


def _merge_metadata(output_dir: Path, num_workers: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    episode_map: dict[tuple[int, str], int] = {}
    next_episode_id = 0
    worker_summaries: list[dict[str, Any]] = []
    sanity_summaries: list[dict[str, Any]] = []

    for worker_id in range(int(num_workers)):
        worker_name = f"worker_{worker_id:03d}"
        worker_dir = output_dir / worker_name
        sanity_name = f"worker_{worker_id:03d}_sanity"
        sanity_dir = output_dir / sanity_name
        csv_path = worker_dir / "metadata.csv"
        summary = _load_json(worker_dir / "summary.json") or {}
        summary["worker_id"] = worker_id
        summary["worker_dir"] = worker_name
        worker_summaries.append(summary)
        sanity_summary = _load_json(sanity_dir / "summary.json") or {}
        sanity_summary["worker_id"] = worker_id
        sanity_summary["worker_sanity_dir"] = sanity_name
        sanity_summaries.append(sanity_summary)
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                original_episode_id = str(row.get("episode_id", "0"))
                key = (worker_id, original_episode_id)
                if key not in episode_map:
                    episode_map[key] = next_episode_id
                    next_episode_id += 1
                merged = dict(row)
                merged["worker_id"] = worker_id
                merged["worker_episode_id"] = original_episode_id
                merged["episode_id"] = episode_map[key]
                merged["source_metadata"] = f"{worker_name}/metadata.csv"
                for image_key in ("image_left", "image_right"):
                    image_path = str(merged.get(image_key, ""))
                    if image_path:
                        merged[image_key] = f"{worker_name}/{image_path}"
                for key_name in merged:
                    if key_name not in fieldnames:
                        fieldnames.append(key_name)
                rows.append(merged)

    if rows:
        csv_path = output_dir / "metadata.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    total_kept = sum(int(summary.get("kept_episodes", 0)) for summary in worker_summaries)
    total_attempted = sum(int(summary.get("attempted_episodes", 0)) for summary in worker_summaries)
    merge_summary = {
        "source": "multi_process_teacher_distill",
        "num_workers": int(num_workers),
        "rows": len(rows),
        "episodes": int(next_episode_id),
        "kept_episodes": int(total_kept),
        "attempted_episodes": int(total_attempted),
        "metadata_csv": str(output_dir / "metadata.csv") if rows else None,
        "worker_summaries": worker_summaries,
        "sanity_summaries": sanity_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(merge_summary, indent=2, sort_keys=True), encoding="utf-8")
    return merge_summary


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists() and not args.merge_only:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_workers = list(output_dir.glob("worker_*/summary.json"))
    if existing_workers and not (args.resume or args.merge_only or args.overwrite):
        raise FileExistsError(
            f"{output_dir} already contains worker summaries. Use --resume, --merge_only, or --overwrite."
        )
    return output_dir


def main() -> None:
    args = parser.parse_args()
    if int(args.num_workers) < 1:
        raise ValueError("--num_workers must be >= 1")
    if int(args.max_parallel) < 1:
        raise ValueError("--max_parallel must be >= 1")
    if int(args.episodes_per_worker) < 1:
        raise ValueError("--episodes_per_worker must be >= 1")
    if int(args.attempts_per_worker) < int(args.episodes_per_worker):
        raise ValueError("--attempts_per_worker must be >= --episodes_per_worker")

    output_dir = _prepare_output_dir(args)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    (output_dir / "launcher_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    results: list[WorkerResult] = []
    if not args.merge_only:
        with ThreadPoolExecutor(max_workers=min(int(args.max_parallel), int(args.num_workers))) as executor:
            futures = {
                executor.submit(_run_worker, args, output_dir, worker_id): worker_id
                for worker_id in range(int(args.num_workers))
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "ok" if result.error is None else "failed"
                print(f"[mp_collect] worker={result.worker_id:03d} status={status} error={result.error}", flush=True)
                if result.error and not args.continue_on_failure:
                    raise RuntimeError(result.error)

    merge_summary = _merge_metadata(output_dir, int(args.num_workers))
    result_payload = {
        "results": [result.__dict__ for result in sorted(results, key=lambda item: item.worker_id)],
        "merge_summary": merge_summary,
    }
    (output_dir / "launcher_results.json").write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")

    expected_clean = int(args.num_workers) * int(args.episodes_per_worker)
    if (
        not args.dry_run
        and not args.skip_collect
        and not args.sanity_only
        and int(merge_summary.get("kept_episodes", 0)) < expected_clean
    ):
        msg = (
            f"merged clean episodes {merge_summary.get('kept_episodes', 0)} < expected {expected_clean}. "
            "Inspect worker run.log files or rerun with --resume."
        )
        if args.continue_on_failure:
            print("[mp_collect] warning: " + msg, file=sys.stderr)
        else:
            raise RuntimeError(msg)

    print("[mp_collect] merged " + json.dumps(merge_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
