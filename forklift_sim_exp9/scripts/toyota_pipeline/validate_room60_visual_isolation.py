"""Validate single-process multi-env RGB visual acceptance.

This wrapper runs the per-env camera recorder for representative env ids and
merges their machine-readable pass/fail summaries into one student-training
acceptance report.  The filename is kept for compatibility with older Room60
commands, but the script also validates CleanView45.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data/jianshi/projects/forklift_sim_exp9")
PIPELINE_DIR = PROJECT_ROOT / "scripts/toyota_pipeline"


parser = argparse.ArgumentParser(description="Validate single-process multi-env RGB visual acceptance")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--run_wrapper", type=str, default=str(PIPELINE_DIR / "run_isaaclab_env.sh"))
parser.add_argument("--record_script", type=str, default=str(PIPELINE_DIR / "record_multi_env_camera_input.py"))
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--env_ids", type=int, nargs="*", default=None)
parser.add_argument(
    "--coverage_mode",
    choices=("legacy", "all", "stratified", "random"),
    default="stratified",
    help="Which target env ids to record. legacy keeps the old {0, middle, last} behavior.",
)
parser.add_argument(
    "--coverage_count",
    type=int,
    default=16,
    help="Target env count for stratified/random coverage. Ignored for --coverage_mode all.",
)
parser.add_argument(
    "--coverage_seed",
    type=int,
    default=20260528,
    help="Random seed used by --coverage_mode random.",
)
parser.add_argument(
    "--min_checked_envs",
    type=int,
    default=None,
    help="Minimum number of target env recordings required for pass. Defaults to num_envs for all, else len(env_ids).",
)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--warmup_steps", type=int, default=4)
parser.add_argument("--record_every", type=int, default=2)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--seed", type=int, default=20260526)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--env_spacing", type=float, default=20.0)
parser.add_argument("--camera_far", type=float, default=8.0)
parser.add_argument("--dual_camera_hfov_deg", type=float, default=100.0)
parser.add_argument("--dual_camera_left_pos", type=float, nargs=3, default=(150.0, 75.0, 140.0))
parser.add_argument("--dual_camera_right_pos", type=float, nargs=3, default=(150.0, -75.0, 140.0))
parser.add_argument("--dual_camera_left_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, -20.0))
parser.add_argument("--dual_camera_right_rpy_deg", type=float, nargs=3, default=(0.0, 40.0, 20.0))
parser.add_argument("--vision_room", action="store_true", default=False)
parser.add_argument("--no_vision_room", action="store_false", dest="vision_room")
parser.add_argument("--red_component_gate", type=int, default=1)
parser.add_argument("--max_second_red_area_px", type=int, default=250)
parser.add_argument("--min_fork_red_area_px", type=int, default=250)
parser.add_argument("--max_red_area_fraction", type=float, default=0.20)
parser.add_argument("--pallet_visibility_audit", action="store_true")
parser.add_argument("--pallet_visible_min_area_px", type=int, default=250)
parser.add_argument("--pallet_confident_min_fraction", type=float, default=0.015)
parser.add_argument("--sentinel_min_area_px", type=int, default=64)
parser.add_argument("--sentinel_room_probes_all_envs", action="store_true", default=False)
parser.add_argument("--no_sentinel_room_probes_all_envs", action="store_false", dest="sentinel_room_probes_all_envs")
parser.add_argument("--sentinel_foreign_envs", action="store_true", default=True)
parser.add_argument("--no_sentinel_foreign_envs", action="store_false", dest="sentinel_foreign_envs")
parser.add_argument("--audit_pose", choices=("reset", "preinsert"), default="preinsert")
parser.add_argument("--audit_preinsert_gap_m", type=float, default=0.35)
parser.add_argument(
    "--preinsert_pose_sweep",
    action="store_true",
    help="Vary preinsert yaw/lateral/gap per env so mosaic frames are not identical by construction.",
)
parser.add_argument("--preinsert_sweep_yaw_min_deg", type=float, default=-14.0)
parser.add_argument("--preinsert_sweep_yaw_max_deg", type=float, default=14.0)
parser.add_argument("--record_mosaic", action="store_true", default=True)
parser.add_argument("--no_record_mosaic", action="store_false", dest="record_mosaic")
parser.add_argument("--mosaic_max_envs", type=int, default=16)
parser.add_argument("--mosaic_cols", type=int, default=4)
parser.add_argument(
    "--mosaic_coverage_mode",
    choices=("legacy", "checked", "all", "stratified", "random"),
    default="checked",
    help="Env ids included in each recorder's mosaic stats.",
)
parser.add_argument("--mosaic_coverage_count", type=int, default=None)
parser.add_argument("--mosaic_chunk_size", type=int, default=128)
parser.add_argument("--mosaic_save_frames", action="store_true", default=True)
parser.add_argument("--no_mosaic_save_frames", action="store_false", dest="mosaic_save_frames")
parser.add_argument(
    "--require_full_mosaic_coverage",
    action="store_true",
    help="Require union of mosaic env ids to cover every env before the report can pass.",
)
parser.add_argument(
    "--require_all_env_ids",
    action="store_true",
    help="Require target env recordings for every env before the report can pass.",
)
parser.add_argument(
    "--required_gate",
    choices=("student", "foreign", "learnability"),
    default="student",
    help=(
        "Which gate controls the process exit code. The report pass field still "
        "means student double-gate pass."
    ),
)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--no_headless", action="store_false", dest="headless")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--continue_on_failure", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--timeout_s", type=float, default=900.0)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_command(path: Path, cmd: list[str]) -> None:
    path.write_text(" ".join(_shell_quote(part) for part in cmd) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ids_by_mode(
    *,
    num_envs: int,
    mode: str,
    count: int,
    seed: int,
    explicit_ids: list[int] | None = None,
) -> list[int]:
    if explicit_ids:
        ids = sorted({int(v) for v in explicit_ids})
    elif mode == "all":
        ids = list(range(int(num_envs)))
    elif mode == "legacy":
        ids = sorted({0, int(num_envs) // 2, int(num_envs) - 1})
    elif mode == "random":
        rng = random.Random(int(seed))
        sample_count = min(max(1, int(count)), int(num_envs))
        ids = sorted(rng.sample(range(int(num_envs)), sample_count))
    elif mode == "stratified":
        sample_count = min(max(1, int(count)), int(num_envs))
        if sample_count == 1:
            ids = [0]
        else:
            ids = sorted(
                {
                    int(round(i * (int(num_envs) - 1) / float(sample_count - 1)))
                    for i in range(sample_count)
                }
            )
    else:
        raise ValueError(f"unknown coverage mode: {mode}")
    for env_id in ids:
        if env_id < 0 or env_id >= int(num_envs):
            raise ValueError(f"env_id {env_id} is outside [0, {int(num_envs)})")
    return ids


def _env_ids(args: argparse.Namespace) -> list[int]:
    return _ids_by_mode(
        num_envs=int(args.num_envs),
        mode=str(args.coverage_mode),
        count=int(args.coverage_count),
        seed=int(args.coverage_seed),
        explicit_ids=args.env_ids,
    )


def _mosaic_env_ids(args: argparse.Namespace, checked_env_ids: list[int]) -> list[int]:
    if not args.record_mosaic:
        return []
    if args.mosaic_coverage_mode == "checked":
        return sorted({int(env_id) for env_id in checked_env_ids})
    if args.mosaic_coverage_mode == "legacy":
        return _ids_by_mode(
            num_envs=int(args.num_envs),
            mode="legacy",
            count=3,
            seed=int(args.coverage_seed),
        )
    count = args.mosaic_coverage_count
    if count is None:
        count = int(args.num_envs) if args.mosaic_coverage_mode == "all" else int(args.mosaic_max_envs)
    return _ids_by_mode(
        num_envs=int(args.num_envs),
        mode=str(args.mosaic_coverage_mode),
        count=int(count),
        seed=int(args.coverage_seed),
    )


def _record_cmd(
    args: argparse.Namespace,
    env_id: int,
    env_dir: Path,
    mosaic_env_ids: list[int],
) -> list[str]:
    cmd = [
        str(args.run_wrapper),
        "-p",
        str(args.record_script),
        "--task",
        str(args.task),
        "--num_envs",
        str(int(args.num_envs)),
        "--env_id",
        str(int(env_id)),
        "--steps",
        str(int(args.steps)),
        "--warmup_steps",
        str(int(args.warmup_steps)),
        "--record_every",
        str(int(args.record_every)),
        "--fps",
        str(int(args.fps)),
        "--output_dir",
        str(env_dir),
        "--seed",
        str(int(args.seed)),
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
        "--red_component_gate",
        str(int(args.red_component_gate)),
        "--max_second_red_area_px",
        str(int(args.max_second_red_area_px)),
        "--min_fork_red_area_px",
        str(int(args.min_fork_red_area_px)),
        "--max_red_area_fraction",
        str(float(args.max_red_area_fraction)),
        "--pallet_visible_min_area_px",
        str(int(args.pallet_visible_min_area_px)),
        "--pallet_confident_min_fraction",
        str(float(args.pallet_confident_min_fraction)),
        "--sentinel_min_area_px",
        str(int(args.sentinel_min_area_px)),
        "--sentinel_audit",
        "--audit_pose",
        str(args.audit_pose),
        "--audit_preinsert_gap_m",
        str(float(args.audit_preinsert_gap_m)),
        "--preinsert_sweep_yaw_min_deg",
        str(float(args.preinsert_sweep_yaw_min_deg)),
        "--preinsert_sweep_yaw_max_deg",
        str(float(args.preinsert_sweep_yaw_max_deg)),
        "--device",
        str(args.device),
        "--enable_cameras",
    ]
    if args.preinsert_pose_sweep:
        cmd.append("--preinsert_pose_sweep")
    if args.pallet_visibility_audit:
        cmd.append("--pallet_visibility_audit")
    if args.vision_room:
        cmd.append("--vision_room")
    else:
        cmd.append("--no_vision_room")
    if args.sentinel_foreign_envs:
        cmd.append("--sentinel_foreign_envs")
    else:
        cmd.append("--no_sentinel_foreign_envs")
    if args.sentinel_room_probes_all_envs:
        cmd.append("--sentinel_room_probes_all_envs")
    else:
        cmd.append("--no_sentinel_room_probes_all_envs")
    if args.record_mosaic:
        chunk_size = max(1, int(args.mosaic_chunk_size))
        if args.mosaic_coverage_mode == "legacy":
            chunk_env_ids = list(range(min(int(args.mosaic_max_envs), int(args.num_envs))))
        else:
            try:
                env_index = mosaic_env_ids.index(int(env_id))
            except ValueError:
                env_index = 0
            chunk_start = (env_index // chunk_size) * chunk_size
            chunk_env_ids = mosaic_env_ids[chunk_start : chunk_start + chunk_size]
        cmd.extend(
            [
                "--record_mosaic",
                "--mosaic_max_envs",
                str(len(chunk_env_ids) if chunk_env_ids else int(args.mosaic_max_envs)),
                "--mosaic_cols",
                str(int(args.mosaic_cols)),
                "--mosaic_env_ids",
                *[str(int(item)) for item in chunk_env_ids],
            ]
        )
        if args.mosaic_save_frames:
            cmd.append("--mosaic_save_frames")
        else:
            cmd.append("--no_mosaic_save_frames")
    if args.headless:
        cmd.append("--headless")
    return cmd


def main() -> None:
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checked_env_ids = _env_ids(args)
    mosaic_env_ids = _mosaic_env_ids(args, checked_env_ids)

    results: list[dict[str, Any]] = []
    for env_id in checked_env_ids:
        env_dir = output_dir / f"check_env_{env_id:03d}"
        env_dir.mkdir(parents=True, exist_ok=True)
        cmd = _record_cmd(args, env_id, env_dir, mosaic_env_ids)
        _write_command(env_dir / "command.sh", cmd)
        if args.dry_run:
            print("[dry-run] " + " ".join(_shell_quote(part) for part in cmd), flush=True)
            result = {"env_id": int(env_id), "pass": False, "dry_run": True, "summary": None}
        else:
            with (env_dir / "run.log").open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(_shell_quote(part) for part in cmd) + "\n\n")
                log.flush()
                timed_out = False
                timeout_s = float(args.timeout_s)
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=timeout_s if timeout_s > 0 else None)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    log.write(f"\n[timeout] exceeded {timeout_s:.1f}s; killing process group {proc.pid}\n")
                    log.flush()
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = proc.wait()
            summary = _load_json(env_dir / "summary.json")
            foreign_leakage_pass = bool(returncode == 0 and summary and summary.get("foreign_leakage_pass") is True)
            camera_learnability_pass = bool(
                returncode == 0 and summary and summary.get("camera_learnability_pass") is True
            )
            total_pass = bool(foreign_leakage_pass and camera_learnability_pass)
            required_gate_pass = total_pass
            if args.required_gate == "foreign":
                required_gate_pass = foreign_leakage_pass
            elif args.required_gate == "learnability":
                required_gate_pass = camera_learnability_pass
            result = {
                "env_id": int(env_id),
                "returncode": int(returncode),
                "timed_out": bool(timed_out),
                "pass": total_pass,
                "foreign_leakage_pass": foreign_leakage_pass,
                "camera_learnability_pass": camera_learnability_pass,
                "required_gate_pass": bool(required_gate_pass),
                "summary_path": str(env_dir / "summary.json"),
                "summary": summary,
            }
            if not result["required_gate_pass"] and not args.continue_on_failure:
                results.append(result)
                break
        results.append(result)

    successful_results = [item for item in results if bool(item.get("returncode", 1) == 0)]
    checked_ids = sorted({int(item["env_id"]) for item in successful_results})
    requested_checked_ids = sorted({int(env_id) for env_id in checked_env_ids})
    actual_mosaic_ids = sorted(
        {
            int(env_id)
            for item in successful_results
            for env_id in (((item.get("summary") or {}).get("mosaic_env_ids")) or [])
        }
    )
    min_checked_envs = args.min_checked_envs
    if min_checked_envs is None:
        min_checked_envs = int(args.num_envs) if bool(args.require_all_env_ids) else len(requested_checked_ids)
    mosaic_coverage_pass = bool(
        not args.record_mosaic
        or not bool(args.require_full_mosaic_coverage)
        or actual_mosaic_ids == list(range(int(args.num_envs)))
    )
    mosaic_covers_requested = bool(set(requested_checked_ids).issubset(set(actual_mosaic_ids)))
    env_id_coverage_pass = bool(
        (
            len(checked_ids) >= int(min_checked_envs)
            and set(requested_checked_ids).issubset(set(checked_ids))
        )
        or (bool(args.record_mosaic) and mosaic_covers_requested and mosaic_coverage_pass)
    )
    env_id_coverage_pass = bool(
        env_id_coverage_pass
        and (not bool(args.require_all_env_ids) or actual_mosaic_ids == list(range(int(args.num_envs))) or checked_ids == list(range(int(args.num_envs))))
    )
    foreign_leakage_pass = bool(results) and all(bool(item.get("foreign_leakage_pass")) for item in results)
    camera_learnability_pass = bool(results) and all(bool(item.get("camera_learnability_pass")) for item in results)
    passed = bool(foreign_leakage_pass and camera_learnability_pass and env_id_coverage_pass and mosaic_coverage_pass)
    required_gate_pass = passed
    if args.required_gate == "foreign":
        required_gate_pass = foreign_leakage_pass
    elif args.required_gate == "learnability":
        required_gate_pass = camera_learnability_pass
    report = {
        "source": "single_process_multi_env_visual_acceptance",
        "acceptance_schema": "student_double_gate_v1",
        "pass": passed,
        "foreign_leakage_pass": foreign_leakage_pass,
        "camera_learnability_pass": camera_learnability_pass,
        "required_gate": str(args.required_gate),
        "required_gate_pass": bool(required_gate_pass),
        "task": str(args.task),
        "num_envs": int(args.num_envs),
        "coverage_mode": str(args.coverage_mode),
        "coverage_count": int(args.coverage_count),
        "coverage_seed": int(args.coverage_seed),
        "env_ids_requested": requested_checked_ids,
        "env_ids": [int(item["env_id"]) for item in results],
        "env_ids_checked": checked_ids,
        "env_id_coverage": {
            "mode": str(args.coverage_mode),
            "requested_count": len(requested_checked_ids),
            "checked_count": len(checked_ids),
            "num_envs": int(args.num_envs),
            "coverage_fraction": float(len(checked_ids) / max(1, int(args.num_envs))),
            "min_checked_envs": int(min_checked_envs),
            "require_all_env_ids": bool(args.require_all_env_ids),
            "mosaic_covers_requested": bool(mosaic_covers_requested),
            "pass": bool(env_id_coverage_pass),
        },
        "env_spacing": float(args.env_spacing),
        "camera_far": float(args.camera_far),
        "dual_camera_config": {
            "hfov_deg": float(args.dual_camera_hfov_deg),
            "left_pos_local": [float(v) for v in args.dual_camera_left_pos],
            "right_pos_local": [float(v) for v in args.dual_camera_right_pos],
            "left_rpy_local_deg": [float(v) for v in args.dual_camera_left_rpy_deg],
            "right_rpy_local_deg": [float(v) for v in args.dual_camera_right_rpy_deg],
        },
        "vision_room_enable": bool(args.vision_room),
        "visual_isolation_mode": "room" if bool(args.vision_room) else "far_clip_spacing",
        "sentinel_audit": True,
        "pallet_visibility_audit": bool(args.pallet_visibility_audit),
        "pallet_visibility_thresholds": {
            "visible_min_area_px": int(args.pallet_visible_min_area_px),
            "confident_min_fraction": float(args.pallet_confident_min_fraction),
        },
        "sentinel_room_probes_all_envs": bool(args.sentinel_room_probes_all_envs),
        "sentinel_foreign_envs": bool(args.sentinel_foreign_envs),
        "audit_pose": str(args.audit_pose),
        "preinsert_pose_sweep": bool(args.preinsert_pose_sweep),
        "preinsert_sweep_yaw_range_deg": [
            float(args.preinsert_sweep_yaw_min_deg),
            float(args.preinsert_sweep_yaw_max_deg),
        ],
        "record_mosaic": bool(args.record_mosaic),
        "mosaic_coverage_mode": str(args.mosaic_coverage_mode),
        "mosaic_env_ids_requested": mosaic_env_ids,
        "mosaic_env_ids_checked": actual_mosaic_ids,
        "mosaic_env_coverage": {
            "mode": str(args.mosaic_coverage_mode),
            "checked_count": len(actual_mosaic_ids),
            "num_envs": int(args.num_envs),
            "coverage_fraction": float(len(actual_mosaic_ids) / max(1, int(args.num_envs))),
            "chunk_size": int(args.mosaic_chunk_size),
            "save_frames": bool(args.mosaic_save_frames),
            "require_full_mosaic_coverage": bool(args.require_full_mosaic_coverage),
            "pass": bool(mosaic_coverage_pass),
        },
        "mosaic_max_envs": int(args.mosaic_max_envs),
        "checks": results,
    }
    report_path = output_dir / "visual_isolation_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not required_gate_pass and not args.dry_run:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
