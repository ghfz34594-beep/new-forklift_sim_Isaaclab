#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
RUN="${RUN:-${ROOT}/scripts/toyota_pipeline/run_isaaclab_env.sh}"
SUMMARY_JSON="${SUMMARY_JSON:-${ROOT}/outputs/progress_teacher_eval/progress_teacher_curriculum_v311_model_399_full_summary.json}"
EPISODES_CSV="${EPISODES_CSV:-${SUMMARY_JSON%_summary.json}_episodes.csv}"
SEEDS="${SEEDS:-20260427 20260428 20260429 20260430 20260431}"

EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-256}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_VIDEO="${RUN_VIDEO:-1}"
RUN_FIXED_CASES="${RUN_FIXED_CASES:-1}"

VIDEO_LENGTH="${VIDEO_LENGTH:-480}"
VIDEO_WIDTH="${VIDEO_WIDTH:-960}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-540}"
VIDEO_NUM_ENVS="${VIDEO_NUM_ENVS:-1}"
FIXED_CASE_SEED="${FIXED_CASE_SEED:-9001}"

if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "[FATAL] summary json not found: ${SUMMARY_JSON}" >&2
  exit 1
fi

json_field() {
  python - "$SUMMARY_JSON" "$1" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
value = data[sys.argv[2]]
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

TASK="$(json_field task)"
CHECKPOINT="$(json_field checkpoint)"
BASE_LABEL="$(json_field label)"
RESET_PROFILE="$(json_field reset_profile)"
STAGE1_EVAL="$(json_field stage1_eval)"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[FATAL] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${OUT_DIR:-}" == "" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${ROOT}/outputs/progress_teacher_visual_eval/${stamp}_${BASE_LABEL}"
fi
mkdir -p "${OUT_DIR}/eval" "${OUT_DIR}/videos" "${OUT_DIR}/logs"

STAGE_ARGS=()
if [[ "${STAGE1_EVAL}" == "true" || "${STAGE1_EVAL}" == "1" ]]; then
  STAGE_ARGS=(--stage1_eval --reset_profile "${RESET_PROFILE}")
fi

echo "[progress-viz] summary=${SUMMARY_JSON}"
echo "[progress-viz] task=${TASK}"
echo "[progress-viz] checkpoint=${CHECKPOINT}"
echo "[progress-viz] seeds=${SEEDS}"
echo "[progress-viz] out_dir=${OUT_DIR}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  for seed in ${SEEDS}; do
    label="${BASE_LABEL}_seed${seed}"
    log_path="${OUT_DIR}/logs/eval_seed${seed}.log"
    echo "[progress-viz] eval seed=${seed} -> ${log_path}"
    "${RUN}" -p "${ROOT}/scripts/eval_geoedge_checkpoint.py" \
      --task "${TASK}" \
      --checkpoint "${CHECKPOINT}" \
      --label "${label}" \
      --num_envs "${EVAL_NUM_ENVS}" \
      --rollouts "${EVAL_ROLLOUTS}" \
      --seed "${seed}" \
      --output_dir "${OUT_DIR}/eval" \
      --headless \
      "${STAGE_ARGS[@]}" 2>&1 | tee "${log_path}"
  done
fi

if [[ "${RUN_VIDEO}" == "1" ]]; then
  for seed in ${SEEDS}; do
    video_dir="${OUT_DIR}/videos/random_seed_${seed}"
    log_path="${OUT_DIR}/logs/video_seed${seed}.log"
    mkdir -p "${video_dir}"
    echo "[progress-viz] video seed=${seed} -> ${video_dir}"
    "${RUN}" -p "${ROOT}/scripts/record_geoedge_video.py" \
      --task "${TASK}" \
      --checkpoint "${CHECKPOINT}" \
      --num_envs "${VIDEO_NUM_ENVS}" \
      --episodes 1 \
      --video_length "${VIDEO_LENGTH}" \
      --video_folder "${video_dir}" \
      --seed "${seed}" \
      --target_env_id 0 \
      --video_width "${VIDEO_WIDTH}" \
      --video_height "${VIDEO_HEIGHT}" \
      --headless \
      "${STAGE_ARGS[@]}" 2>&1 | tee "${log_path}"
  done
fi

FIXED_CASES_TSV="${OUT_DIR}/fixed_cases.tsv"
if [[ "${RUN_FIXED_CASES}" == "1" ]]; then
  if [[ ! -f "${EPISODES_CSV}" ]]; then
    echo "[WARN] episodes csv not found, skipping fixed cases: ${EPISODES_CSV}" >&2
  else
    python - "$SUMMARY_JSON" "$EPISODES_CSV" "$FIXED_CASES_TSV" <<'PY'
import csv
import json
import math
import sys

summary_path, csv_path, out_path = sys.argv[1:4]
with open(summary_path) as f:
    summary = json.load(f)
with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))

push_free_thresh = float(summary.get("push_free_disp_thresh_m", 0.05))
cases = []
seen = set()

def f(row, key):
    return float(row[key])

def add_case(tag, predicate, sort_key, reverse=True):
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return
    row = sorted(candidates, key=sort_key, reverse=reverse)[0]
    row_id = (row.get("rollout"), row.get("env_id"))
    if row_id in seen:
        return
    seen.add(row_id)
    cases.append(
        {
            "tag": tag,
            "init_x_m": row["init_x_m"],
            "init_y_m": row["init_y_m"],
            "init_yaw_deg": row["init_yaw_deg"],
            "strict_success": row["strict_success"],
            "ever_inserted": row["ever_inserted"],
            "dirty_insert": row["dirty_insert"],
            "max_pallet_disp_xy": row["max_pallet_disp_xy"],
            "episode_length": row["episode_length"],
            "source_rollout": row["rollout"],
            "source_env_id": row["env_id"],
        }
    )

add_case(
    "clean_success_low_push",
    lambda r: int(r["strict_success"]) == 1 and f(r, "max_pallet_disp_xy") <= 0.001,
    lambda r: f(r, "episode_length"),
    reverse=True,
)
add_case(
    "clean_success_long",
    lambda r: int(r["strict_success"]) == 1 and f(r, "max_pallet_disp_xy") <= push_free_thresh,
    lambda r: f(r, "episode_length"),
    reverse=True,
)
add_case(
    "dirty_insert",
    lambda r: int(r["dirty_insert"]) == 1,
    lambda r: f(r, "max_pallet_disp_xy"),
    reverse=True,
)
add_case(
    "worst_push_no_insert",
    lambda r: int(r["ever_inserted"]) == 0 and f(r, "max_pallet_disp_xy") > push_free_thresh,
    lambda r: f(r, "max_pallet_disp_xy"),
    reverse=True,
)
add_case(
    "fast_push_no_insert",
    lambda r: int(r["ever_inserted"]) == 0 and f(r, "max_pallet_disp_xy") > push_free_thresh,
    lambda r: f(r, "episode_length"),
    reverse=False,
)

with open(out_path, "w", newline="") as f_out:
    fieldnames = [
        "tag",
        "init_x_m",
        "init_y_m",
        "init_yaw_deg",
        "strict_success",
        "ever_inserted",
        "dirty_insert",
        "max_pallet_disp_xy",
        "episode_length",
        "source_rollout",
        "source_env_id",
    ]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(cases)

print(f"[fixed-cases] wrote {len(cases)} cases to {out_path}")
PY

    tail -n +2 "${FIXED_CASES_TSV}" | while IFS=$'\t' read -r tag x y yaw strict_success ever_inserted dirty_insert max_disp ep_len source_rollout source_env_id; do
      video_dir="${OUT_DIR}/videos/fixed_${tag}"
      log_path="${OUT_DIR}/logs/video_fixed_${tag}.log"
      mkdir -p "${video_dir}"
      echo "[progress-viz] fixed case=${tag} x=${x} y=${y} yaw=${yaw} -> ${video_dir}"
      "${RUN}" -p "${ROOT}/scripts/record_geoedge_video.py" \
        --task "${TASK}" \
        --checkpoint "${CHECKPOINT}" \
        --num_envs 1 \
        --episodes 1 \
        --video_length "${VIDEO_LENGTH}" \
        --video_folder "${video_dir}" \
        --seed "${FIXED_CASE_SEED}" \
        --fixed_stage1_init "${x}" "${y}" "${yaw}" \
        --target_env_id 0 \
        --video_width "${VIDEO_WIDTH}" \
        --video_height "${VIDEO_HEIGHT}" \
        --headless \
        "${STAGE_ARGS[@]}" 2>&1 | tee "${log_path}"
    done
  fi
fi

python - "$OUT_DIR" "$SUMMARY_JSON" "$FIXED_CASES_TSV" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
fixed_cases_tsv = Path(sys.argv[3])

with summary_json.open() as f:
    source_summary = json.load(f)

eval_summaries = []
for path in sorted((out_dir / "eval").glob("*_summary.json")):
    with path.open() as f:
        data = json.load(f)
    data["_path"] = str(path)
    eval_summaries.append(data)

def mean(key):
    values = [float(item[key]) for item in eval_summaries if key in item]
    return statistics.fmean(values) if values else None

def pstdev(key):
    values = [float(item[key]) for item in eval_summaries if key in item]
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None

aggregate = {
    "source_summary": str(summary_json),
    "source_checkpoint": source_summary.get("checkpoint"),
    "task": source_summary.get("task"),
    "output_dir": str(out_dir),
    "eval_summaries": eval_summaries,
    "aggregate": {
        "num_eval_seeds": len(eval_summaries),
        "strict_success_rate_mean": mean("strict_success_rate"),
        "strict_success_rate_std": pstdev("strict_success_rate"),
        "clean_insert_rate_mean": mean("clean_insert_rate"),
        "dirty_insert_rate_mean": mean("dirty_insert_rate"),
        "push_no_insert_rate_mean": mean("push_no_insert_rate"),
        "mean_max_pallet_disp_xy_mean": mean("mean_max_pallet_disp_xy"),
        "p90_max_pallet_disp_xy_mean": mean("p90_max_pallet_disp_xy"),
    },
    "videos": [str(path) for path in sorted((out_dir / "videos").rglob("*.mp4"))],
}

(out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))

lines = [
    "# Progress Teacher Visual Eval",
    "",
    f"- Source summary: `{summary_json}`",
    f"- Task: `{source_summary.get('task')}`",
    f"- Checkpoint: `{source_summary.get('checkpoint')}`",
    f"- Output dir: `{out_dir}`",
    "",
]

if eval_summaries:
    lines += [
        "## Eval Seeds",
        "",
        "| seed | strict_success | clean_insert | dirty_insert | push_no_insert | mean_max_disp | video_dir |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in eval_summaries:
        seed = item.get("eval_seed")
        video_dir = out_dir / "videos" / f"random_seed_{seed}"
        lines.append(
            "| "
            f"{seed} | "
            f"{float(item.get('strict_success_rate', 0.0)):.4f} | "
            f"{float(item.get('clean_insert_rate', 0.0)):.4f} | "
            f"{float(item.get('dirty_insert_rate', 0.0)):.4f} | "
            f"{float(item.get('push_no_insert_rate', 0.0)):.4f} | "
            f"{float(item.get('mean_max_pallet_disp_xy', 0.0)):.4f} | "
            f"`{video_dir}` |"
        )
    agg = aggregate["aggregate"]
    lines += [
        "",
        "## Aggregate",
        "",
        f"- strict_success_rate: {agg['strict_success_rate_mean']:.4f} +/- {agg['strict_success_rate_std']:.4f}",
        f"- clean_insert_rate_mean: {agg['clean_insert_rate_mean']:.4f}",
        f"- dirty_insert_rate_mean: {agg['dirty_insert_rate_mean']:.4f}",
        f"- push_no_insert_rate_mean: {agg['push_no_insert_rate_mean']:.4f}",
        f"- mean_max_pallet_disp_xy_mean: {agg['mean_max_pallet_disp_xy_mean']:.4f} m",
        "",
    ]

if fixed_cases_tsv.is_file():
    with fixed_cases_tsv.open(newline="") as f:
        cases = list(csv.DictReader(f, delimiter="\t"))
    if cases:
        lines += [
            "## Fixed Cases",
            "",
            "| case | source | init_x | init_y | yaw_deg | original_success | original_disp | video_dir |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for case in cases:
            video_dir = out_dir / "videos" / f"fixed_{case['tag']}"
            lines.append(
                "| "
                f"{case['tag']} | "
                f"r{case['source_rollout']}/e{case['source_env_id']} | "
                f"{float(case['init_x_m']):.3f} | "
                f"{float(case['init_y_m']):.3f} | "
                f"{float(case['init_yaw_deg']):.2f} | "
                f"{case['strict_success']} | "
                f"{float(case['max_pallet_disp_xy']):.4f} | "
                f"`{video_dir}` |"
            )
        lines.append("")

if aggregate["videos"]:
    lines += [
        "## MP4 Files",
        "",
    ]
    for path in aggregate["videos"]:
        lines.append(f"- `{path}`")
    lines.append("")

(out_dir / "README.md").write_text("\n".join(lines))
print(f"[progress-viz] aggregate summary: {out_dir / 'aggregate_summary.json'}")
print(f"[progress-viz] report: {out_dir / 'README.md'}")
PY

echo "[progress-viz] done: ${OUT_DIR}"
