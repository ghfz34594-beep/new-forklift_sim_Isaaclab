#!/usr/bin/env bash
set -euo pipefail

echo "[v41] frozen: legacy v41 curriculum is disabled for the clean teacher/visual pipeline." >&2
echo "[v41] this script intentionally has no environment-variable bypass." >&2
exit 90

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/direct_visual_v41_curriculum_20260529}"
LOG_ROOT="${ISAACLAB_DIR}/logs/rsl_rl/direct_visual_insertion_cleanview"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-20260529}"
NUM_ENVS="${NUM_ENVS:-64}"
SMOKE_ENVS="${SMOKE_ENVS:-4}"
SMOKE_ITERS="${SMOKE_ITERS:-2}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_FROM_STAGE="${RUN_FROM_STAGE:-A}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-D}"
CONTINUE_ON_GATE_FAIL="${CONTINUE_ON_GATE_FAIL:-0}"
VIS_SUMMARY="${VIS_SUMMARY:-}"

EVAL_EPISODES_DEFAULT="${EVAL_EPISODES:-12}"
EVAL_STEPS="${EVAL_STEPS:-720}"
EVAL_RECORD_EVERY="${EVAL_RECORD_EVERY:-3}"
EVAL_FPS="${EVAL_FPS:-30}"

STAGES=(A B C D)
TASK_A="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV41AInsertContinue-v0"
TASK_B="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV41BMouthAcquire-v0"
TASK_C="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV41CEdgeCornerAlign-v0"
TASK_D="Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV41DFullApproach-v0"
LABEL_A="v41a_insert_continue"
LABEL_B="v41b_mouth_acquire"
LABEL_C="v41c_edge_corner_align"
LABEL_D="v41d_full_approach"
ITERS_A="${ITERS_A:-201}"
ITERS_B="${ITERS_B:-251}"
ITERS_C="${ITERS_C:-301}"
ITERS_D="${ITERS_D:-501}"

mkdir -p "${OUTPUT_ROOT}"

stage_index() {
  case "$1" in
    A) echo 0 ;;
    B) echo 1 ;;
    C) echo 2 ;;
    D) echo 3 ;;
    *) echo "[v41] unknown stage: $1" >&2; exit 2 ;;
  esac
}

stage_task() {
  local stage="$1"
  local var="TASK_${stage}"
  echo "${!var}"
}

stage_label() {
  local stage="$1"
  local var="LABEL_${stage}"
  echo "${!var}"
}

stage_iters() {
  local stage="$1"
  local var="ITERS_${stage}"
  echo "${!var}"
}

stage_thresholds() {
  case "$1" in
    A) echo "0.70 0.06 1.00 0.00 0.00" ;;
    B) echo "0.50 0.07 0.35 0.00 0.00" ;;
    C) echo "0.15 0.07 0.00 0.10 0.65" ;;
    D) echo "0.20 0.08 1.00 0.00 0.00" ;;
    *) return 1 ;;
  esac
}

common_camera_args=(
  --dual_camera_left_pos 150 75 140
  --dual_camera_right_pos 150 -75 140
  --dual_camera_left_rpy_deg 0 40 -20
  --dual_camera_right_rpy_deg 0 40 20
  --dual_camera_hfov_deg 100
  --camera_far 8
)

train_extra_args=()
if [[ -n "${VIS_SUMMARY}" ]]; then
  train_extra_args+=(--vision_acceptance_summary "${VIS_SUMMARY}")
else
  train_extra_args+=(--allow_multi_env_vision)
  echo "[v41] VIS_SUMMARY not provided; using --allow_multi_env_vision for smoke/V41 runs." >&2
fi

smoke_extra_args=()
if [[ -n "${VIS_SUMMARY}" && "${SMOKE_ENVS}" == "${NUM_ENVS}" ]]; then
  smoke_extra_args+=(--vision_acceptance_summary "${VIS_SUMMARY}")
else
  smoke_extra_args+=(--allow_multi_env_vision)
fi

find_latest_run_dir() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*${run_name}" | sort | tail -n 1
}

find_latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1
}

check_stage_gate() {
  local stage="$1"
  local summary_path="$2"
  local thresholds
  thresholds="$(stage_thresholds "${stage}")"
  python3 - "${stage}" "${summary_path}" ${thresholds} <<'PY'
import json
import math
import sys

stage = sys.argv[1]
summary_path = sys.argv[2]
min_insert = float(sys.argv[3])
max_disp = float(sys.argv[4])
min_side = float(sys.argv[5])
min_lat_drop = float(sys.argv[6])
min_turn = float(sys.argv[7])

summary = json.load(open(summary_path, encoding="utf-8"))
agg = summary.get("aggregate", {})
groups = summary.get("initial_y_groups", {})
episodes = summary.get("episodes", [])
insert_rate = float(agg.get("insert_rate", 0.0))
mean_disp = float(agg.get("mean_max_pallet_disp_xy_m", math.inf))
max_episode_disp = max([float(ep.get("max_pallet_disp_xy_m", 0.0)) for ep in episodes] or [0.0])
ok = True
reasons = []

if insert_rate < min_insert:
    ok = False
    reasons.append(f"insert_rate {insert_rate:.3f} < {min_insert:.3f}")
if mean_disp > max_disp:
    ok = False
    reasons.append(f"mean_max_pallet_disp {mean_disp:.3f} > {max_disp:.3f}")
if stage == "D" and max_episode_disp > max_disp:
    ok = False
    reasons.append(f"max_episode_pallet_disp {max_episode_disp:.3f} > {max_disp:.3f}")
if min_side < 1.0:
    left = float(groups.get("init_y_negative", {}).get("insert_rate", 0.0))
    right = float(groups.get("init_y_nonnegative", {}).get("insert_rate", 0.0))
    if left < min_side or right < min_side:
        ok = False
        reasons.append(f"side insert rates left={left:.3f}, right={right:.3f} < {min_side:.3f}")
if stage == "C":
    drops = []
    turn_ok = []
    for ep in episodes:
        init_lat = abs(float(ep.get("init_signed_lateral_err_m", 0.0)))
        ep_rows = []
        metrics_path = ep.get("metrics_csv")
        if metrics_path:
            import csv
            with open(metrics_path, newline="", encoding="utf-8") as f:
                ep_rows = list(csv.DictReader(f))
        if ep_rows:
            window = ep_rows[: min(80, len(ep_rows))]
            final_lat = abs(float(window[-1].get("signed_lateral_err_m", init_lat)))
            drops.append(init_lat - final_lat)
            init_signed = float(ep.get("init_signed_lateral_err_m", 0.0))
            steers = [float(row.get("env_steer", row.get("steer", 0.0))) for row in window[: min(40, len(window))]]
            mean_steer = sum(steers) / len(steers) if steers else 0.0
            if abs(init_signed) > 1e-6:
                turn_ok.append((init_signed * mean_steer) < 0.0)
    mean_drop = sum(drops) / len(drops) if drops else 0.0
    turn_rate = sum(1 for v in turn_ok if v) / len(turn_ok) if turn_ok else 0.0
    if mean_drop < min_lat_drop:
        ok = False
        reasons.append(f"first80 lateral drop {mean_drop:.3f} < {min_lat_drop:.3f}")
    if turn_rate < min_turn:
        ok = False
        reasons.append(f"turn trend {turn_rate:.3f} < {min_turn:.3f}")

report = {
    "stage": stage,
    "pass": ok,
    "insert_rate": insert_rate,
    "mean_max_pallet_disp_xy_m": mean_disp,
    "max_episode_pallet_disp_xy_m": max_episode_disp,
    "first80_lateral_drop_mean": mean_drop if stage == "C" else None,
    "turn_trend_correct_rate": turn_rate if stage == "C" else None,
    "thresholds": {
        "min_insert": min_insert,
        "max_mean_pallet_disp_xy_m": max_disp,
        "min_side_insert_rate": min_side,
        "min_first80_lateral_drop_m": min_lat_drop,
        "min_turn_trend_correct_rate": min_turn,
    },
    "reasons": reasons,
}
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
}

from_idx="$(stage_index "${RUN_FROM_STAGE}")"
to_idx="$(stage_index "${STOP_AFTER_STAGE}")"
if (( from_idx > to_idx )); then
  echo "[v41] RUN_FROM_STAGE must be <= STOP_AFTER_STAGE" >&2
  exit 2
fi

warm_start_checkpoint="${WARM_START_CHECKPOINT:-}"

for i in "${!STAGES[@]}"; do
  if (( i < from_idx || i > to_idx )); then
    continue
  fi
  stage="${STAGES[$i]}"
  task="$(stage_task "${stage}")"
  label="$(stage_label "${stage}")"
  iters="$(stage_iters "${stage}")"
  stage_root="${OUTPUT_ROOT}/${label}"
  mkdir -p "${stage_root}"

  echo "[v41] stage=${stage} label=${label} task=${task}"

  if [[ "${RUN_SMOKE}" == "1" ]]; then
    smoke_name="${label}_smoke_${SMOKE_ENVS}env${SMOKE_ITERS}_seed${SEED}"
    echo "[v41] smoke ${stage}: ${smoke_name}"
    "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "${task}" \
      --num_envs "${SMOKE_ENVS}" \
      --max_iterations "${SMOKE_ITERS}" \
      --seed "${SEED}" \
      --run_name "${smoke_name}" \
      "${smoke_extra_args[@]}" \
      --headless --enable_cameras --device "${DEVICE}" \
      2>&1 | tee "${stage_root}/smoke.log"
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    run_name="${label}_train_${NUM_ENVS}env${iters}_seed${SEED}"
    train_cmd=(
      "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
      --task "${task}"
      --num_envs "${NUM_ENVS}"
      --max_iterations "${iters}"
      --seed "${SEED}"
      --run_name "${run_name}"
      "${train_extra_args[@]}"
      --headless --enable_cameras --device "${DEVICE}"
    )
    if [[ -n "${warm_start_checkpoint}" ]]; then
      train_cmd+=(--warm_start_checkpoint "${warm_start_checkpoint}")
    fi
    echo "[v41] train ${stage}: ${run_name}"
    printf '%q ' "${train_cmd[@]}" > "${stage_root}/train_command.sh"
    printf '\n' >> "${stage_root}/train_command.sh"
    "${train_cmd[@]}" 2>&1 | tee "${stage_root}/train.log"

    run_dir="$(find_latest_run_dir "${run_name}")"
    if [[ -z "${run_dir}" ]]; then
      echo "[v41] could not locate run dir for ${run_name}" >&2
      exit 3
    fi
    warm_start_checkpoint="$(find_latest_checkpoint "${run_dir}")"
    if [[ -z "${warm_start_checkpoint}" ]]; then
      echo "[v41] no checkpoint found in ${run_dir}" >&2
      exit 4
    fi
    echo "${run_dir}" > "${stage_root}/run_dir.txt"
    echo "${warm_start_checkpoint}" > "${stage_root}/latest_checkpoint.txt"
  else
    if [[ -z "${warm_start_checkpoint}" ]]; then
      for checkpoint_file in "${stage_root}/best_checkpoint.txt" "${stage_root}/latest_checkpoint.txt"; do
        if [[ -f "${checkpoint_file}" ]]; then
          warm_start_checkpoint="$(<"${checkpoint_file}")"
          break
        fi
      done
    fi
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ -z "${warm_start_checkpoint}" || ! -f "${warm_start_checkpoint}" ]]; then
      echo "[v41] eval checkpoint missing for stage ${stage}: ${warm_start_checkpoint}" >&2
      exit 5
    fi
    eval_dir="${stage_root}/eval_no_ref_$(basename "${warm_start_checkpoint}" .pt)_$(date +%Y%m%d_%H%M%S)"
    echo "[v41] eval ${stage}: checkpoint=${warm_start_checkpoint}"
    "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py" \
      --task "${task}" \
      --checkpoint "${warm_start_checkpoint}" \
      --checkpoint_type ppo \
      --output_dir "${eval_dir}" \
      --num_envs 1 \
      --episodes "${EVAL_EPISODES_DEFAULT}" \
      --steps "${EVAL_STEPS}" \
      --record_every "${EVAL_RECORD_EVERY}" \
      --fps "${EVAL_FPS}" \
      --seed "${SEED}" \
      --disable_teacher_reference_reset \
      "${common_camera_args[@]}" \
      --save_raw_camera_frames \
      --save_frame_metadata \
      --device "${DEVICE}" --enable_cameras --headless

    if check_stage_gate "${stage}" "${eval_dir}/summary.json" | tee "${stage_root}/gate_$(basename "${eval_dir}").json"; then
      echo "[v41] stage ${stage} gate passed."
      echo "${warm_start_checkpoint}" > "${stage_root}/best_checkpoint.txt"
      echo "${eval_dir}" > "${stage_root}/best_eval_dir.txt"
    else
      echo "[v41] stage ${stage} gate failed; stopping curriculum." >&2
      if [[ "${CONTINUE_ON_GATE_FAIL}" != "1" ]]; then
        exit 10
      fi
    fi
  fi
done

echo "[v41] done output_root=${OUTPUT_ROOT}"
