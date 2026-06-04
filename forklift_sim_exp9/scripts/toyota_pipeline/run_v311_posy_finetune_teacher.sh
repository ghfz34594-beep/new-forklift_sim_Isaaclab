#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/v311_posy_finetune_teacher_${STAMP}}"

TASK="${TASK:-Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherV311PosYFinetune-v0}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-160}"
RUN_NAME="${RUN_NAME:-v311_posy_finetune_seed${SEED}_${NUM_ENVS}env_${ITERS}iter_${STAMP}}"
WARM_START_CHECKPOINT="${WARM_START_CHECKPOINT:-/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_toyota_geoedge_progress_teacher/2026-05-25_15-20-08_progress_teacher_scratch_curriculum_v311_late_dirty_event/model_399.pt}"
RSL_RL_RESET_ACTION_STD_AFTER_LOAD="${RSL_RL_RESET_ACTION_STD_AFTER_LOAD:-0.12}"
export RSL_RL_RESET_ACTION_STD_AFTER_LOAD
RUN_TRAIN="${RUN_TRAIN:-1}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-}"

FAST_EVAL_NUM_ENVS="${FAST_EVAL_NUM_ENVS:-256}"
FAST_EVAL_ROLLOUTS="${FAST_EVAL_ROLLOUTS:-2}"
FULL_EVAL_NUM_ENVS="${FULL_EVAL_NUM_ENVS:-256}"
FULL_EVAL_ROLLOUTS="${FULL_EVAL_ROLLOUTS:-8}"
EVAL_SEED="${EVAL_SEED:-20260427}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
FULL_CANDIDATE_LIMIT="${FULL_CANDIDATE_LIMIT:-2}"

MIN_AVG_SUCCESS="${MIN_AVG_SUCCESS:-0.95}"
MIN_CHECKPOINT_SUCCESS="${MIN_CHECKPOINT_SUCCESS:-0.95}"
MIN_INIT_Y_POS_SUCCESS="${MIN_INIT_Y_POS_SUCCESS:-0.90}"
MIN_NEAR_LATERAL_SUCCESS="${MIN_NEAR_LATERAL_SUCCESS:-0.85}"
MAX_DIRTY_INSERT="${MAX_DIRTY_INSERT:-0.02}"
MAX_PUSH_NO_INSERT="${MAX_PUSH_NO_INSERT:-0.12}"
MIN_EPISODES_PER_SUMMARY="${MIN_EPISODES_PER_SUMMARY:-2048}"

mkdir -p "${OUTPUT_ROOT}/fast_eval" "${OUTPUT_ROOT}/full_eval"

log() {
  printf '[v311-posy-finetune] %s\n' "$*"
}

write_command() {
  local path="$1"
  shift
  printf '%q ' "$@" > "${path}"
  printf '\n' >> "${path}"
  chmod +x "${path}"
}

find_latest_run_dir() {
  local run_name="$1"
  find "${ISAACLAB_DIR}/logs/rsl_rl" -mindepth 2 -maxdepth 2 -type d -name "*${run_name}" | sort | tail -n 1
}

find_latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1
}

append_checkpoint_if_exists() {
  local -n out_ref="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    out_ref+=("${path}")
  fi
}

dedupe_checkpoints() {
  local -n arr_ref="$1"
  local seen=""
  local deduped=()
  local item
  for item in "${arr_ref[@]}"; do
    case "|${seen}|" in
      *"|${item}|"*) ;;
      *)
        deduped+=("${item}")
        seen="${seen}|${item}"
        ;;
    esac
  done
  arr_ref=("${deduped[@]}")
}

select_full_candidates() {
  python3 - "$FULL_CANDIDATE_LIMIT" "$@" <<'PY'
import json
import sys
from pathlib import Path

limit = int(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
rows = []
for path in paths:
    with path.open() as f:
        data = json.load(f)
    episodes = path.with_name(path.name.replace("_summary.json", "_episodes.csv"))
    score = (
        2.0 * float(data.get("strict_success_rate", 0.0))
        - 1.5 * float(data.get("push_no_insert_rate", 0.0))
        - 2.0 * float(data.get("dirty_insert_rate", 0.0))
        - 0.25 * float(data.get("mean_max_pallet_disp_xy", 0.0))
    )
    rows.append((score, float(data.get("strict_success_rate", 0.0)), str(data["checkpoint"]), str(episodes)))

rows.sort(reverse=True)
for _, _, checkpoint, _ in rows[:limit]:
    print(checkpoint)
PY
}

cat > "${OUTPUT_ROOT}/manifest.json" <<EOF
{
  "created_at": "${STAMP}",
  "project_root": "${PROJECT_ROOT}",
  "isaaclab_dir": "${ISAACLAB_DIR}",
  "task": "${TASK}",
  "run_name": "${RUN_NAME}",
  "warm_start_checkpoint": "${WARM_START_CHECKPOINT}",
  "reset_action_std_after_load": ${RSL_RL_RESET_ACTION_STD_AFTER_LOAD},
  "num_envs": ${NUM_ENVS},
  "iterations": ${ITERS},
  "seed": ${SEED},
  "device": "${DEVICE}",
  "fast_eval_total_episodes": $((FAST_EVAL_NUM_ENVS * FAST_EVAL_ROLLOUTS)),
  "full_eval_total_episodes": $((FULL_EVAL_NUM_ENVS * FULL_EVAL_ROLLOUTS)),
  "eval_seed": ${EVAL_SEED},
  "acceptance": {
    "min_avg_success": ${MIN_AVG_SUCCESS},
    "min_checkpoint_success": ${MIN_CHECKPOINT_SUCCESS},
    "min_init_y_pos_success": ${MIN_INIT_Y_POS_SUCCESS},
    "min_near_lateral_success": ${MIN_NEAR_LATERAL_SUCCESS},
    "max_dirty_insert": ${MAX_DIRTY_INSERT},
    "max_push_no_insert": ${MAX_PUSH_NO_INSERT},
    "min_episodes_per_summary": ${MIN_EPISODES_PER_SUMMARY}
  },
  "visual_pipeline": "blocked_until_teacher_acceptance_passes"
}
EOF

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_cmd=(
    "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${ITERS}"
    --seed "${SEED}"
    --run_name "${RUN_NAME}"
    --warm_start_checkpoint "${WARM_START_CHECKPOINT}"
    --headless
    --device "${DEVICE}"
  )
  write_command "${OUTPUT_ROOT}/train_command.sh" "${train_cmd[@]}"
  log "starting pos-y finetune: task=${TASK} run_name=${RUN_NAME}"
  "${train_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/train.log"

  TEACHER_RUN_DIR="$(find_latest_run_dir "${RUN_NAME}")"
  if [[ -z "${TEACHER_RUN_DIR}" ]]; then
    log "could not locate teacher run dir for run_name=${RUN_NAME}"
    exit 3
  fi
else
  if [[ -z "${TEACHER_RUN_DIR}" ]]; then
    log "RUN_TRAIN=0 requires TEACHER_RUN_DIR"
    exit 3
  fi
fi

echo "${TEACHER_RUN_DIR}" > "${OUTPUT_ROOT}/run_dir.txt"
log "teacher run dir: ${TEACHER_RUN_DIR}"

if [[ -f "${OUTPUT_ROOT}/train.log" ]]; then
  python3 "${PIPELINE_DIR}/summarize_training_log.py" \
    --log "${OUTPUT_ROOT}/train.log" \
    --output "${OUTPUT_ROOT}/training_summary.json" || true
fi

checkpoints=()
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_20.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_40.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_60.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_80.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_100.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_120.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_140.pt"
append_checkpoint_if_exists checkpoints "${TEACHER_RUN_DIR}/model_159.pt"
latest_checkpoint="$(find_latest_checkpoint "${TEACHER_RUN_DIR}")"
append_checkpoint_if_exists checkpoints "${latest_checkpoint}"
dedupe_checkpoints checkpoints

if (( ${#checkpoints[@]} == 0 )); then
  log "no checkpoints found in ${TEACHER_RUN_DIR}"
  exit 4
fi

fast_summary_paths=()
for checkpoint in "${checkpoints[@]}"; do
  label="posy_fast_$(basename "${checkpoint}" .pt)"
  eval_cmd=(
    "${RUN_WRAPPER}" -p "${PROJECT_ROOT}/scripts/eval_geoedge_checkpoint.py"
    --task "${TASK}"
    --checkpoint "${checkpoint}"
    --label "${label}"
    --num_envs "${FAST_EVAL_NUM_ENVS}"
    --rollouts "${FAST_EVAL_ROLLOUTS}"
    --seed "${EVAL_SEED}"
    --output_dir "${OUTPUT_ROOT}/fast_eval"
    --stage1_eval
    --reset_profile "${EVAL_PROFILE}"
    --headless
    --device "${DEVICE}"
  )
  write_command "${OUTPUT_ROOT}/fast_eval_${label}_command.sh" "${eval_cmd[@]}"
  log "fast evaluating checkpoint=${checkpoint}"
  "${eval_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/fast_eval/${label}.log"

  summary_path="${OUTPUT_ROOT}/fast_eval/${label}_summary.json"
  episode_path="${OUTPUT_ROOT}/fast_eval/${label}_episodes.csv"
  bucket_path="${OUTPUT_ROOT}/fast_eval/${label}_bucket_summary.md"
  fast_summary_paths+=("${summary_path}")

  python3 "${PROJECT_ROOT}/scripts/summarize_geoedge_eval.py" \
    --episodes-csv "${episode_path}" \
    --output "${bucket_path}" >/dev/null
done

mapfile -t full_candidates < <(select_full_candidates "${fast_summary_paths[@]}")
if (( ${#full_candidates[@]} == 0 )); then
  log "no full-eval candidates selected"
  exit 4
fi

summary_paths=()
episode_paths=()
for checkpoint in "${full_candidates[@]}"; do
  label="posy_full_$(basename "${checkpoint}" .pt)"
  eval_cmd=(
    "${RUN_WRAPPER}" -p "${PROJECT_ROOT}/scripts/eval_geoedge_checkpoint.py"
    --task "${TASK}"
    --checkpoint "${checkpoint}"
    --label "${label}"
    --num_envs "${FULL_EVAL_NUM_ENVS}"
    --rollouts "${FULL_EVAL_ROLLOUTS}"
    --seed "${EVAL_SEED}"
    --output_dir "${OUTPUT_ROOT}/full_eval"
    --stage1_eval
    --reset_profile "${EVAL_PROFILE}"
    --headless
    --device "${DEVICE}"
  )
  write_command "${OUTPUT_ROOT}/full_eval_${label}_command.sh" "${eval_cmd[@]}"
  log "full evaluating candidate=${checkpoint}"
  "${eval_cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/full_eval/${label}.log"

  summary_path="${OUTPUT_ROOT}/full_eval/${label}_summary.json"
  episode_path="${OUTPUT_ROOT}/full_eval/${label}_episodes.csv"
  bucket_path="${OUTPUT_ROOT}/full_eval/${label}_bucket_summary.md"
  summary_paths+=("${summary_path}")
  episode_paths+=("${episode_path}")

  python3 "${PROJECT_ROOT}/scripts/summarize_geoedge_eval.py" \
    --episodes-csv "${episode_path}" \
    --output "${bucket_path}" >/dev/null
done

acceptance_cmd=(
  python3 "${PROJECT_ROOT}/scripts/check_teacher_acceptance.py"
  --min-avg-success "${MIN_AVG_SUCCESS}"
  --min-checkpoint-success "${MIN_CHECKPOINT_SUCCESS}"
  --min-init-y-pos-success "${MIN_INIT_Y_POS_SUCCESS}"
  --min-near-lateral-success "${MIN_NEAR_LATERAL_SUCCESS}"
  --max-dirty-insert "${MAX_DIRTY_INSERT}"
  --max-push-no-insert "${MAX_PUSH_NO_INSERT}"
  --min-episodes-per-summary "${MIN_EPISODES_PER_SUMMARY}"
  --output-json "${OUTPUT_ROOT}/teacher_acceptance.json"
  --output-md "${OUTPUT_ROOT}/TEACHER_ACCEPTANCE.md"
)
for episode_path in "${episode_paths[@]}"; do
  acceptance_cmd+=(--episodes-csv "${episode_path}")
done
acceptance_cmd+=("${summary_paths[@]}")

write_command "${OUTPUT_ROOT}/check_acceptance_command.sh" "${acceptance_cmd[@]}"
log "checking teacher acceptance gates"
if "${acceptance_cmd[@]}"; then
  log "teacher acceptance PASS; visual pipeline may be started by a separate command"
else
  log "teacher acceptance FAIL; visual pipeline remains blocked"
  exit 5
fi

log "done: ${OUTPUT_ROOT}"
