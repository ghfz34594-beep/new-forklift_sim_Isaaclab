#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/clean_teacher_visual_pipeline_${STAMP}}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-20260529}"

RUN_TELEOP="${RUN_TELEOP:-0}"
CONFIRM_PHYSICS_OK="${CONFIRM_PHYSICS_OK:-0}"
RUN_TEACHER="${RUN_TEACHER:-1}"
CONFIRM_TEACHER_OK="${CONFIRM_TEACHER_OK:-0}"
AUTO_CONTINUE_AFTER_TEACHER="${AUTO_CONTINUE_AFTER_TEACHER:-0}"
RUN_SHAPE_AUDIT="${RUN_SHAPE_AUDIT:-1}"
RUN_VISUAL_TRAIN="${RUN_VISUAL_TRAIN:-1}"
RUN_VISUAL_EVAL="${RUN_VISUAL_EVAL:-1}"

TELEOP_TASK="${TELEOP_TASK:-Isaac-Forklift-PalletApproach-ToyotaDualCamera-v0}"

TEACHER_TASK="${TEACHER_TASK:-Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherRecoveryFix-v0}"
TEACHER_NUM_ENVS="${TEACHER_NUM_ENVS:-1024}"
TEACHER_ITERS="${TEACHER_ITERS:-1200}"
TEACHER_SEED="${TEACHER_SEED:-42}"
TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-clean_progress_teacher_${STAMP}}"
TEACHER_EVAL_NUM_ENVS="${TEACHER_EVAL_NUM_ENVS:-256}"
TEACHER_EVAL_ROLLOUTS="${TEACHER_EVAL_ROLLOUTS:-8}"
TEACHER_EVAL_SEED="${TEACHER_EVAL_SEED:-20260427}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-}"
TEACHER_ACCEPTANCE_JSON="${TEACHER_ACCEPTANCE_JSON:-}"
TEACHER_ACCEPTANCE_MIN_AVG_SUCCESS="${TEACHER_ACCEPTANCE_MIN_AVG_SUCCESS:-0.95}"
TEACHER_ACCEPTANCE_MIN_CHECKPOINT_SUCCESS="${TEACHER_ACCEPTANCE_MIN_CHECKPOINT_SUCCESS:-0.95}"
TEACHER_ACCEPTANCE_MIN_INIT_Y_POS_SUCCESS="${TEACHER_ACCEPTANCE_MIN_INIT_Y_POS_SUCCESS:-0.90}"
TEACHER_ACCEPTANCE_MIN_NEAR_LATERAL_SUCCESS="${TEACHER_ACCEPTANCE_MIN_NEAR_LATERAL_SUCCESS:-0.85}"
TEACHER_ACCEPTANCE_MAX_DIRTY_INSERT="${TEACHER_ACCEPTANCE_MAX_DIRTY_INSERT:-0.02}"
TEACHER_ACCEPTANCE_MAX_PUSH_NO_INSERT="${TEACHER_ACCEPTANCE_MAX_PUSH_NO_INSERT:-0.12}"
TEACHER_ACCEPTANCE_MIN_EPISODES="${TEACHER_ACCEPTANCE_MIN_EPISODES:-2048}"

VISUAL_TASK="${VISUAL_TASK:-Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV40Direct-v0}"
VISUAL_NUM_ENVS="${VISUAL_NUM_ENVS:-64}"
VISUAL_ITERS="${VISUAL_ITERS:-501}"
VISUAL_RUN_NAME="${VISUAL_RUN_NAME:-clean_visual_${VISUAL_ITERS}_${STAMP}}"
VISUAL_EVAL_EPISODES="${VISUAL_EVAL_EPISODES:-12}"
VISUAL_EVAL_STEPS="${VISUAL_EVAL_STEPS:-720}"
VISUAL_EVAL_RECORD_EVERY="${VISUAL_EVAL_RECORD_EVERY:-3}"

AUDIT_NUM_ENVS="${AUDIT_NUM_ENVS:-2}"

mkdir -p "${OUTPUT_ROOT}"

log() {
  printf '[clean-pipeline] %s\n' "$*"
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

write_manifest() {
  cat > "${OUTPUT_ROOT}/manifest.json" <<EOF
{
  "created_at": "${STAMP}",
  "project_root": "${PROJECT_ROOT}",
  "isaaclab_dir": "${ISAACLAB_DIR}",
  "old_v41_policy": "historical_reference_only_no_warm_start",
  "teacher_task": "${TEACHER_TASK}",
  "teacher_observation": "21D = 12D edge_obs + 9D proprio",
  "teacher_acceptance_required": true,
  "teacher_acceptance_json": "${TEACHER_ACCEPTANCE_JSON}",
  "visual_task": "${VISUAL_TASK}",
  "visual_policy_input": "left RGB + right RGB + 5D Toyota proprio",
  "visual_iterations": ${VISUAL_ITERS},
  "device": "${DEVICE}",
  "seed": ${SEED}
}
EOF
}

write_manifest

teleop_cmd=(
  "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/teleop_dual_camera.py"
  --task "${TELEOP_TASK}"
  --num_envs 1
  --device "${DEVICE}"
  --enable_cameras
)
write_command "${OUTPUT_ROOT}/manual_physics_check_command.sh" "${teleop_cmd[@]}"

if [[ "${RUN_TELEOP}" == "1" ]]; then
  log "launching keyboard teleop for physical reachability check"
  "${teleop_cmd[@]}"
  log "teleop exited; rerun with CONFIRM_PHYSICS_OK=1 after manually confirming insertion/reachability"
  exit 2
fi

if [[ "${CONFIRM_PHYSICS_OK}" != "1" ]]; then
  log "physical reachability is not confirmed; not starting teacher/visual training"
  log "run ${OUTPUT_ROOT}/manual_physics_check_command.sh, then rerun with CONFIRM_PHYSICS_OK=1"
  exit 2
fi

teacher_root="${OUTPUT_ROOT}/teacher"
teacher_summary_paths=()
teacher_episode_paths=()

if [[ "${RUN_TEACHER}" == "1" ]]; then
  mkdir -p "${teacher_root}/eval"
  teacher_train_cmd=(
    "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
    --task "${TEACHER_TASK}"
    --num_envs "${TEACHER_NUM_ENVS}"
    --max_iterations "${TEACHER_ITERS}"
    --seed "${TEACHER_SEED}"
    --run_name "${TEACHER_RUN_NAME}"
    --headless
    --device "${DEVICE}"
  )
  write_command "${teacher_root}/train_command.sh" "${teacher_train_cmd[@]}"
  log "starting clean privileged teacher training: run_name=${TEACHER_RUN_NAME}"
  "${teacher_train_cmd[@]}" 2>&1 | tee "${teacher_root}/train.log"

  TEACHER_RUN_DIR="$(find_latest_run_dir "${TEACHER_RUN_NAME}")"
  if [[ -z "${TEACHER_RUN_DIR}" ]]; then
    log "could not locate teacher run dir for ${TEACHER_RUN_NAME}"
    exit 3
  fi
  echo "${TEACHER_RUN_DIR}" > "${teacher_root}/run_dir.txt"
  python3 "${PIPELINE_DIR}/summarize_training_log.py" \
    --log "${teacher_root}/train.log" \
    --output "${teacher_root}/training_summary.json"

  teacher_checkpoints=()
  append_checkpoint_if_exists teacher_checkpoints "${TEACHER_RUN_DIR}/model_500.pt"
  append_checkpoint_if_exists teacher_checkpoints "${TEACHER_RUN_DIR}/model_1000.pt"
  teacher_latest="$(find_latest_checkpoint "${TEACHER_RUN_DIR}")"
  append_checkpoint_if_exists teacher_checkpoints "${teacher_latest}"
  dedupe_checkpoints teacher_checkpoints

  if (( ${#teacher_checkpoints[@]} == 0 )); then
    log "no teacher checkpoints found in ${TEACHER_RUN_DIR}"
    exit 4
  fi

  for checkpoint in "${teacher_checkpoints[@]}"; do
    label="teacher_$(basename "${checkpoint}" .pt)"
    eval_cmd=(
      "${RUN_WRAPPER}" -p "${PROJECT_ROOT}/scripts/eval_geoedge_checkpoint.py"
      --task "${TEACHER_TASK}"
      --checkpoint "${checkpoint}"
      --label "${label}"
      --num_envs "${TEACHER_EVAL_NUM_ENVS}"
      --rollouts "${TEACHER_EVAL_ROLLOUTS}"
      --seed "${TEACHER_EVAL_SEED}"
      --output_dir "${teacher_root}/eval"
      --stage1_eval
      --reset_profile full
      --headless
      --device "${DEVICE}"
    )
    write_command "${teacher_root}/eval_${label}_command.sh" "${eval_cmd[@]}"
    log "evaluating teacher checkpoint=${checkpoint}"
    "${eval_cmd[@]}" 2>&1 | tee "${teacher_root}/eval/${label}.log"

    summary_path="${teacher_root}/eval/${label}_summary.json"
    episode_path="${teacher_root}/eval/${label}_episodes.csv"
    bucket_path="${teacher_root}/eval/${label}_bucket_summary.md"
    teacher_summary_paths+=("${summary_path}")
    teacher_episode_paths+=("${episode_path}")
    python3 "${PROJECT_ROOT}/scripts/summarize_geoedge_eval.py" \
      --episodes-csv "${episode_path}" \
      --output "${bucket_path}" >/dev/null
  done
else
  if [[ -z "${TEACHER_RUN_DIR}" ]]; then
    log "RUN_TEACHER=0 requires TEACHER_RUN_DIR to point at an already trained clean teacher run"
    exit 3
  fi
fi

if [[ -n "${TEACHER_ACCEPTANCE_JSON}" ]]; then
  log "checking provided teacher acceptance report: ${TEACHER_ACCEPTANCE_JSON}"
  python3 - "${TEACHER_ACCEPTANCE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as f:
    report = json.load(f)
if report.get("passed") is not True:
    raise SystemExit(f"teacher acceptance report is not passing: {path}")
PY
else
  if (( ${#teacher_summary_paths[@]} == 0 )); then
    if [[ -d "${teacher_root}/eval" ]]; then
      mapfile -t teacher_summary_paths < <(find "${teacher_root}/eval" -maxdepth 1 -type f -name '*_summary.json' | sort)
      mapfile -t teacher_episode_paths < <(find "${teacher_root}/eval" -maxdepth 1 -type f -name '*_episodes.csv' | sort)
    fi
  fi
  if (( ${#teacher_summary_paths[@]} == 0 || ${#teacher_episode_paths[@]} == 0 )); then
    log "teacher acceptance is required but no eval summaries/episodes were found"
    log "run teacher eval first, or provide TEACHER_ACCEPTANCE_JSON pointing at a passing report"
    exit 5
  fi

  acceptance_cmd=(
    python3 "${PROJECT_ROOT}/scripts/check_teacher_acceptance.py"
    --min-avg-success "${TEACHER_ACCEPTANCE_MIN_AVG_SUCCESS}"
    --min-checkpoint-success "${TEACHER_ACCEPTANCE_MIN_CHECKPOINT_SUCCESS}"
    --min-init-y-pos-success "${TEACHER_ACCEPTANCE_MIN_INIT_Y_POS_SUCCESS}"
    --min-near-lateral-success "${TEACHER_ACCEPTANCE_MIN_NEAR_LATERAL_SUCCESS}"
    --max-dirty-insert "${TEACHER_ACCEPTANCE_MAX_DIRTY_INSERT}"
    --max-push-no-insert "${TEACHER_ACCEPTANCE_MAX_PUSH_NO_INSERT}"
    --min-episodes-per-summary "${TEACHER_ACCEPTANCE_MIN_EPISODES}"
    --output-json "${teacher_root}/teacher_acceptance.json"
    --output-md "${teacher_root}/TEACHER_ACCEPTANCE.md"
  )
  for episode_path in "${teacher_episode_paths[@]}"; do
    acceptance_cmd+=(--episodes-csv "${episode_path}")
  done
  acceptance_cmd+=("${teacher_summary_paths[@]}")
  write_command "${teacher_root}/check_acceptance_command.sh" "${acceptance_cmd[@]}"
  log "checking teacher acceptance before visual pipeline"
  if ! "${acceptance_cmd[@]}"; then
    log "teacher acceptance failed; visual pipeline remains blocked"
    exit 5
  fi
fi

if [[ "${CONFIRM_TEACHER_OK}" != "1" && "${AUTO_CONTINUE_AFTER_TEACHER}" != "1" ]]; then
  log "teacher training/eval is complete; inspect teacher success/loss summaries before visual training"
  log "rerun with CONFIRM_PHYSICS_OK=1 CONFIRM_TEACHER_OK=1 RUN_TEACHER=0 TEACHER_RUN_DIR=${TEACHER_RUN_DIR}"
  exit 5
fi

if [[ "${RUN_SHAPE_AUDIT}" == "1" ]]; then
  audit_root="${OUTPUT_ROOT}/visual_shape_audit"
  mkdir -p "${audit_root}"
  audit_cmd=(
    "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/audit_visual_actor_shapes.py"
    --task "${VISUAL_TASK}"
    --num_envs "${AUDIT_NUM_ENVS}"
    --seed "${SEED}"
    --output "${audit_root}/summary.json"
    --headless
    --enable_cameras
    --device "${DEVICE}"
  )
  write_command "${audit_root}/audit_command.sh" "${audit_cmd[@]}"
  log "running visual tensor shape audit before long training"
  "${audit_cmd[@]}" 2>&1 | tee "${audit_root}/audit.log"
fi

if [[ "${RUN_VISUAL_TRAIN}" == "1" ]]; then
  visual_root="${OUTPUT_ROOT}/visual_train"
  mkdir -p "${visual_root}/eval"
  visual_train_cmd=(
    "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py
    --task "${VISUAL_TASK}"
    --num_envs "${VISUAL_NUM_ENVS}"
    --max_iterations "${VISUAL_ITERS}"
    --seed "${SEED}"
    --run_name "${VISUAL_RUN_NAME}"
    --allow_multi_env_vision
    --headless
    --enable_cameras
    --device "${DEVICE}"
  )
  write_command "${visual_root}/train_command.sh" "${visual_train_cmd[@]}"
  log "starting clean visual training from scratch: run_name=${VISUAL_RUN_NAME}"
  "${visual_train_cmd[@]}" 2>&1 | tee "${visual_root}/train.log"

  visual_run_dir="$(find_latest_run_dir "${VISUAL_RUN_NAME}")"
  if [[ -z "${visual_run_dir}" ]]; then
    log "could not locate visual run dir for ${VISUAL_RUN_NAME}"
    exit 6
  fi
  echo "${visual_run_dir}" > "${visual_root}/run_dir.txt"
  visual_checkpoint="$(find_latest_checkpoint "${visual_run_dir}")"
  if [[ -z "${visual_checkpoint}" ]]; then
    log "no visual checkpoint found in ${visual_run_dir}"
    exit 7
  fi
  echo "${visual_checkpoint}" > "${visual_root}/latest_checkpoint.txt"
  python3 "${PIPELINE_DIR}/summarize_training_log.py" \
    --log "${visual_root}/train.log" \
    --output "${visual_root}/training_summary.json"

  if [[ "${RUN_VISUAL_EVAL}" == "1" ]]; then
    visual_eval_dir="${visual_root}/eval/$(basename "${visual_checkpoint}" .pt)_$(date +%Y%m%d_%H%M%S)"
    visual_eval_cmd=(
      "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py"
      --task "${VISUAL_TASK}"
      --checkpoint "${visual_checkpoint}"
      --checkpoint_type ppo
      --output_dir "${visual_eval_dir}"
      --num_envs 1
      --episodes "${VISUAL_EVAL_EPISODES}"
      --steps "${VISUAL_EVAL_STEPS}"
      --record_every "${VISUAL_EVAL_RECORD_EVERY}"
      --fps 30
      --seed "${SEED}"
      --disable_teacher_reference_reset
      --dual_camera_left_pos 150 75 140
      --dual_camera_right_pos 150 -75 140
      --dual_camera_left_rpy_deg 0 40 -20
      --dual_camera_right_rpy_deg 0 40 20
      --dual_camera_hfov_deg 100
      --camera_far 8
      --save_raw_camera_frames
      --save_frame_metadata
      --headless
      --enable_cameras
      --device "${DEVICE}"
    )
    write_command "${visual_root}/eval_command.sh" "${visual_eval_cmd[@]}"
    log "evaluating clean visual checkpoint=${visual_checkpoint}"
    "${visual_eval_cmd[@]}" 2>&1 | tee "${visual_root}/eval.log"
  fi
fi

log "done: ${OUTPUT_ROOT}"
