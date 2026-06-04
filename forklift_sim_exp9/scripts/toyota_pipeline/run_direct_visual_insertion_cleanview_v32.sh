#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"
INSTALL_PATCH="${INSTALL_PATCH:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASK="${TASK:-Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV32-v0}"
RUN_LABEL="${RUN_LABEL:-direct_visual_insertion_cleanview_v32_trainable_steer_sign}"
RUN_NAME_BASE="${RUN_NAME_BASE:-${RUN_LABEL}_${STAMP}}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-${RUN_NAME_BASE}_smoke}"
MAIN_RUN_NAME="${MAIN_RUN_NAME:-${RUN_NAME_BASE}_main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/student_distill/${RUN_NAME_BASE}}"

DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-16}"
ENV_SPACING="${ENV_SPACING:-20}"
CAMERA_FAR="${CAMERA_FAR:-8}"
HFOV="${HFOV:-100}"
SEED="${SEED:-20260527}"
ISO_COVERAGE_MODE="${ISO_COVERAGE_MODE:-stratified}"
ISO_COVERAGE_COUNT="${ISO_COVERAGE_COUNT:-16}"
ISO_MOSAIC_COVERAGE_MODE="${ISO_MOSAIC_COVERAGE_MODE:-checked}"
ISO_MOSAIC_CHUNK_SIZE="${ISO_MOSAIC_CHUNK_SIZE:-128}"
ISO_REQUIRE_FULL_MOSAIC="${ISO_REQUIRE_FULL_MOSAIC:-0}"
SMOKE_ITERS="${SMOKE_ITERS:-20}"
MAX_ITER_MAIN="${MAX_ITER_MAIN:-401}"
RUN_MAIN="${RUN_MAIN:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_STEPS="${EVAL_STEPS:-720}"

RUN_PROBE="${RUN_PROBE:-1}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"
PROBE_STEPS="${PROBE_STEPS:-180}"
PROBE_EPOCHS="${PROBE_EPOCHS:-6}"
PROBE_MIN_ACCURACY="${PROBE_MIN_ACCURACY:-0.78}"

VIS_SUMMARY="${VIS_SUMMARY:-}"
VIS_DIR="${OUTPUT_ROOT}/visual_acceptance_cleanview45"
PROBE_DATASET_DIR="${PROBE_DATASET_DIR:-${OUTPUT_ROOT}/signed_geometry_probe_dataset}"
PROBE_DIR="${OUTPUT_ROOT}/signed_geometry_probe"
SMOKE_LOG="${OUTPUT_ROOT}/smoke_command.log"
MAIN_LOG="${OUTPUT_ROOT}/main_command.log"

LEFT_POS=(${LEFT_POS:-150 75 140})
RIGHT_POS=(${RIGHT_POS:-150 -75 140})
LEFT_RPY=(${LEFT_RPY:-0 40 -20})
RIGHT_RPY=(${RIGHT_RPY:-0 40 20})

mkdir -p "${OUTPUT_ROOT}"

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

COMMON_CAMERA_ARGS=(
  --env_spacing "${ENV_SPACING}"
  --camera_far "${CAMERA_FAR}"
  --dual_camera_hfov_deg "${HFOV}"
  --dual_camera_left_pos "${LEFT_POS[@]}"
  --dual_camera_right_pos "${RIGHT_POS[@]}"
  --dual_camera_left_rpy_deg "${LEFT_RPY[@]}"
  --dual_camera_right_rpy_deg "${RIGHT_RPY[@]}"
  --no_vision_room
)

echo "[direct-visual-v32] task=${TASK}"
echo "[direct-visual-v32] output_root=${OUTPUT_ROOT}"
echo "[direct-visual-v32] main_run_name=${MAIN_RUN_NAME}"

if [[ "${INSTALL_PATCH}" == "1" ]]; then
  bash "${PROJECT_ROOT}/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "${ISAACLAB_DIR}"
fi

if [[ -z "${VIS_SUMMARY}" || ! -f "${VIS_SUMMARY}" ]]; then
  echo "[direct-visual-v32] VIS_SUMMARY not provided; running visual acceptance for ${TASK}."
  python3 "${PIPELINE_DIR}/validate_room60_visual_isolation.py" \
    --output_dir "${VIS_DIR}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --steps 120 \
    --record_every 2 \
    --fps 20 \
    "${COMMON_CAMERA_ARGS[@]}" \
    --sentinel_room_probes_all_envs \
    --record_mosaic \
    --mosaic_max_envs "${NUM_ENVS}" \
    --mosaic_cols 4 \
    --coverage_mode "${ISO_COVERAGE_MODE}" \
    --coverage_count "${ISO_COVERAGE_COUNT}" \
    --mosaic_coverage_mode "${ISO_MOSAIC_COVERAGE_MODE}" \
    --mosaic_chunk_size "${ISO_MOSAIC_CHUNK_SIZE}" \
    $([[ "${ISO_REQUIRE_FULL_MOSAIC}" == "1" ]] && printf '%s' "--require_full_mosaic_coverage") \
    --overwrite
  VIS_SUMMARY="${VIS_DIR}/visual_isolation_summary.json"
else
  echo "[direct-visual-v32] reusing visual acceptance: ${VIS_SUMMARY}"
fi

if [[ "${RUN_PROBE}" == "1" ]]; then
  echo "[direct-visual-v32] collecting signed geometry probe dataset: ${PROBE_DATASET_DIR}"
  "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/collect_cleanview_signed_geometry_probe_dataset.py" \
    --task "${TASK}" \
    --output_dir "${PROBE_DATASET_DIR}" \
    --num_envs "${NUM_ENVS}" \
    --samples "${PROBE_SAMPLES}" \
    --steps "${PROBE_STEPS}" \
    --record_every 1 \
    --reset_every 4 \
    --seed "${SEED}" \
    "${COMMON_CAMERA_ARGS[@]}" \
    --device "${DEVICE}" --enable_cameras --headless

  "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/train_cleanview_signed_geometry_probe.py" \
    --dataset_dir "${PROBE_DATASET_DIR}" \
    --output_dir "${PROBE_DIR}" \
    --epochs "${PROBE_EPOCHS}" \
    --min_accuracy "${PROBE_MIN_ACCURACY}" \
    --device "${DEVICE}" \
    --seed "${SEED}"

  python3 - "${PROBE_DIR}/signed_geometry_probe_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("pass") is not True:
    raise SystemExit(f"signed geometry probe did not pass: {json.dumps(summary, sort_keys=True)}")
PY
else
  echo "[direct-visual-v32] signed geometry probe skipped; RUN_PROBE=1 to require it."
fi

echo "[direct-visual-v32] starting smoke: ${SMOKE_RUN_NAME}"
{
  printf '%q ' "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${SMOKE_ITERS}" \
    --seed "${SEED}" \
    --run_name "${SMOKE_RUN_NAME}" \
    --vision_acceptance_summary "${VIS_SUMMARY}" \
    --headless --enable_cameras --device "${DEVICE}"
  echo
} > "${SMOKE_LOG}"
"${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${SMOKE_ITERS}" \
  --seed "${SEED}" \
  --run_name "${SMOKE_RUN_NAME}" \
  --vision_acceptance_summary "${VIS_SUMMARY}" \
  --headless --enable_cameras --device "${DEVICE}"

if [[ "${RUN_MAIN}" != "1" ]]; then
  echo "[direct-visual-v32] main run skipped; RUN_MAIN=1 to enable."
  echo "[direct-visual-v32] visual_acceptance=${VIS_SUMMARY}"
  echo "[direct-visual-v32] output_root=${OUTPUT_ROOT}"
  exit 0
fi

echo "[direct-visual-v32] starting main run: ${MAIN_RUN_NAME}"
{
  printf '%q ' "${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITER_MAIN}" \
    --seed "${SEED}" \
    --run_name "${MAIN_RUN_NAME}" \
    --vision_acceptance_summary "${VIS_SUMMARY}" \
    --headless --enable_cameras --device "${DEVICE}"
  echo
} > "${MAIN_LOG}"
"${RUN_WRAPPER}" -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITER_MAIN}" \
  --seed "${SEED}" \
  --run_name "${MAIN_RUN_NAME}" \
  --vision_acceptance_summary "${VIS_SUMMARY}" \
  --headless --enable_cameras --device "${DEVICE}"

LOG_ROOT="${ISAACLAB_DIR}/logs/rsl_rl/direct_visual_insertion_cleanview"
MAIN_RUN_DIR="$(find "${LOG_ROOT}" -maxdepth 1 -type d -name "*${MAIN_RUN_NAME}*" | sort | tail -n 1)"
if [[ -z "${MAIN_RUN_DIR}" ]]; then
  echo "[direct-visual-v32] could not find main run dir under ${LOG_ROOT}" >&2
  exit 3
fi

CKPT_FINAL="$(find "${MAIN_RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [[ -z "${CKPT_FINAL}" ]]; then
  echo "[direct-visual-v32] no model checkpoint found in ${MAIN_RUN_DIR}" >&2
  exit 4
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  EVAL_DIR="${OUTPUT_ROOT}/eval_$(basename "${CKPT_FINAL}" .pt)"
  echo "[direct-visual-v32] evaluating final checkpoint: ${CKPT_FINAL}"
  "${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py" \
    --task "${TASK}" \
    --checkpoint "${CKPT_FINAL}" \
    --checkpoint_type ppo \
    --output_dir "${EVAL_DIR}" \
    --episodes "${EVAL_EPISODES}" \
    --num_envs 16 \
    --steps "${EVAL_STEPS}" \
    --record_every 6 \
    --fps 20 \
    --seed "${SEED}" \
    --baseline_mean_max_insert_depth_m 0.3296 \
    "${COMMON_CAMERA_ARGS[@]}" \
    --device "${DEVICE}" --enable_cameras --headless
else
  echo "[direct-visual-v32] eval skipped; RUN_EVAL=1 to enable."
fi

echo "[direct-visual-v32] done"
echo "[direct-visual-v32] visual_acceptance=${VIS_SUMMARY}"
echo "[direct-visual-v32] run_dir=${MAIN_RUN_DIR}"
echo "[direct-visual-v32] output_root=${OUTPUT_ROOT}"
