#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/data/jianshi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_toyota_geoedge_progress_teacher/2026-05-25_15-20-08_progress_teacher_scratch_curriculum_v311_late_dirty_event/model_399.pt}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-student_cleanview45_16env_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/student_distill/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/toyota_teacher_distill/${RUN_NAME}}"
LEGACY_DATASET_BASENAME="progress_v311_multi_env_clean_v1"

DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-16}"
ENV_SPACING="${ENV_SPACING:-20}"
CAMERA_FAR="${CAMERA_FAR:-8}"
SEED="${SEED:-20260526}"
ISO_COVERAGE_MODE="${ISO_COVERAGE_MODE:-stratified}"
ISO_COVERAGE_COUNT="${ISO_COVERAGE_COUNT:-16}"
ISO_MOSAIC_COVERAGE_MODE="${ISO_MOSAIC_COVERAGE_MODE:-checked}"
ISO_MOSAIC_CHUNK_SIZE="${ISO_MOSAIC_CHUNK_SIZE:-128}"
ISO_REQUIRE_FULL_MOSAIC="${ISO_REQUIRE_FULL_MOSAIC:-0}"

ROOM60_TASK="${ROOM60_TASK:-Isaac-Forklift-PalletApproach-ToyotaDualCameraRoom60-v0}"
CLEANVIEW_TASK="${CLEANVIEW_TASK:-Isaac-Forklift-PalletApproach-ToyotaProgressStudentCleanView-v0}"
COLLECT_TASK="${COLLECT_TASK:-Isaac-Forklift-PalletApproach-ToyotaGeoEdgeProgressTeacherCollectCleanView-v0}"
HFOV="${HFOV:-100}"
LEFT_POS=(${LEFT_POS:-150 75 140})
RIGHT_POS=(${RIGHT_POS:-150 -75 140})
LEFT_RPY=(${LEFT_RPY:-0 40 -20})
RIGHT_RPY=(${RIGHT_RPY:-0 40 20})

TARGET_CLEAN_EPISODES="${TARGET_CLEAN_EPISODES:-160}"
MAX_ATTEMPTED_EPISODES="${MAX_ATTEMPTED_EPISODES:-220}"
MAX_STEPS="${MAX_STEPS:-900}"
IMAGE_EVERY="${IMAGE_EVERY:-1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MAX_FRAC_ABS_DRIVE_GT_095="${MAX_FRAC_ABS_DRIVE_GT_095:-0.05}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"

ROOM60_VIS_DIR="${OUTPUT_ROOT}/visual_acceptance_room60_pressure"
VIS_DIR="${OUTPUT_ROOT}/visual_acceptance_cleanview45"
BC_OUTPUT="${OUTPUT_ROOT}/approach_student_bc.pt"
EVAL_DIR="${OUTPUT_ROOT}/visual_eval_bc"

if [[ "$(basename "${DATASET_DIR}")" == "${LEGACY_DATASET_BASENAME}" ]]; then
  echo "[pipeline] refusing to write formal CleanView45 data into legacy dataset: ${DATASET_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${DATASET_DIR}"

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
ROOM60_CAMERA_ARGS=(
  --env_spacing "${ENV_SPACING}"
  --camera_far "${CAMERA_FAR}"
  --dual_camera_hfov_deg 100
  --dual_camera_left_pos 150 75 140
  --dual_camera_right_pos 150 -75 140
  --dual_camera_left_rpy_deg 0 40 -20
  --dual_camera_right_rpy_deg 0 40 20
  --no_vision_room
)

echo "[pipeline] output_root=${OUTPUT_ROOT}"
echo "[pipeline] dataset_dir=${DATASET_DIR}"

python3 "${PIPELINE_DIR}/validate_room60_visual_isolation.py" \
  --output_dir "${ROOM60_VIS_DIR}" \
  --task "${ROOM60_TASK}" \
  --num_envs "${NUM_ENVS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --steps 120 \
  --record_every 2 \
  --fps 20 \
  "${ROOM60_CAMERA_ARGS[@]}" \
  --sentinel_room_probes_all_envs \
  --record_mosaic \
  --mosaic_max_envs "${NUM_ENVS}" \
  --mosaic_cols 4 \
  --coverage_mode "${ISO_COVERAGE_MODE}" \
  --coverage_count "${ISO_COVERAGE_COUNT}" \
  --mosaic_coverage_mode "${ISO_MOSAIC_COVERAGE_MODE}" \
  --mosaic_chunk_size "${ISO_MOSAIC_CHUNK_SIZE}" \
  $([[ "${ISO_REQUIRE_FULL_MOSAIC}" == "1" ]] && printf '%s' "--require_full_mosaic_coverage") \
  --continue_on_failure \
  --required_gate foreign \
  --overwrite

python3 "${PIPELINE_DIR}/validate_room60_visual_isolation.py" \
  --output_dir "${VIS_DIR}" \
  --task "${CLEANVIEW_TASK}" \
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
  --continue_on_failure \
  --overwrite

VIS_SUMMARY="${VIS_DIR}/visual_isolation_summary.json"

"${RUN_WRAPPER}" -p "${PIPELINE_DIR}/collect_teacher_approach_dataset.py" \
  --task "${COLLECT_TASK}" \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --output_dir "${DATASET_DIR}" \
  --num_envs "${NUM_ENVS}" \
  --target_clean_episodes "${TARGET_CLEAN_EPISODES}" \
  --episodes "${MAX_ATTEMPTED_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --image_every "${IMAGE_EVERY}" \
  --flush_every 25 \
  --seed "${SEED}" \
  --vision_acceptance_summary "${VIS_SUMMARY}" \
  --relabel_teacher_actions \
  "${COMMON_CAMERA_ARGS[@]}" \
  --device "${DEVICE}" --enable_cameras --headless

python3 "${PIPELINE_DIR}/train_approach_student_from_teacher.py" \
  --dataset_dir "${DATASET_DIR}" \
  --output "${BC_OUTPUT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --action_source relabel \
  --action_loss_space raw \
  --prev_action_source label \
  --max_frac_abs_drive_gt_095 "${MAX_FRAC_ABS_DRIVE_GT_095}" \
  --clean_episode_max_pallet_disp_xy_m 0.05 \
  --train_backbone

"${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py" \
  --task "${CLEANVIEW_TASK}" \
  --checkpoint "${BC_OUTPUT}" \
  --checkpoint_type bc \
  --output_dir "${EVAL_DIR}" \
  --episodes "${EVAL_EPISODES}" \
  --num_envs 1 \
  --steps 720 \
  --record_every 3 \
  --fps 30 \
  --seed "${SEED}" \
  "${COMMON_CAMERA_ARGS[@]}" \
  --device "${DEVICE}" --enable_cameras --headless

echo "[pipeline] done"
echo "[pipeline] room60_pressure=${ROOM60_VIS_DIR}/visual_isolation_summary.json"
echo "[pipeline] visual_acceptance=${VIS_SUMMARY}"
echo "[pipeline] dataset=${DATASET_DIR}"
echo "[pipeline] student=${BC_OUTPUT}"
echo "[pipeline] eval=${EVAL_DIR}/summary.json"
