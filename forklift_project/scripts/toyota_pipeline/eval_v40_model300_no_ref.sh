#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/jianshi/projects/forklift_sim_exp9}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/jianshi/projects/forklift_sim/IsaacLab}"
PIPELINE_DIR="${PROJECT_ROOT}/scripts/toyota_pipeline"
RUN_WRAPPER="${RUN_WRAPPER:-${PIPELINE_DIR}/run_isaaclab_env.sh}"

TASK="${TASK:-Isaac-Forklift-PalletApproach-DirectVisualInsertionCleanViewV40Direct-v0}"
CHECKPOINT="${CHECKPOINT:-${ISAACLAB_DIR}/logs/rsl_rl/direct_visual_insertion_cleanview/2026-05-29_09-37-31_rewardv40_direct_smoke_pushstrict_scratch_s16_mb32/model_300.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/direct_visual_v40_direct_20260529}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/eval_model_300_no_ref_${STAMP}}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-20260529}"
EPISODES="${EPISODES:-12}"
STEPS="${STEPS:-720}"
RECORD_EVERY="${RECORD_EVERY:-3}"
FPS="${FPS:-30}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[v40-eval] checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "[v40-eval] task=${TASK}"
echo "[v40-eval] checkpoint=${CHECKPOINT}"
echo "[v40-eval] output_dir=${OUTPUT_DIR}"

"${RUN_WRAPPER}" -p "${PIPELINE_DIR}/record_toyota_checkpoint_visual_eval.py" \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_type ppo \
  --output_dir "${OUTPUT_DIR}" \
  --num_envs 1 \
  --episodes "${EPISODES}" \
  --steps "${STEPS}" \
  --record_every "${RECORD_EVERY}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --disable_teacher_reference_reset \
  --dual_camera_left_pos 150 75 140 \
  --dual_camera_right_pos 150 -75 140 \
  --dual_camera_left_rpy_deg 0 40 -20 \
  --dual_camera_right_rpy_deg 0 40 20 \
  --dual_camera_hfov_deg 100 \
  --camera_far 8 \
  --save_raw_camera_frames \
  --save_frame_metadata \
  --device "${DEVICE}" --enable_cameras --headless

echo "[v40-eval] summary=${OUTPUT_DIR}/summary.json"
