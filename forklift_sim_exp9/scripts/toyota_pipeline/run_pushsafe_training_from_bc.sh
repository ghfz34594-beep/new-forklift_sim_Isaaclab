#!/usr/bin/env bash
# Run the Toyota PushSafe training sequence after formal teleop data collection.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/jianshi/projects/forklift_sim_exp9}"
RUN="${RUN:-${ROOT_DIR}/scripts/toyota_pipeline/run_isaaclab_env.sh}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/data/toyota_approach_bc/formal_v1}"
BC_OUTPUT="${BC_OUTPUT:-${ROOT_DIR}/data/toyota_approach_bc/approach_bc_v1.pt}"
BC_EPOCHS="${BC_EPOCHS:-10}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-32}"
TASK="${TASK:-Isaac-Forklift-PalletApproach-ToyotaDualCameraPushSafe-v0}"
SMOKE_ITERS="${SMOKE_ITERS:-20}"
SMOKE_ENVS="${SMOKE_ENVS:-1}"
MAIN_ENVS="${MAIN_ENVS:-1}"
MAIN_ITERS="${MAIN_ITERS:-2000}"
START_MAIN="${START_MAIN:-0}"

cd "${ROOT_DIR}"

python "${ROOT_DIR}/scripts/toyota_pipeline/validate_teleop_dataset.py" \
  --dataset_dir "${DATASET_DIR}" \
  --min_sessions 20 \
  --min_clean_sessions 10 \
  --require_summary

"${RUN}" -p "${ROOT_DIR}/scripts/toyota_pipeline/train_approach_bc.py" \
  --dataset_dir "${DATASET_DIR}" \
  --output "${BC_OUTPUT}" \
  --epochs "${BC_EPOCHS}" \
  --batch_size "${BC_BATCH_SIZE}" \
  --num_workers 0 \
  --device cuda:0

"${RUN}" -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --headless --enable_cameras \
  --num_envs "${SMOKE_ENVS}" \
  --max_iterations "${SMOKE_ITERS}" \
  --bc_checkpoint "${BC_OUTPUT}"

if [[ "${START_MAIN}" == "1" ]]; then
  "${RUN}" -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${TASK}" \
    --headless --enable_cameras \
    --num_envs "${MAIN_ENVS}" \
    --max_iterations "${MAIN_ITERS}" \
    --bc_checkpoint "${BC_OUTPUT}"
else
  echo "[pushsafe] START_MAIN=0, main PPO training not started."
fi
