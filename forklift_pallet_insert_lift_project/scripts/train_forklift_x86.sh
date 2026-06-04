#!/usr/bin/env bash
# Forklift Pallet Insert+Lift Training Script for x86_64 Environment
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECTS_DIR}/forklift_sim/IsaacLab}"
CONDA_ENV="env_isaaclab"
TASK_NAME="Isaac-Forklift-PalletInsertLift-Direct-v0"
NUM_ENVS="${1:-128}"
MAX_ITERATIONS="${2:-2000}"

if [ -f "/home/uniubi/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/uniubi/miniconda3/etc/profile.d/conda.sh
else
    echo "[ERROR] Conda not found"
    exit 1
fi

conda activate ${CONDA_ENV}
cd "${ISAACLAB_DIR}"

echo "[INFO] Starting training: ${TASK_NAME}"
echo "  Environments: ${NUM_ENVS}, Max Iterations: ${MAX_ITERATIONS}"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task ${TASK_NAME} \
  --headless --enable_cameras \
  --num_envs ${NUM_ENVS} \
  --max_iterations ${MAX_ITERATIONS} \
  2>&1 | tee forklift_train.log
