#!/usr/bin/env bash
# Forklift evaluation script
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECTS_DIR}/forklift_sim/IsaacLab}"
CONDA_ENV="env_isaaclab"
TASK_NAME="Isaac-Forklift-PalletInsertLift-Direct-v0"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_name> <checkpoint_path> [num_envs] [video_length]"
    exit 1
fi

RUN_NAME="$1"
CHECKPOINT_PATH="$2"
NUM_ENVS="${3:-16}"
VIDEO_LENGTH="${4:-300}"

source /home/uniubi/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd "${ISAACLAB_DIR}"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task ${TASK_NAME} \
  --num_envs ${NUM_ENVS} \
  --load_run "${RUN_NAME}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --headless --enable_cameras --video --video_length ${VIDEO_LENGTH}
