#!/usr/bin/env bash
# Run IsaacLab commands through the known-good conda environment.
#
# Usage:
#   scripts/toyota_pipeline/run_isaaclab_env.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECTS_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECTS_DIR}/forklift_sim/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/uniubi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[run_isaaclab_env] conda hook not found: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -x "${ISAACLAB_DIR}/isaaclab.sh" ]]; then
  echo "[run_isaaclab_env] IsaacLab launcher not found: ${ISAACLAB_DIR}/isaaclab.sh" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# isaaclab.sh tries to reset terminal tabs; TERM=dumb makes that fail in some
# non-interactive shells.
if [[ "${TERM:-}" == "" || "${TERM:-}" == "dumb" ]]; then
  export TERM=xterm
fi

task_name=""
has_warm_start=0
prev_arg=""
for arg in "$@"; do
  if [[ "${prev_arg}" == "--task" ]]; then
    task_name="${arg}"
  fi
  case "${arg}" in
    --task=*)
      task_name="${arg#--task=}"
      ;;
    --warm_start_checkpoint|--warm_start_checkpoint=*)
      has_warm_start=1
      ;;
  esac
  prev_arg="${arg}"
done

if [[ "${task_name}" == *"DirectVisualInsertionCleanViewV41"* ]]; then
  echo "[run_isaaclab_env] frozen: legacy v41 visual tasks are disabled." >&2
  echo "[run_isaaclab_env] task=${task_name}" >&2
  echo "[run_isaaclab_env] this wrapper intentionally has no environment-variable bypass." >&2
  exit 90
fi

if [[ "${has_warm_start}" == "1" && ( "${task_name}" == *"DirectVisual"* || "${task_name}" == *"DualCamera"* ) ]]; then
  echo "[run_isaaclab_env] frozen: legacy visual warm-start training is disabled for the clean pipeline." >&2
  echo "[run_isaaclab_env] remove --warm_start_checkpoint; this wrapper intentionally has no environment-variable bypass." >&2
  exit 91
fi

args=("$@")
for idx in "${!args[@]}"; do
  case "${args[$idx]}" in
    -p|--python)
      next_idx=$((idx + 1))
      if (( next_idx < ${#args[@]} )); then
        script_arg="${args[$next_idx]}"
        if [[ "${script_arg}" != /* && -f "${PROJECT_ROOT}/${script_arg}" ]]; then
          args[$next_idx]="${PROJECT_ROOT}/${script_arg}"
        fi
      fi
      ;;
  esac
done

cd "${ISAACLAB_DIR}"
exec ./isaaclab.sh "${args[@]}"
