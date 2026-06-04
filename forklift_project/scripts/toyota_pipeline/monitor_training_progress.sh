#!/usr/bin/env bash
# Lightweight monitor for the Toyota dual-camera training run.

set -euo pipefail

LOG_PATH="${1:?usage: monitor_training_progress.sh /path/to/train.log /path/to/run_dir [interval_s]}"
RUN_DIR="${2:?usage: monitor_training_progress.sh /path/to/train.log /path/to/run_dir [interval_s]}"
INTERVAL_S="${3:-180}"
OUT_DIR="${OUT_DIR:-/data/jianshi/projects/forklift_sim_exp9/outputs/toyota_training_main}"
STATUS_PATH="${STATUS_PATH:-${OUT_DIR}/training_status.txt}"

mkdir -p "${OUT_DIR}"

while true; do
  {
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    if pgrep -af "train.py --task Isaac-Forklift-PalletApproach-ToyotaDualCamera" >/dev/null; then
      echo "process=running"
      pgrep -af "train.py --task Isaac-Forklift-PalletApproach-ToyotaDualCamera"
    else
      echo "process=not_running"
    fi
    echo
    echo "gpu="
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    echo
    echo "latest_checkpoint="
    find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%T@ %f %s\n' 2>/dev/null | sort -n | tail -3 || true
    echo
    echo "latest_progress="
    grep -E "Total timesteps:|Time elapsed:|ETA:|Learning iteration" "${LOG_PATH}" 2>/dev/null | tail -12 || true
  } > "${STATUS_PATH}.tmp"
  mv "${STATUS_PATH}.tmp" "${STATUS_PATH}"
  sleep "${INTERVAL_S}"
done
