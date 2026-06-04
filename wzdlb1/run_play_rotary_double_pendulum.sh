#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECTS_DIR}/forklift_sim/IsaacLab}"

# Activate conda env (adjust if needed)
CONDA_SH="/home/uniubi/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  echo "[INFO] Activating conda env: env_isaaclab"
  source "$CONDA_SH"
  conda activate env_isaaclab || {
    echo "[ERROR] Failed to activate conda env: env_isaaclab"
    exit 1
  }
else
  echo "[ERROR] Conda init script not found: $CONDA_SH"
  exit 1
fi

echo "[INFO] Using IsaacLab: ${ISAACLAB_DIR}"
export TERM="${TERM:-xterm-256color}"
# Avoid failure from `tabs` in non-tty shells.
tabs() { :; }
export -f tabs
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_ISAAC_SIM_ACCEPT_EULA=YES
export OMNI_KIT_DISABLE_POPUPS=1
cd "${ISAACLAB_DIR}"

# Play 模式 - 使用 Play 任务变体（优化渲染）
# 配置：num_envs=16, replicate_physics=False, clone_in_fabric=False
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Rotary-Double-Pendulum-Direct-Play-v0 \
  "$@"
