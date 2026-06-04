#!/usr/bin/env bash
set -e

# ============================================================
# 二阶倒立摆模型查看脚本
# 启动 Isaac Sim 并加载模型，让用户可以查看初始结构
# ============================================================

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

export TERM="${TERM:-xterm-256color}"
tabs() { :; }
export -f tabs
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_ISAAC_SIM_ACCEPT_EULA=YES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECTS_DIR}/forklift_sim/IsaacLab}"

echo ""
echo "============================================================"
echo "  二阶倒立摆模型查看器"
echo "============================================================"
echo ""
echo "  启动 Isaac Sim 并加载模型..."
echo "  关闭窗口或按 Ctrl+C 退出"
echo ""

cd "${ISAACLAB_DIR}"
./isaaclab.sh -p "${SCRIPT_DIR}/scripts/view_double_pendulum.py" "$@"
