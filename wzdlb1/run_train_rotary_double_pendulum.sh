#!/usr/bin/env bash
set -e

# ============================================================
# 二阶倒立摆训练脚本
# 
# 用法:
#   ./run_train_rotary_double_pendulum.sh [选项]
#
# 选项:
#   --num_envs N       并行环境数 (默认: 8192)
#   --max_epochs N     最大训练轮次 (默认: 500)
#   --horizon N        每轮收集的步数 (默认: 128)
#   --headless         无渲染模式 (默认: 开启)
#   --render           有渲染模式
#   其他参数会透传给训练脚本
# ============================================================

# 默认参数
NUM_ENVS=8192
MAX_EPOCHS=500
HORIZON=128
HEADLESS="--headless"
EXTRA_ARGS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECTS_DIR}/forklift_sim/IsaacLab}"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --horizon)
      HORIZON="$2"
      shift 2
      ;;
    --headless)
      HEADLESS="--headless"
      shift
      ;;
    --render)
      HEADLESS=""
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# 激活 conda 环境
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

echo ""
echo "============================================================"
echo "  二阶倒立摆强化学习训练"
echo "============================================================"
echo ""
echo "  配置参数:"
echo "    - num_envs:    $NUM_ENVS"
echo "    - max_epochs:  $MAX_EPOCHS"
echo "    - horizon:     $HORIZON"
echo "    - headless:    ${HEADLESS:-disabled}"
echo ""
echo "  每轮样本数: $((NUM_ENVS * HORIZON))"
echo "  总样本量:   $((NUM_ENVS * HORIZON * MAX_EPOCHS))"
echo ""

cd "${ISAACLAB_DIR}"
# 训练模式 - 使用训练任务变体（高性能）
# 配置：replicate_physics=True, clone_in_fabric=True
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Rotary-Double-Pendulum-Direct-v0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_EPOCHS" \
  $HEADLESS \
  "${EXTRA_ARGS[@]}"
