#!/usr/bin/env bash
# ==============================================================
# 叉车仿真启动脚本
# 使用 conda 环境 env_isaaclab 中的 Isaac Sim 5.1
# ==============================================================

set -euo pipefail

PYTHON="/home/uniubi/miniconda3/envs/env_isaaclab/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/main.py"

# ---------- 默认参数 ----------
PORT=4161
HOST="0.0.0.0"
HEADLESS=false

# ---------- 解析命令行 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) HEADLESS=true; shift ;;
    --port)     PORT="$2"; shift 2 ;;
    --host)     HOST="$2"; shift 2 ;;
    -h|--help)
      echo "用法: $0 [--headless] [--port PORT] [--host HOST]"
      echo ""
      echo "  --headless       无窗口模式运行（适合远程服务器）"
      echo "  --port PORT      Web 服务端口（默认 8080）"
      echo "  --host HOST      Web 服务监听地址（默认 0.0.0.0）"
      exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# ---------- 检查 Python 解释器 ----------
if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] 未找到 Python 解释器: ${PYTHON}"
  echo "        请确认 conda 环境 env_isaaclab 已正确安装。"
  exit 1
fi

# ---------- 打印启动信息 ----------
echo "=================================================="
echo "  叉车仿真 Web 控制  (Isaac Sim 5.1)"
echo "=================================================="
echo "  Python   : ${PYTHON}"
echo "  脚本     : ${MAIN_SCRIPT}"
echo "  Web 地址 : http://localhost:${PORT}"
echo "  无窗口   : ${HEADLESS}"
echo "--------------------------------------------------"

# ---------- 构建参数列表 ----------
ARGS=("--port" "${PORT}" "--host" "${HOST}")
if [[ "${HEADLESS}" == "true" ]]; then
  ARGS+=("--headless")
fi

# ---------- 启动 ----------
cd "${SCRIPT_DIR}"
exec "${PYTHON}" "${MAIN_SCRIPT}" "${ARGS[@]}"
