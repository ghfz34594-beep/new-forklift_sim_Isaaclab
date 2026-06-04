#!/usr/bin/env python3
import os
import glob
from pathlib import Path


def _find_candidates():
    home = str(Path.home())
    candidates = []

    env_vars = [
        "ISAACSIM_PATH",
        "ISAAC_SIM_PATH",
        "OMNI_ISAAC_SIM_PATH",
        "OMNIVERSE_ROOT",
        "OV_PKG_ROOT",
    ]
    for key in env_vars:
        value = os.environ.get(key)
        if value:
            candidates.append((f"env:{key}", value))

    # Common Omniverse install roots
    search_roots = [
        os.path.join(home, ".local", "share", "ov", "pkg"),
        os.path.join(home, "omniverse"),
        os.path.join(home, "Omniverse"),
        os.path.join(home, ".nvidia-omniverse"),
    ]

    patterns = [
        "isaac_sim-*",
        "isaac-sim-*",
        "IsaacSim-*",
        "Isaac_Sim-*",
    ]

    for root in search_roots:
        for pat in patterns:
            for path in glob.glob(os.path.join(root, pat)):
                candidates.append(("scan", path))

    # Deduplicate and keep existing directories only
    uniq = []
    seen = set()
    for source, path in candidates:
        if path in seen:
            continue
        if os.path.isdir(path):
            uniq.append((source, path))
            seen.add(path)
    return uniq


def main():
    candidates = _find_candidates()
    if not candidates:
        print("未找到 Isaac Sim 安装目录。")
        print("请确认已安装，并设置环境变量 ISAACSIM_PATH 或 ISAAC_SIM_PATH。")
        return

    print("检测到的 Isaac Sim 安装目录候选：")
    for idx, (source, path) in enumerate(candidates, start=1):
        print(f"{idx:2d}. [{source}] {path}")

    print("\n推荐：选择一个正确路径，设置环境变量，例如：")
    print("  export ISAACSIM_PATH=/path/to/isaac_sim")


if __name__ == "__main__":
    main()
