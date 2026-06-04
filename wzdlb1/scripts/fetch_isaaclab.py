#!/usr/bin/env python3
import argparse
import io
import os
import subprocess
import sys
import time
import urllib.request
import zipfile


def _download_zip(url: str, timeout: int, retries: int) -> bytes:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            print(f"下载失败（第 {attempt}/{retries} 次）：{exc}")
            time.sleep(2)
    raise RuntimeError(last_err)


def _find_train_script(isaaclab_root: str) -> str | None:
    candidates = [
        os.path.join(isaaclab_root, "source", "isaaclab", "isaaclab", "train.py"),
        os.path.join(isaaclab_root, "source", "isaaclab", "isaaclab", "app", "train.py"),
        os.path.join(isaaclab_root, "source", "isaaclab", "isaaclab", "scripts", "train.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _find_play_script(isaaclab_root: str) -> str | None:
    candidates = [
        os.path.join(isaaclab_root, "source", "isaaclab", "isaaclab", "play.py"),
        os.path.join(isaaclab_root, "source", "isaaclab", "isaaclab", "app", "play.py"),
        os.path.join(isaaclab_root, "source", "isaaclab", "isaaclab", "scripts", "play.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="isaaclab", help="目标目录")
    parser.add_argument("--branch", default="main", help="Isaac Lab 分支/标签")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--run-train", action="store_true", help="下载后直接启动训练")
    parser.add_argument("--run-eval", action="store_true", help="启动评估/回放")
    parser.add_argument("--task", default="Isaac-Rotary-Double-Pendulum-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--checkpoint", default="", help="评估时的 checkpoint 路径")
    args, extra = parser.parse_known_args()

    url = f"https://github.com/isaac-sim/IsaacLab/archive/refs/heads/{args.branch}.zip"
    print("下载 Isaac Lab：", url)

    try:
        data = _download_zip(url, timeout=args.timeout, retries=args.retries)
    except Exception as exc:
        print("下载失败：", exc)
        print("建议：安装 git 后执行：")
        print("  git clone https://github.com/isaac-sim/IsaacLab.git isaaclab")
        sys.exit(1)

    os.makedirs(args.dest, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(data))
    z.extractall(args.dest)

    root = os.path.join(args.dest, f"IsaacLab-{args.branch}")
    if os.path.isdir(root):
        for name in os.listdir(root):
            os.replace(os.path.join(root, name), os.path.join(args.dest, name))
        os.rmdir(root)

    print("完成：Isaac Lab 已解压到", os.path.abspath(args.dest))

    if args.run_train or args.run_eval:
        isaaclab_sh = os.path.join(args.dest, "isaaclab.sh")
        if not os.path.isfile(isaaclab_sh):
            print("未找到 isaaclab.sh：", isaaclab_sh)
            sys.exit(1)

        if args.run_train:
            train_script = _find_train_script(args.dest)
            if train_script is None:
                print("未找到 train.py，请检查 Isaac Lab 版本目录结构。")
                sys.exit(1)
            cmd = [
                isaaclab_sh,
                "-p",
                train_script,
                "--task",
                args.task,
                "--num_envs",
                str(args.num_envs),
                *extra,
            ]
            print("执行训练：", " ".join(cmd))
            subprocess.run(cmd, check=True)

        if args.run_eval:
            play_script = _find_play_script(args.dest)
            if play_script is None:
                print("未找到 play.py，请检查 Isaac Lab 版本目录结构。")
                sys.exit(1)
            cmd = [
                isaaclab_sh,
                "-p",
                play_script,
                "--task",
                args.task,
            ]
            if args.checkpoint:
                cmd += ["--checkpoint", args.checkpoint]
            cmd += extra
            print("执行评估：", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
