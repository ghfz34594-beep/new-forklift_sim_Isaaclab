#!/usr/bin/env python3
print("placeholder")
#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaaclab", required=True, help="Isaac Lab 根目录")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--task", default="Isaac-Rotary-Double-Pendulum-Direct-v0")
    parser.add_argument("extra", nargs="*", help="额外参数（直接透传到训练脚本）")
    args = parser.parse_args()

    isaaclab_sh = os.path.join(args.isaaclab, "isaaclab.sh")
    train_script = _find_train_script(args.isaaclab)
    if not os.path.isfile(isaaclab_sh):
        print("未找到 isaaclab.sh：", isaaclab_sh)
        sys.exit(1)
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
        *args.extra,
    ]
    print("执行命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
