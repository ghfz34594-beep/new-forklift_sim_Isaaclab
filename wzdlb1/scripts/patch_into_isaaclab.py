#!/usr/bin/env python3
import argparse
import os
import shutil


def copy_tree(src: str, dst: str):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(target_root, name)
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaaclab", required=True, help="Isaac Lab 源码目录")
    parser.add_argument("--patch", required=True, help="补丁根目录（含 source/）")
    args = parser.parse_args()

    src_root = os.path.join(args.patch, "source")
    if not os.path.isdir(src_root):
        raise SystemExit(f"补丁目录缺少 source/: {src_root}")

    if not os.path.isdir(args.isaaclab):
        raise SystemExit(f"Isaac Lab 目录不存在: {args.isaaclab}")

    copy_tree(src_root, args.isaaclab)
    print("补丁已合入:", os.path.abspath(args.isaaclab))


if __name__ == "__main__":
    main()
