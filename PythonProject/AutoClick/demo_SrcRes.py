#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 src 目录中查找与 Res 目录里同名的图片文件，并拷贝到 finded 目录。

特点：
- src 和 Res 都支持多层文件夹嵌套（递归遍历）。
- 只处理图片文件（扩展名见 IMAGE_EXTS）。
- 两个开关参数开放给你自己设置：
    RES_BY_DIR   控制 Res 用什么方式作为“同名”的匹配依据；
    OUT_KEEP_DIR 控制找到的文件输出到 finded 时是否保持 src 的目录结构。
"""

import os
import shutil

# ========================= 配置区（自己改这里） =========================
SRC_DIR = r"E:\BJBImage\Data\2026.11.6\11.6-MDB41-A跑量后\src"    # 被搜索的源目录（可嵌套）
RES_DIR = r"E:\BJBImage\Data\2026.11.6\11.6-MDB41-A跑量后\Res"    # 提供参考文件名的目录（可嵌套）
FIND_DIR = r"E:\BJBImage\Data\2026.11.6\finded"  # 输出目录

# Res 的匹配方式：
#   False -> 仅按“文件名(含扩展名)”匹配（不管 Res 内部怎么嵌套）
#   True  -> 按“相对 Res 的目录路径 + 文件名”匹配（src 中需有相同相对路径）
RES_BY_DIR = False

# 输出到 finded 的存放方式：
#   False -> 全部平铺到 finded 根目录（重名自动加数字后缀避免覆盖）
#   True  -> 保持 src 中的目录结构存放
OUT_KEEP_DIR = False

# 视为图片的扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp", ".ico"}
# =======================================================================


def collect_res_keys(res_dir, by_dir):
    """收集 Res 中用于匹配的 key。

    by_dir=False: key = 文件名(小写)           -> 值 = 原始文件名
    by_dir=True : key = 相对路径(小写,/)       -> 值 = 原始相对路径
    """
    keys = {}
    for root, _, files in os.walk(res_dir):
        for f in files:
            full = os.path.join(root, f)
            if by_dir:
                rel = os.path.relpath(full, res_dir).replace(os.sep, "/")
                keys[rel.lower()] = rel
            else:
                keys[f.lower()] = f
    return keys


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def main():
    if not os.path.isdir(SRC_DIR):
        raise SystemExit(f"源目录不存在: {SRC_DIR}")
    if not os.path.isdir(RES_DIR):
        raise SystemExit(f"参考目录不存在: {RES_DIR}")
    os.makedirs(FIND_DIR, exist_ok=True)

    res_keys = collect_res_keys(RES_DIR, RES_BY_DIR)
    if not res_keys:
        print("Res 目录中未发现任何文件。")
        return

    copied = 0
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if not is_image(f):
                continue
            full = os.path.join(root, f)

            if RES_BY_DIR:
                rel = os.path.relpath(full, SRC_DIR).replace(os.sep, "/").lower()
                matched = rel in res_keys
            else:
                matched = f.lower() in res_keys

            if not matched:
                continue

            if OUT_KEEP_DIR:
                rel_dir = os.path.relpath(root, SRC_DIR)
                dest_dir = os.path.join(FIND_DIR, rel_dir)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, f)
                if os.path.exists(dest):
                    # 同路径同名校验：已存在则跳过
                    print(f"跳过(已存在): {full}")
                    continue
            else:
                dest = os.path.join(FIND_DIR, f)
                if os.path.exists(dest):
                    # 平铺模式下重名，加计数器后缀避免覆盖
                    base, ext = os.path.splitext(f)
                    dest = os.path.join(FIND_DIR, f"{base}_{copied}{ext}")

            shutil.copy2(full, dest)
            copied += 1
            print(f"已拷贝: {full} -> {dest}")

    print(f"\n完成。共拷贝 {copied} 个文件到 {FIND_DIR}")


if __name__ == "__main__":
    main()
