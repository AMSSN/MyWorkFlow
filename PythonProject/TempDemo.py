import cv2
import numpy as np
import os
import shutil
from pathlib import Path


# src拆分
def split_src(src_path: str, split_index, sample_index):
    '''
    src_path 结构:
      --src
      ----0714100540_1(样本名)
      ------0714100540_1_P1_I0_T1.jpg
      ------0714100540_1_P2_I1_T1.jpg
      ......
    图片命名: P=工位, I=第几张, 前缀为样本名
    :param src_path: 源目录
    :param split_index: 需按 I 细分拆分的工位号列表, 如 [1,3]
    :param sample_index: 按样本拆分并归类的工位号(int), 仅可能为 2 或 4, 0 表示不按样本拆
    :return:
    '''
    src_path = Path(src_path)
    # dst 与 src 同级, 目录名改为 split (只替换最后一个 src 段, 避免误伤)
    parts = list(src_path.parts)
    if parts[-1] == "src":
        parts[-1] = "split"
    else:
        parts.append("split")
    dst_path = Path(*parts)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 1. split_index 中需要按 I 细分的工位, 建好目录
    split_dirs = {}  # (工位, I序号) -> 目标目录
    for idx in split_index:
        for i in range(4):  # I0~I3
            d = dst_path / f"P{idx}_I{i}"
            d.mkdir(parents=True, exist_ok=True)
            split_dirs[(idx, i)] = d

    # 2. sample_index 指定的工位, 在 dst/sample{sample_index}/ 下按样本名建目录
    sample_base = None
    sample_dirs = {}  # 样本名 -> 目标目录
    if sample_index in (2, 4):
        sample_base = dst_path / f"sample{sample_index}"
        sample_base.mkdir(parents=True, exist_ok=True)

    for folder in src_path.iterdir():
        if not folder.is_dir():
            continue
        sample_name = folder.name

        # sample_index 工位的目标目录(按样本)
        if sample_base is not None:
            sdir = sample_base / sample_name
            sdir.mkdir(parents=True, exist_ok=True)
            sample_dirs[sample_name] = sdir

        for file in folder.iterdir():
            if not file.is_file():
                continue
            name = file.name
            # 解析工位与 I 序号
            p_idx = None
            i_idx = None
            for idx in (1, 2, 3, 4):
                if f"P{idx}" in name:
                    p_idx = idx
                    break
            # 取 I 序号
            for i in range(4):
                if f"I{i}" in name:
                    i_idx = i
                    break

            if p_idx is None:
                continue

            # 属于 sample_index 工位 -> 归到 sample 目录
            if sample_base is not None and p_idx == sample_index:
                shutil.copy2(file, sample_dirs[sample_name] / name)
                continue

            # 否则按 split_index 规则细分
            if (p_idx, i_idx) in split_dirs:
                shutil.copy2(file, split_dirs[(p_idx, i_idx)] / name)

if __name__ == '__main__':

    # split_src(r"E:\BaiduNetdiskDownload\7.14 0177漏检\src", split_index=[1], sample_index=2)


    pass
