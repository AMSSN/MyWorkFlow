# -*- coding: utf-8 -*-
"""合并 配殖体1 目录下的 PF/pf 标签，并按新类别顺序重排 YOLO txt 的类别序号。"""

import sys
from collections import Counter
from pathlib import Path

FOLDER = Path(r"E:\OtherFile\TRAIN_FOR\标注文件\配殖体12")
CLASSES_FILE = FOLDER / "classes.txt"

OLD_CLASSES = ["PF", "pf", "PZT", "WBC"]
NEW_CLASSES = ["WBC", "PO", "PF", "PV", "PM", "PZT"]

# PF 与 pf 合并为新的 PF；其余类别按新顺序重新编号
OLD_TO_NEW = {
    OLD_CLASSES.index("PF"): NEW_CLASSES.index("PF"),
    OLD_CLASSES.index("pf"): NEW_CLASSES.index("PF"),
    OLD_CLASSES.index("PZT"): NEW_CLASSES.index("PZT"),
    OLD_CLASSES.index("WBC"): NEW_CLASSES.index("WBC"),
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    txt_files = sorted(
        p for p in FOLDER.glob("*.txt")
        if p.name.lower() != "classes.txt" and p.is_file()
    )

    old_counts = Counter()
    new_counts = Counter()
    unknown = Counter()
    changed_files = 0
    total_lines = 0

    for path in txt_files:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        result = []
        file_changed = False

        for line in lines:
            if not line.strip():
                result.append(line)
                continue

            parts = line.split(maxsplit=1)
            try:
                old_id = int(parts[0])
            except ValueError:
                unknown[f"非数字:{parts[0]}"] += 1
                result.append(line)
                continue

            new_id = OLD_TO_NEW.get(old_id)
            if new_id is None:
                unknown[f"未知类别:{old_id}"] += 1
                result.append(line)
                continue

            rest = parts[1] if len(parts) > 1 else ""
            result.append(f"{new_id} {rest}".rstrip())
            old_counts[OLD_CLASSES[old_id]] += 1
            new_counts[NEW_CLASSES[new_id]] += 1
            total_lines += 1
            if old_id != new_id:
                file_changed = True

        if file_changed:
            path.write_text("\n".join(result) + "\n", encoding="utf-8")
            changed_files += 1

    CLASSES_FILE.write_text("\n".join(NEW_CLASSES) + "\n", encoding="utf-8")

    print(f"处理 txt 文件数: {len(txt_files)}（实际内容有改动: {changed_files}）")
    print(f"处理的标注行数: {total_lines}")
    print("旧类别统计:", {name: old_counts[name] for name in OLD_CLASSES})
    print("新类别统计:", {name: new_counts[name] for name in NEW_CLASSES})
    print("映射关系:", {OLD_CLASSES[k]: (NEW_CLASSES[v], v) for k, v in OLD_TO_NEW.items()})
    if unknown:
        print("未处理的异常行:", dict(unknown))
    print(f"已更新类别表 {CLASSES_FILE.name}: {NEW_CLASSES}")


if __name__ == "__main__":
    main()
