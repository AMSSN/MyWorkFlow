#!/usr/bin/env python3
"""
Defect contour refinement for visual inspection.

Typical usage:
  python contour_refine.py --input defect_mask.png --output-dir result --mask-input
  python contour_refine.py --input defect_image.png --output-dir result --threshold otsu --polarity dark

The script outputs:
  - refined_mask.png: cleaned binary defect mask
  - overlay.png: contours drawn on the source image
  - contours.json: contour points, boxes, area, perimeter, and rotated rectangles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def require_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install -r requirements.txt"
        ) from exc
    return cv2, np


def odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def to_gray(cv2: Any, image: Any) -> Any:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def build_initial_mask(cv2: Any, np: Any, image: Any, args: argparse.Namespace) -> Any:
    gray = to_gray(cv2, image)

    if args.blur > 0:
        blur_size = odd_kernel_size(args.blur)
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    if args.mask_input:
        _, mask = cv2.threshold(gray, args.mask_threshold, 255, cv2.THRESH_BINARY)
        return mask

    if args.threshold == "fixed":
        threshold_value = args.threshold_value
        if args.polarity == "both":
            baseline = int(np.median(gray))
            diff = cv2.absdiff(gray, np.full(gray.shape, baseline, dtype=np.uint8))
            _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
            return mask
        flag = cv2.THRESH_BINARY_INV if args.polarity == "dark" else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, threshold_value, 255, flag)
        return mask

    if args.threshold == "adaptive":
        block_size = odd_kernel_size(args.adaptive_block)
        c_value = args.adaptive_c
        if args.polarity == "both":
            background = cv2.medianBlur(gray, block_size)
            diff = cv2.absdiff(gray, background)
            return cv2.adaptiveThreshold(
                diff,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_value,
            )
        flag = cv2.THRESH_BINARY_INV if args.polarity == "dark" else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            flag,
            block_size,
            c_value,
        )

    if args.polarity == "dark":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif args.polarity == "bright":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        baseline = int(np.median(gray))
        diff = cv2.absdiff(gray, np.full(gray.shape, baseline, dtype=np.uint8))
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask.astype(np.uint8)


def fill_holes(cv2: Any, np: Any, mask: Any) -> Any:
    height, width = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def remove_small_components(cv2: Any, np: Any, mask: Any, min_area: float, max_area: float) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        cleaned[labels == label] = 255
    return cleaned


def refine_mask(cv2: Any, np: Any, mask: Any, args: argparse.Namespace) -> Any:
    mask = remove_small_components(cv2, np, mask, args.min_area, args.max_area)

    if args.close > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_kernel_size(args.close), odd_kernel_size(args.close))
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=args.close_iter)

    if args.open > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_kernel_size(args.open), odd_kernel_size(args.open))
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=args.open_iter)

    if args.fill_holes:
        mask = fill_holes(cv2, np, mask)

    return remove_small_components(cv2, np, mask, args.min_area, args.max_area)


def contour_to_record(cv2: Any, contour: Any, index: int, epsilon_ratio: float) -> dict[str, Any]:
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(0.0, epsilon_ratio) * perimeter
    optimized = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour

    x, y, w, h = cv2.boundingRect(optimized)
    area = float(cv2.contourArea(optimized))
    rect = cv2.minAreaRect(optimized)
    box = cv2.boxPoints(rect).round().astype(int).tolist()
    points = optimized.reshape(-1, 2).astype(int).tolist()

    return {
        "id": index,
        "area_px": round(area, 3),
        "perimeter_px": round(perimeter, 3),
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "rotated_box_points": box,
        "point_count": len(points),
        "contour_points": points,
    }


def process_one(input_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    cv2, np = require_cv2()

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Cannot read image: {input_path}")
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    mask = build_initial_mask(cv2, np, image, args)
    refined = refine_mask(cv2, np, mask, args)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    records = []
    draw_contours = []
    for index, contour in enumerate(contours, start=1):
        if cv2.contourArea(contour) < args.min_area:
            continue
        record = contour_to_record(cv2, contour, index, args.epsilon_ratio)
        records.append(record)
        draw_contours.append(np.array(record["contour_points"], dtype=np.int32).reshape(-1, 1, 2))

    if args.mask_input:
        overlay = cv2.cvtColor(to_gray(cv2, image), cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(overlay, draw_contours, -1, (0, 0, 255), args.line_width)
    for record in records:
        x, y, w, h = record["bbox_xywh"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 180, 255), 1)
        cv2.putText(
            overlay,
            str(record["id"]),
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem_dir = output_dir / input_path.stem if args.batch else output_dir
    stem_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(stem_dir / "refined_mask.png"), refined)
    cv2.imwrite(str(stem_dir / "overlay.png"), overlay)
    with (stem_dir / "contours.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "input": str(input_path),
                "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
                "contour_count": len(records),
                "contours": records,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine defect contours and export contour geometry.")
    parser.add_argument("--input", default=r"E:\TempFolder\Testlunkuo\0707190411_1_P1_I2_T1.png", help="Input image, mask, or directory in batch mode.")
    parser.add_argument("--output-dir", default="contour_result", help="Directory for output files.")
    parser.add_argument("--batch", action="store_true", help="Process all common image files in --input directory.")

    parser.add_argument("--mask-input", action="store_true", help="Treat input as an existing binary/grayscale mask.")
    parser.add_argument("--mask-threshold", type=int, default=127, help="Threshold for --mask-input.")
    parser.add_argument("--threshold", choices=["otsu", "adaptive", "fixed"], default="otsu")
    parser.add_argument("--threshold-value", type=int, default=127, help="Used when --threshold fixed.")
    parser.add_argument("--polarity", choices=["dark", "bright", "both"], default="bright")
    parser.add_argument("--adaptive-block", type=int, default=51)
    parser.add_argument("--adaptive-c", type=float, default=3)
    parser.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size. Use 0 to disable.")

    parser.add_argument("--min-area", type=float, default=20, help="Remove defect regions smaller than this area.")
    parser.add_argument("--max-area", type=float, default=0, help="Remove regions larger than this area. 0 disables.")
    parser.add_argument("--close", type=int, default=5, help="Closing kernel size for joining broken edges. 0 disables.")
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--open", type=int, default=3, help="Opening kernel size for removing speckles. 0 disables.")
    parser.add_argument("--open-iter", type=int, default=1)
    parser.add_argument("--fill-holes", action="store_true", default=True)
    parser.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.002,
        help="Contour simplification ratio. Lower is more precise, higher is smoother.",
    )
    parser.add_argument("--line-width", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.batch:
        suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        paths = sorted(path for path in input_path.iterdir() if path.suffix.lower() in suffixes)
        if not paths:
            raise SystemExit(f"No image files found in: {input_path}")
        for path in paths:
            process_one(path, output_dir, args)
        return

    process_one(input_path, output_dir, args)


if __name__ == "__main__":
    main()
