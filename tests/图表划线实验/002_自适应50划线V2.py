"""统一50%划线V2：连续黑线 + 自适应晕染 + 2～14px白带。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
    adaptive_line_segments,
    clean_suspicious_column_lines,
    content_envelope_mask,
)

Image.MAX_IMAGE_PIXELS = None

CASES = (("0f372a06", 0), ("d8b59365", 0), ("1829aea8", 0), ("aecf66d9", 0), ("0cd74f08", 0))
OUTPUT = Path("work/验证/自适应50黑白划线V2")
PURE_WHITE = 225
MAX_BREAK_COUNT = 2
MAX_BREAK_WIDTH = 2
WHITE_MIN_WIDTH = 2
WHITE_MAX_WIDTH = 14


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    Path(__file__).with_name("001_统一50分析划线V1.py"),
    "unified_50_v1",
)
SPLITTER = BASE.load_splitter()
TABLE_CONFIG = TableConfig(
    table_analysis_scale=0.50,
    table_black_line_scale=0.50,
    pipeline_version="test-adaptive-50-v2",
)


def adaptive_ratios(ink_density: float) -> tuple[float, float]:
    """墨迹越稀疏，左右晕染和白缝正交扩张越强。"""

    safe_density = max(0.01, ink_density)
    body_x = float(np.clip(0.015 * 0.10 / safe_density, 0.015, 0.04))
    white_dilate = float(np.clip(0.01 * 0.15 / safe_density, 0.01, 0.03))
    return body_x, white_dilate


def adaptive_body(image: Image.Image):
    """逐级增强上下晕染，直到主要墨迹段自然连成至少85%高度。"""

    gray = np.asarray(image.convert("L"))
    ink = gray < 225
    density = float(ink.mean())
    body_x, white_dilate = adaptive_ratios(density)
    kx = max(3, round(image.width * body_x))
    horizontal = cv2.dilate(
        ink.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)),
    )

    candidates = (0.008, 0.015, 0.025, 0.04, 0.05)
    selected = None
    trials = []
    for ratio_y in candidates:
        ky = max(3, round(image.height * ratio_y))
        smeared = cv2.dilate(
            horizontal,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)),
        ).astype(bool)
        row_density = smeared.mean(axis=1)
        row_runs = BASE.runs(
            row_density >= max(0.002, float(row_density.max()) * 0.01)
        )
        if not row_runs:
            continue
        main = max(row_runs, key=lambda item: item[1] - item[0])
        coverage = (main[1] - main[0]) / image.height
        trials.append({"ratio_y": ratio_y, "coverage": coverage})
        selected = (ratio_y, smeared, main)
        if coverage >= 0.85:
            break

    if selected is None:
        return (
            (0, 0, image.width, image.height),
            ink,
            ink,
            {
                "ink_density": density,
                "body_x_ratio": body_x,
                "white_dilate_ratio": white_dilate,
                "body_y_ratio": 0.0,
                "trials": trials,
            },
        )

    ratio_y, smeared, (y1, y2) = selected
    column_density = smeared[y1:y2].mean(axis=0)
    columns = np.flatnonzero(
        column_density
        >= max(0.002, float(column_density.max()) * 0.01)
    )
    x1, x2 = (
        (0, image.width)
        if columns.size == 0
        else (int(columns[0]), int(columns[-1]) + 1)
    )
    padding = max(4, round(min(image.size) * 0.01))
    box = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image.width, x2 + padding),
        min(image.height, y2 + padding),
    )
    return (
        box,
        ink,
        smeared,
        {
            "ink_density": density,
            "body_x_ratio": body_x,
            "white_dilate_ratio": white_dilate,
            "body_y_ratio": ratio_y,
            "trials": trials,
        },
    )


def continuity(line, gray: np.ndarray, axis: int) -> dict[str, object]:
    """统计线芯中接近纯白的连续断点，不把灰色抗锯齿当成断点。"""

    core_radius = 1
    if axis == 0:
        lower = max(0, line.position - core_radius)
        upper = min(gray.shape[0], line.position + core_radius + 1)
        profile = gray[lower:upper, line.start : line.end].min(axis=0)
    else:
        lower = max(0, line.position - core_radius)
        upper = min(gray.shape[1], line.position + core_radius + 1)
        profile = gray[line.start : line.end, lower:upper].min(axis=1)
    trim = min(round(len(profile) * 0.05), max(0, (len(profile) - 1) // 2))
    if trim:
        profile = profile[trim:-trim]
    white_runs = BASE.runs(profile >= PURE_WHITE)
    widths = [b - a for a, b in white_runs]
    count = len(widths)
    maximum = max(widths, default=0)
    return {
        "position": line.position,
        "break_count": count,
        "maximum_break_width": maximum,
        "break_widths": widths,
        "passed": count <= MAX_BREAK_COUNT and maximum <= MAX_BREAK_WIDTH,
    }


def black_with_continuity(body: Image.Image):
    """先跑原黑线算法，再加严格断点复核；原阈值完全不动。"""

    gray = np.asarray(body.convert("L"))
    ink = gray < TABLE_CONFIG.grid_white_threshold
    envelope = content_envelope_mask(ink)
    raw_rows = adaptive_line_segments(
        ink, envelope, 0, TABLE_CONFIG.grid_black_line_ratio
    )
    raw_columns = adaptive_line_segments(
        ink,
        envelope,
        1,
        TABLE_CONFIG.grid_black_column_line_ratio,
        grayscale=gray,
        endpoint_trim_ratio=TABLE_CONFIG.grid_black_column_endpoint_trim_ratio,
        minimum_contrast=TABLE_CONFIG.grid_black_column_min_contrast,
        contrast_bypass_ratio=TABLE_CONFIG.grid_black_column_contrast_bypass_ratio,
    )

    row_checks = [continuity(line, gray, 0) for line in raw_rows]
    column_checks = [continuity(line, gray, 1) for line in raw_columns]
    used_rows = [
        line for line, check in zip(raw_rows, row_checks) if check["passed"]
    ]
    continuous_columns = [
        line
        for line, check in zip(raw_columns, column_checks)
        if check["passed"]
    ]
    used_columns, spacing_rejected, spacing_message = (
        clean_suspicious_column_lines(continuous_columns, TABLE_CONFIG)
    )
    return {
        "gray": gray,
        "ink": ink,
        "envelope": envelope,
        "raw_rows": raw_rows,
        "raw_columns": raw_columns,
        "rows": used_rows,
        "columns": used_columns,
        "row_checks": row_checks,
        "column_checks": column_checks,
        "spacing_rejected": spacing_rejected,
        "spacing_message": spacing_message,
    }


def spacing_score(items):
    positions = BASE.centers(items)
    if len(positions) < 3:
        return 0.0
    gaps = np.diff(np.asarray(positions))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 0.0
    typical = float(np.median(gaps))
    tolerance = max(2.0, typical * 0.18)
    return float(np.mean(np.abs(gaps - typical) <= tolerance))


def select_width(raw_bands):
    """在2～14px中选择能让间距最稳定的统一白带宽度。"""

    base = [item for item in raw_bands if item[1] - item[0] >= WHITE_MIN_WIDTH]
    if len(base) < 6:
        return base, WHITE_MIN_WIDTH, "候选不足6根，使用2px"
    before = spacing_score(base)
    best, best_width, best_score = base, WHITE_MIN_WIDTH, before
    required = max(5, int(np.ceil(len(base) * 0.25)))
    for width in range(WHITE_MIN_WIDTH + 1, WHITE_MAX_WIDTH + 1):
        kept = [item for item in base if item[1] - item[0] >= width]
        removed = [item for item in base if item[1] - item[0] < width]
        if len(kept) < required or not removed:
            continue
        kept_width = float(np.median([b - a for a, b in kept]))
        removed_width = float(np.median([b - a for a, b in removed]))
        if kept_width < removed_width * 1.6:
            continue
        score = spacing_score(kept)
        if score > best_score:
            best, best_width, best_score = kept, width, score
    if best_width == WHITE_MIN_WIDTH or best_score - before < 0.08:
        return base, WHITE_MIN_WIDTH, (
            f"2～14px尝试后稳定度改善不足8%，使用2px；原稳定度{before:.1%}"
        )
    return best, best_width, (
        f"白带统一宽度2→{best_width}px，稳定度{before:.1%}→{best_score:.1%}"
    )


def white_adaptive(erased: np.ndarray, dilate_ratio: float):
    """同一自适应比例分别用于左右扩张和上下扩张。"""

    kx = max(3, round(erased.shape[1] * dilate_ratio))
    ky = max(3, round(erased.shape[0] * dilate_ratio))
    row_mask = cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)),
    ).astype(bool)
    column_mask = cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)),
    ).astype(bool)
    raw_rows = BASE.bands(row_mask.mean(axis=1), 0.01)
    raw_columns = BASE.bands(column_mask.mean(axis=0), 0.01)
    rows, row_width, row_message = select_width(raw_rows)
    columns, column_width, column_message = select_width(raw_columns)
    return {
        "row_mask": row_mask,
        "column_mask": column_mask,
        "raw_rows": raw_rows,
        "raw_columns": raw_columns,
        "rows": rows,
        "columns": columns,
        "row_width": row_width,
        "column_width": column_width,
        "row_message": row_message,
        "column_message": column_message,
    }


def draw_rejected(body, raw_rows, raw_columns, used_rows, used_columns):
    image = BASE.draw_lines(
        body,
        [line.position for line in raw_rows],
        [line.position for line in raw_columns],
        row_color=(255, 120, 120),
        col_color=(120, 120, 255),
    )
    return BASE.draw_lines(
        image,
        [line.position for line in used_rows],
        [line.position for line in used_columns],
    )


def save_case(prefix: str, region_index: int, raw_root: Path):
    source = BASE.find_source(raw_root, prefix)
    regions, split_debug = BASE.split_boxes(source, SPLITTER)
    region_box = regions[region_index]
    with Image.open(source) as image:
        region = image.convert("RGB").crop(region_box)
    analysis = BASE.half_image(region)

    body_box, full_ink, smeared, adaptive = adaptive_body(analysis)
    body = analysis.crop(body_box)
    black = black_with_continuity(body)
    erased = BASE.erase_lines(
        black["ink"], black["rows"], black["columns"]
    )
    white = white_adaptive(erased, adaptive["white_dilate_ratio"])

    cfg = BASE.Config(
        row_dilate_x=adaptive["white_dilate_ratio"],
        column_dilate_y=adaptive["white_dilate_ratio"],
    )
    final_rows, row_fusion = BASE.fuse(
        [line.position for line in black["rows"]],
        BASE.centers(white["rows"]),
        body.height,
        cfg,
    )
    final_columns, column_fusion = BASE.fuse(
        [line.position for line in black["columns"]],
        BASE.centers(white["columns"]),
        body.width,
        cfg,
    )

    out = OUTPUT / f"{prefix}_region_{region_index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    analysis.save(out / "01_子表原图50.png")
    BASE.mask_image(full_ink).save(out / "02_灰度225墨迹.png")
    BASE.mask_image(smeared).save(out / "03_自适应墨迹晕染.png")
    BASE.draw_box(analysis, body_box).save(out / "04_主要表格块.png")
    body.save(out / "05_主体块50.png")
    BASE.mask_image(black["envelope"]).save(out / "06_表格墨迹包络.png")
    BASE.draw_lines(
        body,
        [line.position for line in black["raw_rows"]],
        [line.position for line in black["raw_columns"]],
    ).save(out / "07_原始黑线.png")
    draw_rejected(
        body,
        black["raw_rows"],
        black["raw_columns"],
        black["rows"],
        black["columns"],
    ).save(out / "08_黑线连续性复核.png")
    BASE.mask_image(erased).save(out / "09_擦除黑线后.png")
    BASE.mask_image(white["row_mask"]).save(out / "10_自适应行晕染.png")
    BASE.mask_image(white["column_mask"]).save(out / "11_自适应列晕染.png")
    BASE.draw_lines(
        body,
        BASE.centers(white["raw_rows"]),
        BASE.centers(white["raw_columns"]),
        row_color=(0, 180, 0),
        col_color=(0, 180, 180),
    ).save(out / "12_原始白线.png")
    BASE.draw_lines(
        body,
        BASE.centers(white["rows"]),
        BASE.centers(white["columns"]),
        row_color=(0, 180, 0),
        col_color=(0, 180, 180),
    ).save(out / "13_宽度复核后白线.png")
    BASE.draw_lines(body, final_rows, final_columns).save(
        out / "14_最终RC.png"
    )

    result = {
        "image_name": source.name,
        "region_index": region_index,
        "split_debug": split_debug,
        "region_box": list(region_box),
        "analysis_size": list(analysis.size),
        "body_box": list(body_box),
        "adaptive": adaptive,
        "black_break_rule": {
            "pure_white_threshold": PURE_WHITE,
            "maximum_break_count": MAX_BREAK_COUNT,
            "maximum_break_width": MAX_BREAK_WIDTH,
        },
        "raw_black_row_count": len(black["raw_rows"]),
        "used_black_row_count": len(black["rows"]),
        "raw_black_column_count": len(black["raw_columns"]),
        "used_black_column_count": len(black["columns"]),
        "row_continuity": black["row_checks"],
        "column_continuity": black["column_checks"],
        "spacing_cleanup": black["spacing_message"],
        "raw_white_row_count": len(white["raw_rows"]),
        "used_white_row_count": len(white["rows"]),
        "raw_white_column_count": len(white["raw_columns"]),
        "used_white_column_count": len(white["columns"]),
        "row_white_width": white["row_width"],
        "column_white_width": white["column_width"],
        "row_white_message": white["row_message"],
        "column_white_message": white["column_message"],
        "row_fusion": row_fusion,
        "column_fusion": column_fusion,
        "row_count": len(final_rows) - 1,
        "column_count": len(final_columns) - 1,
    }
    (out / "诊断数据.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def make_sheet(results):
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (1800, 620 * len(results)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        prefix = result["image_name"][:8]
        region = result["region_index"]
        picture = Image.open(
            OUTPUT / f"{prefix}_region_{region:03d}" / "14_最终RC.png"
        ).convert("RGB")
        scale = min(1700 / picture.width, 520 / picture.height, 1)
        picture = picture.resize(
            (round(picture.width * scale), round(picture.height * scale)),
            Image.Resampling.LANCZOS,
        )
        y = index * 620
        draw.text(
            (20, y + 15),
            (
                f"{prefix} R={result['row_count']} C={result['column_count']} "
                f"body_y={result['adaptive']['body_y_ratio']:.3f} "
                f"white={result['adaptive']['white_dilate_ratio']:.3f}"
            ),
            fill=(0, 0, 0),
            font=font,
        )
        sheet.paste(picture, ((1800 - picture.width) // 2, y + 60))
    sheet.save(OUTPUT / "五图V2最终RC汇总.jpg", quality=90)


def main():
    raw_root = Path(
        "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A"
    )
    results = []
    for index, (prefix, region) in enumerate(CASES, 1):
        print(f"[{index}/{len(CASES)}] {prefix}", flush=True)
        result = save_case(prefix, region, raw_root)
        results.append(result)
        print(
            (
                f"  density={result['adaptive']['ink_density']:.3f} "
                f"body_y={result['adaptive']['body_y_ratio']:.3f} "
                f"white={result['adaptive']['white_dilate_ratio']:.3f} "
                f"R={result['row_count']} C={result['column_count']}"
            ),
            flush=True,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "汇总.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_sheet(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
