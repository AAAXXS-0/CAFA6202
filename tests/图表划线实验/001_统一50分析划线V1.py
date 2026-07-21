"""统一50%黑白划线实验。

输入使用已经固化的横向分表V2。本脚本只测试几个代表子表，
保存主体定位、黑线、擦线、白线和最终融合网格，不接正式流水线。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
    adaptive_line_segments,
    clean_suspicious_column_lines,
    content_envelope_mask,
)

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class Config:
    """所有可调参数都放在这里。"""

    gray: int = 225
    body_smear_x: float = 0.015
    body_smear_y: float = 0.008
    body_padding: float = 0.01
    row_dilate_x: float = 0.01
    column_dilate_y: float = 0.01
    white_ink_max: float = 0.01
    fusion_large_gap: float = 1.8
    fusion_merge_ratio: float = 0.002


# 有线、无线、梯形、超大图和小残表各取一例。
CASES = (
    ("0cd74f08", 0),
    ("0f372a06", 0),
    ("1829aea8", 0),
    ("aecf66d9", 0),
    ("d8b59365", 0),
    ("d1752e16", 3),
)


def load_splitter():
    path = Path(__file__).parents[1] / "图表分表实验" / "002_横向分表V2.py"
    spec = importlib.util.spec_from_file_location("fixed_split_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载分表脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runs(mask):
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(ids) > 1)
    starts = np.r_[ids[0], ids[breaks + 1]]
    ends = np.r_[ids[breaks] + 1, ids[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def mask_image(mask):
    return Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")


def find_source(root, prefix):
    matches = sorted(root.rglob(f"{prefix}*.jpg"))
    if not matches:
        raise FileNotFoundError(prefix)
    return matches[0]


def map_box(box, from_size, to_size):
    fw, fh = from_size
    tw, th = to_size
    x1, y1, x2, y2 = box
    return (
        round(x1 * tw / fw),
        round(y1 * th / fh),
        round(x2 * tw / fw),
        round(y2 * th / fh),
    )


def split_boxes(source_path, splitter):
    """固化入口：分表数量和安全落刀全部来自V2。"""

    cfg = splitter.V2Config()
    preview = splitter.v1.make_analysis_image(source_path, cfg)
    with Image.open(source_path) as source:
        original_size = source.size
    result, debug = splitter.detect_horizontal_tables_v2(
        preview,
        cfg,
        image_name=source_path.name,
        expected_count=None,
        original_size=original_size,
    )
    boxes = [map_box(box, preview.size, original_size) for box in result.split_boxes]
    return boxes, {
        "preview_size": list(preview.size),
        "preview_boxes": [list(item) for item in result.split_boxes],
        "original_boxes": [list(item) for item in boxes],
        "raw_gaps": debug["raw_gaps"],
        "merged_gaps": debug["merged_gaps"],
        "cut_positions": debug["cut_positions"],
    }


def half_image(image):
    return image.convert("RGB").resize(
        (max(1, round(image.width * 0.5)), max(1, round(image.height * 0.5))),
        Image.Resampling.LANCZOS,
    )


def locate_body(image, cfg):
    """墨迹先左右、再上下晕染，选择纵向最长的主要表格块。"""

    gray = np.asarray(image.convert("L"))
    ink = gray < cfg.gray
    kx = max(3, round(image.width * cfg.body_smear_x))
    ky = max(3, round(image.height * cfg.body_smear_y))
    horizontal = cv2.dilate(
        ink.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)),
    )
    smeared = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)),
    ).astype(bool)

    row_density = smeared.mean(axis=1)
    row_runs = runs(row_density >= max(0.002, float(row_density.max()) * 0.01))
    if not row_runs:
        return (0, 0, image.width, image.height), ink, smeared
    y1, y2 = max(row_runs, key=lambda item: item[1] - item[0])

    col_density = smeared[y1:y2].mean(axis=0)
    columns = np.flatnonzero(
        col_density >= max(0.002, float(col_density.max()) * 0.01)
    )
    x1, x2 = (
        (0, image.width)
        if columns.size == 0
        else (int(columns[0]), int(columns[-1]) + 1)
    )
    pad = max(4, round(min(image.size) * cfg.body_padding))
    return (
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(image.width, x2 + pad),
            min(image.height, y2 + pad),
        ),
        ink,
        smeared,
    )


def black_lines(body, table_cfg):
    """黑线暂不改：横90%，竖95%+灰度差，98%时免灰度对比。"""

    gray = np.asarray(body.convert("L"))
    ink = gray < table_cfg.grid_white_threshold
    envelope = content_envelope_mask(ink)
    rows_found = adaptive_line_segments(
        ink, envelope, 0, table_cfg.grid_black_line_ratio
    )
    raw_columns = adaptive_line_segments(
        ink,
        envelope,
        1,
        table_cfg.grid_black_column_line_ratio,
        grayscale=gray,
        endpoint_trim_ratio=table_cfg.grid_black_column_endpoint_trim_ratio,
        minimum_contrast=table_cfg.grid_black_column_min_contrast,
        contrast_bypass_ratio=table_cfg.grid_black_column_contrast_bypass_ratio,
    )
    columns, rejected, cleanup = clean_suspicious_column_lines(
        raw_columns, table_cfg
    )
    return gray, envelope, rows_found, raw_columns, columns, rejected, cleanup


def erase_lines(ink, rows_found, columns):
    """仅在白线工作矩阵中擦除可信黑线。"""

    result = ink.copy()
    radius = max(1, round(min(ink.shape) * 0.001))
    for line in rows_found:
        a = max(0, line.position - radius)
        b = min(ink.shape[0], line.position + radius + 1)
        result[a:b, line.start : line.end] = False
    for line in columns:
        a = max(0, line.position - radius)
        b = min(ink.shape[1], line.position + radius + 1)
        result[line.start : line.end, a:b] = False
    return result


def bands(profile, maximum):
    """与主体框外沿相连的白带吸附为外框。"""

    return [
        (a, b)
        for a, b in runs(profile <= maximum)
        if a > 0 and b < len(profile)
    ]


def white_lines(erased, cfg):
    """行缝只左右扩张，列缝只上下扩张，两边都先试1%。"""

    kx = max(3, round(erased.shape[1] * cfg.row_dilate_x))
    ky = max(3, round(erased.shape[0] * cfg.column_dilate_y))
    row_mask = cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)),
    ).astype(bool)
    column_mask = cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)),
    ).astype(bool)
    row_bands = bands(row_mask.mean(axis=1), cfg.white_ink_max)
    column_bands = bands(column_mask.mean(axis=0), cfg.white_ink_max)
    return row_mask, column_mask, row_bands, column_bands


def centers(items):
    return [round((a + b - 1) / 2) for a, b in items]


def fuse(black, white, length, cfg):
    """原026～040合并为一个框：黑线作硬边界，白线只负责补缺。"""

    black = sorted({x for x in black if 0 < x < length})
    white = sorted({x for x in white if 0 < x < length})
    anchors = sorted({0, *black, length})
    gaps = [b - a for a, b in zip(anchors, anchors[1:]) if b > a]
    typical = float(np.median(gaps)) if gaps else float(length)

    if len(black) < 3:
        accepted_white = white
        mode = "黑线不足3根，全部采用白线补齐"
    else:
        limit = typical * cfg.fusion_large_gap
        accepted_white = [
            x
            for x in white
            if any(
                a < x < b and b - a >= limit
                for a, b in zip(anchors, anchors[1:])
            )
        ]
        mode = f"黑线作硬边界；白线只补大于{limit:.1f}px的黑线间隔"

    tolerance = max(2, round(length * cfg.fusion_merge_ratio))
    candidates = [(0, 3), (length, 3)]
    candidates += [(x, 2) for x in black]
    candidates += [(x, 1) for x in accepted_white]
    candidates.sort()

    clusters = []
    for item in candidates:
        if not clusters or item[0] - clusters[-1][-1][0] > tolerance:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    result = []
    for cluster in clusters:
        priority = max(x[1] for x in cluster)
        values = [x[0] for x in cluster if x[1] == priority]
        result.append(round(float(np.mean(values))))
    result[0], result[-1] = 0, length
    return sorted(set(result)), {
        "black": black,
        "white": white,
        "accepted_white": accepted_white,
        "typical_black_gap": typical,
        "merge_tolerance": tolerance,
        "mode": mode,
    }


def draw_box(image, box):
    output = image.copy()
    ImageDraw.Draw(output).rectangle(box, outline=(255, 0, 0), width=5)
    return output


def draw_lines(image, rows_found, columns, row_color=(255, 0, 0), col_color=(0, 80, 255)):
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    for y in rows_found:
        draw.line((0, y, output.width - 1, y), fill=row_color, width=3)
    for x in columns:
        draw.line((x, 0, x, output.height - 1), fill=col_color, width=3)
    return output


def save_case(source_path, region_index, out, splitter, cfg, table_cfg):
    regions, split_debug = split_boxes(source_path, splitter)
    if region_index >= len(regions):
        raise IndexError(f"{source_path.name}只有{len(regions)}块")
    region_box = regions[region_index]
    with Image.open(source_path) as source:
        region = source.convert("RGB").crop(region_box)
    analysis = half_image(region)

    body_box, full_ink, smeared = locate_body(analysis, cfg)
    body = analysis.crop(body_box)
    gray, envelope, black_rows, raw_black_cols, black_cols, rejected, cleanup = (
        black_lines(body, table_cfg)
    )
    erased = erase_lines(gray < cfg.gray, black_rows, black_cols)
    row_mask, column_mask, row_bands, column_bands = white_lines(erased, cfg)

    final_rows, row_fusion = fuse(
        [x.position for x in black_rows],
        centers(row_bands),
        body.height,
        cfg,
    )
    final_cols, col_fusion = fuse(
        [x.position for x in black_cols],
        centers(column_bands),
        body.width,
        cfg,
    )

    out.mkdir(parents=True, exist_ok=True)
    analysis.save(out / "01_子表原图50.png")
    mask_image(full_ink).save(out / "02_灰度225墨迹.png")
    mask_image(smeared).save(out / "03_墨迹晕染.png")
    draw_box(analysis, body_box).save(out / "04_主要表格块.png")
    body.save(out / "05_主体块50.png")
    mask_image(envelope).save(out / "06_表格墨迹包络.png")
    draw_lines(
        body,
        [x.position for x in black_rows],
        [x.position for x in raw_black_cols],
    ).save(out / "07_原始黑线.png")
    draw_lines(
        body,
        [x.position for x in black_rows],
        [x.position for x in black_cols],
    ).save(out / "08_清理后黑线.png")
    mask_image(erased).save(out / "09_擦除黑线后.png")
    mask_image(row_mask).save(out / "10_行白缝左右扩张1percent.png")
    mask_image(column_mask).save(out / "11_列白缝上下扩张1percent.png")
    draw_lines(
        body,
        centers(row_bands),
        centers(column_bands),
        row_color=(0, 180, 0),
        col_color=(0, 180, 180),
    ).save(out / "12_白线候选.png")
    draw_lines(body, final_rows, final_cols).save(out / "13_黑白融合最终RC.png")

    result = {
        "source": str(source_path.resolve()),
        "image_name": source_path.name,
        "region_index": region_index,
        "split_count": len(regions),
        "split_debug": split_debug,
        "region_original_box": list(region_box),
        "analysis_size": list(analysis.size),
        "body_box_at_50_percent": list(body_box),
        "config": asdict(cfg),
        "black_rows": [x.__dict__ for x in black_rows],
        "raw_black_columns": [x.__dict__ for x in raw_black_cols],
        "used_black_columns": [x.__dict__ for x in black_cols],
        "rejected_black_columns": [x.to_dict() for x in rejected],
        "black_cleanup": cleanup,
        "row_white_bands": [list(x) for x in row_bands],
        "column_white_bands": [list(x) for x in column_bands],
        "row_fusion": row_fusion,
        "column_fusion": col_fusion,
        "row_boundaries_50": final_rows,
        "column_boundaries_50": final_cols,
        "row_count": len(final_rows) - 1,
        "column_count": len(final_cols) - 1,
    }
    (out / "诊断数据.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def make_sheet(results, output):
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (1800, 620 * 3), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, result in enumerate(results):
        prefix = result["image_name"][:8]
        region = result["region_index"]
        path = output / f"{prefix}_region_{region:03d}" / "13_黑白融合最终RC.png"
        picture = Image.open(path).convert("RGB")
        scale = min(850 / picture.width, 540 / picture.height, 1)
        picture = picture.resize(
            (max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x, y = (i % 2) * 900, (i // 2) * 620
        draw.text(
            (x + 20, y + 15),
            f"{prefix} region={region} R={result['row_count']} C={result['column_count']}",
            fill=(0, 0, 0),
            font=font,
        )
        sheet.paste(picture, (x + (900 - picture.width) // 2, y + 60))
    sheet.save(output / "代表图最终RC汇总.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/验证/统一50黑白划线V1"),
    )
    args = parser.parse_args()

    splitter = load_splitter()
    cfg = Config()
    table_cfg = TableConfig(
        table_analysis_scale=0.50,
        table_black_line_scale=0.50,
        pipeline_version="test-unified-50-v1",
    )
    results = []
    for i, (prefix, region_index) in enumerate(CASES, 1):
        source = find_source(args.raw_root, prefix)
        print(f"[{i}/{len(CASES)}] {prefix} region={region_index}", flush=True)
        result = save_case(
            source,
            region_index,
            args.output / f"{prefix}_region_{region_index:03d}",
            splitter,
            cfg,
            table_cfg,
        )
        results.append(result)
        print(f"  -> R={result['row_count']} C={result['column_count']}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "汇总.json").write_text(
        json.dumps(
            {"config": asdict(cfg), "cases": [list(x) for x in CASES], "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    make_sheet(results, args.output)
    print(f"完成：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
