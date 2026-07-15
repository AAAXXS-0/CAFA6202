"""批量实验：低密度分表，以及黑线优先/白带兜底的边界绘制。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from afac_pipeline.common.models import Box  # noqa: E402
from afac_pipeline.table.config import TableConfig  # noqa: E402
from afac_pipeline.table.density_split import (  # noqa: E402
    _runs,
    boxes_from_bands,
    find_density_bands,
)
from afac_pipeline.table.grid import _whitespace_centers  # noqa: E402
from afac_pipeline.table.ink_region import density_visualization, detect_ink_regions  # noqa: E402


@dataclass(frozen=True)
class LineSegment:
    """局部有效墨水范围内的一条横线或竖线。"""

    position: int
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="墨水分表与黑白边界实验")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-scale", type=float, default=0.20)
    parser.add_argument("--density-scale", type=float, default=0.25)
    parser.add_argument("--analysis-max-side", type=int, default=4096)
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="保存每张分表切图的完整检测中间产物",
    )
    return parser.parse_args()


def map_box(box: Box, from_size: tuple[int, int], to_size: tuple[int, int]) -> Box:
    from_width, from_height = from_size
    to_width, to_height = to_size
    return Box(
        round(box.x1 * to_width / from_width),
        round(box.y1 * to_height / from_height),
        round(box.x2 * to_width / from_width),
        round(box.y2 * to_height / from_height),
    ).clamp(to_width, to_height)


def dense_content_box(ink: np.ndarray, projection_ratio: float = 0.06) -> Box:
    """删除标题、页边等稀疏墨水，只保留当前分表块中的主体区域。

    同一分表块里可能带有左侧标题。标题横向很长，但只占少数几行；按横纵
    投影峰值的 6% 取范围，可以保留主体表格，同时不让标题扩大分析矩形。
    """

    if not ink.any():
        return Box(0, 0, ink.shape[1], ink.shape[0])
    row_ratio = ink.mean(axis=1)
    column_ratio = ink.mean(axis=0)
    rows = np.flatnonzero(
        row_ratio >= max(0.001, float(row_ratio.max()) * projection_ratio)
    )
    columns = np.flatnonzero(
        column_ratio >= max(0.001, float(column_ratio.max()) * projection_ratio)
    )
    if rows.size == 0 or columns.size == 0:
        return Box(0, 0, ink.shape[1], ink.shape[0])
    padding = 2
    return Box(
        max(0, int(columns[0]) - padding),
        max(0, int(rows[0]) - padding),
        min(ink.shape[1], int(columns[-1]) + 1 + padding),
        min(ink.shape[0], int(rows[-1]) + 1 + padding),
    )


def content_envelope_mask(ink: np.ndarray) -> np.ndarray:
    """由整块表格墨水估计二维外形，作为候选线长度的独立分母。

    先在二维上模糊、闭合文字和单元格，再取得每行/列的表格跨度。候选线
    自己的首尾黑点不会决定分母，因此一行文字不能把自身包装成“完整黑线”。
    """

    height, width = ink.shape
    density = ink.astype(np.float32)
    blurred = cv2.GaussianBlur(
        density,
        (0, 0),
        sigmaX=max(1.0, width * 0.008),
        sigmaY=max(1.0, height * 0.008),
    )
    mask = (blurred >= 0.015).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            max(3, round(width * 0.015)) | 1,
            max(3, round(height * 0.015)) | 1,
        ),
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)


def adaptive_line_segments(
    black_ink: np.ndarray,
    envelope_mask: np.ndarray,
    axis: int,
    minimum_ratio: float = 0.90,
    minimum_span_ratio: float = 0.20,
) -> list[LineSegment]:
    """候选线在独立估计的完整表格跨度内必须至少有 90% 黑像素。

    梯形表仍按该行/列在二维包络中的真实长度计算，不把外部白底放入分母；
    但分母绝不能由候选线自己的首尾黑点生成，否则文字横画会被误判为线。
    """

    data = black_ink if axis == 0 else black_ink.T
    envelope = envelope_mask if axis == 0 else envelope_mask.T
    scores = np.zeros(len(data), dtype=np.float32)
    starts = np.zeros(len(data), dtype=np.int32)
    ends = np.zeros(len(data), dtype=np.int32)
    for index, (line, expected) in enumerate(zip(data, envelope)):
        positions = np.flatnonzero(expected)
        if positions.size == 0:
            continue
        start = int(positions[0])
        end = int(positions[-1]) + 1
        if (end - start) / len(line) < minimum_span_ratio:
            continue
        starts[index] = start
        ends[index] = end
        scores[index] = float(black_ink[index, start:end].mean()) if axis == 0 else float(black_ink[start:end, index].mean())

    segments: list[LineSegment] = []
    for run_start, run_end in _runs(scores >= minimum_ratio):
        segments.append(
            LineSegment(
                position=round((run_start + run_end - 1) / 2),
                start=int(starts[run_start:run_end].min()),
                end=int(ends[run_start:run_end].max()),
            )
        )
    return segments


def draw_segments(
    draw: ImageDraw.ImageDraw,
    box: Box,
    horizontal: list[LineSegment],
    vertical: list[LineSegment],
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    """只在真实有效跨度上画线，梯形区域不会再被补成完整矩形。"""

    for line in horizontal:
        draw.line(
            (box.x1 + line.start, box.y1 + line.position, box.x1 + line.end, box.y1 + line.position),
            fill=color,
            width=width,
        )
    for line in vertical:
        draw.line(
            (box.x1 + line.position, box.y1 + line.start, box.x1 + line.position, box.y1 + line.end),
            fill=color,
            width=width,
        )


def draw_full_lines(
    draw: ImageDraw.ImageDraw,
    box: Box,
    horizontal: list[int],
    vertical: list[int],
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    for y in horizontal:
        draw.line((box.x1, box.y1 + y, box.x2, box.y1 + y), fill=color, width=width)
    for x in vertical:
        draw.line((box.x1 + x, box.y1, box.x1 + x, box.y2), fill=color, width=width)


def keep_interior_centers(centers: list[int], length: int) -> list[int]:
    """排除候选表外沿的大片白边，只展示可能属于内部行列的白带。

    墨水分表得到的矩形会带少量外围留白。若直接把这些留白当作无线表格
    的行列空隙，图上就会在表格外缘多画一圈误导性的橙线。这里仅作用于
    实验可视化，保留离两端至少 3%（且至少 8 像素）的白带中心。
    """

    margin = max(8, round(length * 0.03))
    return [center for center in centers if margin < center < length - margin]


def binary_preview(mask: np.ndarray) -> Image.Image:
    """将 True=墨水 的布尔图保存成便于查看的黑白图。"""

    return Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")


def mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 180, 255),
    alpha: int = 75,
) -> Image.Image:
    """把二维包络半透明覆盖到原图，说明实际用作分母的区域。"""

    result = image.convert("RGBA")
    layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
    colored = Image.new("RGBA", result.size, (*color, alpha))
    layer.paste(colored, (0, 0), Image.fromarray(mask.astype(np.uint8) * 255))
    return Image.alpha_composite(result, layer).convert("RGB")


def whitespace_debug_data(
    ink: np.ndarray, config: TableConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """复现 grid._whitespace_centers 的文字扩张，返回可视化和精确投影。"""

    binary = ink.astype(np.uint8)
    horizontal_kernel = max(
        3, round(ink.shape[1] * config.whitespace_dilate_ratio)
    )
    vertical_kernel = max(
        3, round(ink.shape[0] * config.whitespace_dilate_ratio)
    )
    for_rows = cv2.dilate(
        binary,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_kernel, 1)
        ),
    )
    for_columns = cv2.dilate(
        binary,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vertical_kernel)
        ),
    )
    return (
        for_rows,
        for_columns,
        for_rows.mean(axis=1),
        for_columns.mean(axis=0),
        horizontal_kernel,
        vertical_kernel,
    )


def profile_plot(
    values: np.ndarray,
    output_path: Path,
    threshold: float,
    zoom_maximum: float = 0.05,
) -> None:
    """画 0～5% 的墨水比例曲线，红线是 1% 白带上限。"""

    width, height = 1200, 360
    margin = 30
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=(80, 80, 80),
        width=1,
    )
    threshold_y = height - margin - round(
        min(threshold, zoom_maximum)
        / zoom_maximum
        * (height - 2 * margin)
    )
    draw.line(
        (margin, threshold_y, width - margin, threshold_y),
        fill=(255, 0, 0),
        width=2,
    )
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        x = margin + round(
            index / max(1, len(values) - 1) * (width - 2 * margin)
        )
        y = height - margin - round(
            min(float(value), zoom_maximum)
            / zoom_maximum
            * (height - 2 * margin)
        )
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=(0, 80, 220), width=1)
    canvas.save(output_path)


def draw_local_candidates(
    image: Image.Image,
    raw_rows: list[int],
    raw_columns: list[int],
    kept_rows: list[int],
    kept_columns: list[int],
) -> tuple[Image.Image, Image.Image]:
    """分别绘制过滤前候选，以及保留/删除后的对比。"""

    box = Box(0, 0, image.width, image.height)
    before = image.copy()
    draw_full_lines(
        ImageDraw.Draw(before),
        box,
        raw_rows,
        raw_columns,
        (255, 128, 0),
        1,
    )
    after = image.copy()
    after_draw = ImageDraw.Draw(after)
    removed_rows = sorted(set(raw_rows) - set(kept_rows))
    removed_columns = sorted(set(raw_columns) - set(kept_columns))
    draw_full_lines(
        after_draw,
        box,
        removed_rows,
        removed_columns,
        (255, 0, 0),
        2,
    )
    draw_full_lines(
        after_draw,
        box,
        kept_rows,
        kept_columns,
        (0, 180, 0),
        1,
    )
    return before, after


def save_table_intermediates(
    *,
    output_directory: Path,
    table_index: int,
    table_image: Image.Image,
    local_box: Box,
    analysis_box: Box,
    structure_ink: np.ndarray,
    black_ink: np.ndarray,
    envelope: np.ndarray,
    black_rows: list[LineSegment],
    black_columns: list[LineSegment],
    raw_white_rows: list[int],
    raw_white_columns: list[int],
    white_rows: list[int],
    white_columns: list[int],
    row_source: str,
    column_source: str,
    config: TableConfig,
) -> None:
    """保存单张分表切图从输入到最终边界的所有关键中间状态。"""

    table_directory = output_directory / f"table_{table_index:03d}"
    table_directory.mkdir(parents=True, exist_ok=True)
    table_image.save(table_directory / "001_分表切图.png")

    location = table_image.convert("RGBA")
    shade = Image.new("RGBA", location.size, (255, 0, 0, 55))
    visible = Image.new("L", location.size, 255)
    ImageDraw.Draw(visible).rectangle(
        (local_box.x1, local_box.y1, local_box.x2, local_box.y2),
        fill=0,
    )
    location = Image.composite(shade, location, visible)
    location_draw = ImageDraw.Draw(location)
    location_draw.rectangle(
        (local_box.x1, local_box.y1, local_box.x2, local_box.y2),
        outline=(160, 0, 255, 255),
        width=3,
    )
    location.convert("RGB").save(
        table_directory / "002_分表切图中的实际分析框.png"
    )

    analysis_image = table_image.crop(
        (local_box.x1, local_box.y1, local_box.x2, local_box.y2)
    ).convert("RGB")
    analysis_image.save(table_directory / "003_实际分析区域原图.png")
    binary_preview(structure_ink).save(
        table_directory / "004_结构墨水_灰度低于245.png"
    )
    binary_preview(black_ink).save(
        table_directory / "005_黑线墨水_灰度低于225.png"
    )
    binary_preview(envelope).save(
        table_directory / "006_二维表格包络二值图.png"
    )
    mask_overlay(analysis_image, envelope).save(
        table_directory / "007_二维表格包络覆盖位置.png"
    )

    black_image = analysis_image.copy()
    draw_segments(
        ImageDraw.Draw(black_image),
        Box(0, 0, analysis_image.width, analysis_image.height),
        black_rows,
        black_columns,
        (0, 180, 0),
        1,
    )
    black_image.save(table_directory / "008_整线90黑线候选.png")

    (
        for_rows,
        for_columns,
        row_ratios,
        column_ratios,
        horizontal_kernel,
        vertical_kernel,
    ) = whitespace_debug_data(structure_ink, config)
    binary_preview(for_rows).save(
        table_directory / "009_找横向白带_文字左右扩张后.png"
    )
    binary_preview(for_columns).save(
        table_directory / "010_找纵向白带_文字上下扩张后.png"
    )

    edge_image = analysis_image.convert("RGBA")
    edge_layer = Image.new("RGBA", edge_image.size, (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_layer)
    row_margin = max(8, round(analysis_image.height * 0.03))
    column_margin = max(8, round(analysis_image.width * 0.03))
    edge_draw.rectangle(
        (0, 0, analysis_image.width, row_margin),
        fill=(255, 0, 0, 70),
    )
    edge_draw.rectangle(
        (
            0,
            analysis_image.height - row_margin,
            analysis_image.width,
            analysis_image.height,
        ),
        fill=(255, 0, 0, 70),
    )
    edge_draw.rectangle(
        (0, 0, column_margin, analysis_image.height),
        fill=(255, 0, 0, 70),
    )
    edge_draw.rectangle(
        (
            analysis_image.width - column_margin,
            0,
            analysis_image.width,
            analysis_image.height,
        ),
        fill=(255, 0, 0, 70),
    )
    Image.alpha_composite(edge_image, edge_layer).convert("RGB").save(
        table_directory / "011_红色为3%或8像素外沿排除区.png"
    )

    before, after = draw_local_candidates(
        analysis_image,
        raw_white_rows,
        raw_white_columns,
        white_rows,
        white_columns,
    )
    before.save(table_directory / "012_外沿过滤前全部白带.png")
    after.save(
        table_directory / "013_外沿过滤后_绿色保留_红色删除.png"
    )
    profile_plot(
        row_ratios,
        table_directory / "014_横向白带墨水比例_红线为1%.png",
        config.whitespace_blank_ratio,
    )
    profile_plot(
        column_ratios,
        table_directory / "015_纵向白带墨水比例_红线为1%.png",
        config.whitespace_blank_ratio,
    )
    (table_directory / "014_横向白带墨水比例.csv").write_text(
        "position,ink_ratio\n"
        + "\n".join(
            f"{index},{float(value):.8f}"
            for index, value in enumerate(row_ratios)
        ),
        encoding="utf-8",
    )
    (table_directory / "015_纵向白带墨水比例.csv").write_text(
        "position,ink_ratio\n"
        + "\n".join(
            f"{index},{float(value):.8f}"
            for index, value in enumerate(column_ratios)
        ),
        encoding="utf-8",
    )

    final_image = analysis_image.copy()
    final_draw = ImageDraw.Draw(final_image)
    if row_source.startswith("black"):
        draw_segments(final_draw, Box(0, 0, analysis_image.width, analysis_image.height), black_rows, [], (0, 180, 0), 1)
    else:
        draw_full_lines(final_draw, Box(0, 0, analysis_image.width, analysis_image.height), white_rows, [], (255, 128, 0), 1)
    if column_source.startswith("black"):
        draw_segments(final_draw, Box(0, 0, analysis_image.width, analysis_image.height), [], black_columns, (0, 180, 0), 1)
    else:
        draw_full_lines(final_draw, Box(0, 0, analysis_image.width, analysis_image.height), [], white_columns, (255, 128, 0), 1)
    final_image.save(table_directory / "016_当前规则最终边界.png")

    removed_rows = sorted(set(raw_white_rows) - set(white_rows))
    removed_columns = sorted(set(raw_white_columns) - set(white_columns))
    diagnostic = {
        "table_index": table_index,
        "split_image_size": list(table_image.size),
        "analysis_box_in_split": local_box.to_dict(),
        "analysis_box_in_full_analysis_image": analysis_box.to_dict(),
        "structure_ink_gray_threshold": 245,
        "black_ink_gray_threshold": config.grid_white_threshold,
        "black_line_required_ratio": 0.90,
        "white_band_maximum_ink_ratio": config.whitespace_blank_ratio,
        "white_band_minimum_thickness": config.whitespace_min_band,
        "horizontal_detection_dilate_kernel": [horizontal_kernel, 1],
        "vertical_detection_dilate_kernel": [1, vertical_kernel],
        "row_edge_margin": row_margin,
        "column_edge_margin": column_margin,
        "raw_white_rows": raw_white_rows,
        "raw_white_columns": raw_white_columns,
        "kept_white_rows": white_rows,
        "kept_white_columns": white_columns,
        "removed_white_rows": removed_rows,
        "removed_white_columns": removed_columns,
        "black_rows": [line.__dict__ for line in black_rows],
        "black_columns": [line.__dict__ for line in black_columns],
        "selected_row_source": row_source,
        "selected_column_source": column_source,
        "global_kept_horizontal_lines": [
            analysis_box.y1 + value for value in white_rows
        ],
        "global_kept_vertical_lines": [
            analysis_box.x1 + value for value in white_columns
        ],
        "global_removed_horizontal_lines": [
            analysis_box.y1 + value for value in removed_rows
        ],
        "global_removed_vertical_lines": [
            analysis_box.x1 + value for value in removed_columns
        ],
    }
    (table_directory / "诊断数据.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < args.analysis_scale <= 1:
        raise ValueError("analysis-scale 必须位于 (0, 1] 内")
    if not 0 < args.density_scale <= 1:
        raise ValueError("density-scale 必须位于 (0, 1] 内")
    with Image.open(args.image) as source:
        original_size = source.size
        analysis_size = (
            max(1, round(source.width * args.analysis_scale)),
            max(1, round(source.height * args.analysis_scale)),
        )
        if max(analysis_size) > args.analysis_max_side:
            raise ValueError(
                f"固定缩放后最长边为 {max(analysis_size)}，超过安全上限 "
                f"{args.analysis_max_side}；请显式调整 analysis-scale"
            )
        preview = source.convert("RGB").resize(
            analysis_size, Image.Resampling.LANCZOS
        )
    preview.save(args.output_dir / "001_原始缩略图.png")

    ink_result = detect_ink_regions(preview, coarse_scale=args.density_scale)
    density = ink_result.coarse_density
    density_visualization(density).resize(preview.size, Image.Resampling.NEAREST).save(
        args.output_dir / "002_墨水密度图.png"
    )
    horizontal_bands, vertical_bands = find_density_bands(density)
    coarse_boxes = boxes_from_bands(
        density.shape[1], density.shape[0], horizontal_bands, vertical_bands, density
    )
    preview_boxes = [
        map_box(box, (density.shape[1], density.shape[0]), preview.size)
        for box in coarse_boxes
    ]

    split_image = preview.copy()
    split_draw = ImageDraw.Draw(split_image, "RGBA")
    for band in horizontal_bands:
        top = round(band.start * preview.height / density.shape[0])
        bottom = round(band.end * preview.height / density.shape[0])
        split_draw.rectangle((0, top, preview.width, bottom), fill=(255, 0, 0, 70))
    for band in vertical_bands:
        left = round(band.start * preview.width / density.shape[1])
        right = round(band.end * preview.width / density.shape[1])
        split_draw.rectangle((left, 0, right, preview.height), fill=(255, 0, 0, 70))
    for index, box in enumerate(preview_boxes):
        split_draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(0, 80, 255, 255), width=4)
        split_draw.text((box.x1 + 5, box.y1 + 5), f"table-{index + 1}", fill=(0, 80, 255, 255))
    split_image.save(args.output_dir / "003_低密度分表.png")
    analysis_overview = split_image.copy()

    split_directory = args.output_dir / "切表"
    split_directory.mkdir(exist_ok=True)
    black_image = preview.copy()
    white_image = preview.copy()
    choice_image = preview.copy()
    black_draw = ImageDraw.Draw(black_image)
    white_draw = ImageDraw.Draw(white_image)
    choice_draw = ImageDraw.Draw(choice_image)
    config = TableConfig(
        grid_white_threshold=225,
        whitespace_blank_ratio=0.01,
        whitespace_min_band=1,
        whitespace_dilate_ratio=0.004,
    )
    reports: list[dict[str, object]] = []
    for index, box in enumerate(preview_boxes):
        table_image = preview.crop((box.x1, box.y1, box.x2, box.y2))
        table_image.save(split_directory / f"table_{index:03d}.png")
        gray = np.asarray(table_image.convert("L"))
        structure_ink = gray < 245
        black_ink = gray < config.grid_white_threshold
        local_box = dense_content_box(structure_ink)
        analysis_box = Box(
            box.x1 + local_box.x1,
            box.y1 + local_box.y1,
            box.x1 + local_box.x2,
            box.y1 + local_box.y2,
        )
        structure_ink = structure_ink[
            local_box.y1 : local_box.y2,
            local_box.x1 : local_box.x2,
        ]
        black_ink = black_ink[
            local_box.y1 : local_box.y2,
            local_box.x1 : local_box.x2,
        ]
        envelope = content_envelope_mask(structure_ink)
        black_rows = adaptive_line_segments(black_ink, envelope, 0, 0.90)
        black_columns = adaptive_line_segments(black_ink, envelope, 1, 0.90)
        draw_segments(
            black_draw,
            analysis_box,
            black_rows,
            black_columns,
            (0, 180, 0),
        )

        raw_white_rows, raw_white_columns = _whitespace_centers(
            structure_ink, config
        )
        white_rows = keep_interior_centers(raw_white_rows, structure_ink.shape[0])
        white_columns = keep_interior_centers(
            raw_white_columns, structure_ink.shape[1]
        )
        draw_full_lines(
            white_draw,
            analysis_box,
            white_rows,
            white_columns,
            (255, 128, 0),
        )

        reliable_line_count = 5
        row_source = (
            "black-line-0.90"
            if len(black_rows) >= reliable_line_count
            else "white-band"
        )
        column_source = (
            "black-line-0.90"
            if len(black_columns) >= reliable_line_count
            else "white-band"
        )
        if row_source.startswith("black"):
            draw_segments(choice_draw, analysis_box, black_rows, [], (0, 180, 0), 1)
        else:
            draw_full_lines(choice_draw, analysis_box, white_rows, [], (255, 128, 0), 1)
        if column_source.startswith("black"):
            draw_segments(choice_draw, analysis_box, [], black_columns, (0, 180, 0), 1)
        else:
            draw_full_lines(choice_draw, analysis_box, [], white_columns, (255, 128, 0), 1)

        if args.save_intermediates:
            overview_draw = ImageDraw.Draw(analysis_overview)
            overview_draw.rectangle(
                (
                    analysis_box.x1,
                    analysis_box.y1,
                    analysis_box.x2,
                    analysis_box.y2,
                ),
                outline=(160, 0, 255),
                width=3,
            )
            save_table_intermediates(
                output_directory=args.output_dir / "中间产物",
                table_index=index,
                table_image=table_image,
                local_box=local_box,
                analysis_box=analysis_box,
                structure_ink=structure_ink,
                black_ink=black_ink,
                envelope=envelope,
                black_rows=black_rows,
                black_columns=black_columns,
                raw_white_rows=raw_white_rows,
                raw_white_columns=raw_white_columns,
                white_rows=white_rows,
                white_columns=white_columns,
                row_source=row_source,
                column_source=column_source,
                config=config,
            )

        reports.append(
            {
                "table_index": index,
                "split_box": box.to_dict(),
                "analysis_box": analysis_box.to_dict(),
                "black_rows": len(black_rows),
                "black_columns": len(black_columns),
                "white_rows": len(white_rows),
                "white_columns": len(white_columns),
                "filtered_edge_white_rows": len(raw_white_rows) - len(white_rows),
                "filtered_edge_white_columns": len(raw_white_columns) - len(white_columns),
                "selected_row_source": row_source,
                "selected_column_source": column_source,
            }
        )
    black_image.save(args.output_dir / "004_整线90黑线.png")
    white_image.save(args.output_dir / "005_局部白带.png")
    choice_image.save(args.output_dir / "006_最终边界.png")
    if args.save_intermediates:
        analysis_overview.save(
            args.output_dir / "007_蓝色分表_紫色实际分析框.png"
        )
    report = {
        "image": str(args.image.resolve()),
        "original_size": list(original_size),
        "analysis_size": list(preview.size),
        "density_size": [density.shape[1], density.shape[0]],
        "parameters": {
            "analysis_scale": args.analysis_scale,
            "density_scale_from_analysis": args.density_scale,
            "density_scale_from_original": args.analysis_scale * args.density_scale,
            "black_gray_threshold": config.grid_white_threshold,
            "black_line_ratio": 0.90,
            "white_band_ink_ratio": 0.01,
            "density_split_row_ratio": 0.03,
            "density_split_band_mean_ratio": 0.02,
        },
        "horizontal_split_bands": [band.__dict__ for band in horizontal_bands],
        "vertical_split_bands": [band.__dict__ for band in vertical_bands],
        "tables": reports,
    }
    (args.output_dir / "实验报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"实验完成：{args.output_dir}")


if __name__ == "__main__":
    main()
