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
