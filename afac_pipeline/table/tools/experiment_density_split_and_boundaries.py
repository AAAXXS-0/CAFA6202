"""批量实验：低密度分表，以及黑线优先/白带兜底的边界绘制。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from afac_pipeline.common.models import Box  # noqa: E402
from afac_pipeline.table.config import TableConfig  # noqa: E402
from afac_pipeline.table.density_split import boxes_from_bands, find_density_bands  # noqa: E402
from afac_pipeline.table.grid import _line_centers, _whitespace_centers  # noqa: E402
from afac_pipeline.table.ink_region import density_visualization, detect_ink_regions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="墨水分表与黑白边界实验")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preview-max-side", type=int, default=1600)
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


def draw_full_lines(
    draw: ImageDraw.ImageDraw,
    box: Box,
    horizontal: list[int],
    vertical: list[int],
    color: tuple[int, int, int],
    width: int = 3,
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
    with Image.open(args.image) as source:
        preview = source.convert("RGB")
        preview.thumbnail((args.preview_max_side, args.preview_max_side), Image.Resampling.LANCZOS)
    preview.save(args.output_dir / "001_原始缩略图.png")

    ink_result = detect_ink_regions(preview)
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

    black_image = preview.copy()
    white_image = preview.copy()
    choice_image = preview.copy()
    black_draw = ImageDraw.Draw(black_image)
    white_draw = ImageDraw.Draw(white_image)
    choice_draw = ImageDraw.Draw(choice_image)
    config = TableConfig(
        grid_white_threshold=245,
        grid_line_min_ratio=0.90,
        whitespace_blank_ratio=0.01,
        whitespace_min_band=8,
    )
    reports: list[dict[str, object]] = []
    for index, box in enumerate(preview_boxes):
        crop = np.asarray(preview.crop((box.x1, box.y1, box.x2, box.y2)).convert("L"))
        ink = crop < config.grid_white_threshold
        black_rows = _line_centers(ink.mean(axis=1), config.grid_line_min_ratio)
        black_columns = _line_centers(ink.mean(axis=0), config.grid_line_min_ratio)
        raw_white_rows, raw_white_columns = _whitespace_centers(ink, config)
        white_rows = keep_interior_centers(raw_white_rows, crop.shape[0])
        white_columns = keep_interior_centers(raw_white_columns, crop.shape[1])
        draw_full_lines(black_draw, box, black_rows, black_columns, (0, 180, 0))
        draw_full_lines(white_draw, box, white_rows, white_columns, (255, 128, 0))
        selected_rows = black_rows if len(black_rows) >= 2 else white_rows
        selected_columns = black_columns if len(black_columns) >= 2 else white_columns
        row_source = "black-line" if len(black_rows) >= 2 else "white-band"
        column_source = "black-line" if len(black_columns) >= 2 else "white-band"
        draw_full_lines(choice_draw, box, selected_rows, [], (0, 180, 0) if row_source == "black-line" else (255, 128, 0), 4)
        draw_full_lines(choice_draw, box, [], selected_columns, (0, 180, 0) if column_source == "black-line" else (255, 128, 0), 4)
        reports.append({
            "table_index": index,
            "box": box.to_dict(),
            "black_rows": len(black_rows),
            "black_columns": len(black_columns),
            "white_rows": len(white_rows),
            "white_columns": len(white_columns),
            "filtered_edge_white_rows": len(raw_white_rows) - len(white_rows),
            "filtered_edge_white_columns": len(raw_white_columns) - len(white_columns),
            "selected_row_source": row_source,
            "selected_column_source": column_source,
        })
    black_image.save(args.output_dir / "004_黑线候选.png")
    white_image.save(args.output_dir / "005_白带候选.png")
    choice_image.save(args.output_dir / "006_最终黑白边界选择.png")
    report = {
        "image": str(args.image.resolve()),
        "parameters": {
            "gray_ink_threshold": 245,
            "black_line_ratio": 0.90,
            "white_band_ink_ratio": 0.01,
            "density_split_ratio": 0.01,
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
