"""固定分析区域，只比较灰度阈值对横向墨迹扩张和白带数量的影响。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from afac_pipeline.common.models import Box  # noqa: E402
from afac_pipeline.table.config import TableConfig  # noqa: E402
from afac_pipeline.table.grid import _whitespace_centers  # noqa: E402
from afac_pipeline.table.tools.experiment_density_split_and_boundaries import (  # noqa: E402
    binary_preview,
    draw_full_lines,
    whitespace_debug_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="横向白带灰度梯度实验")
    parser.add_argument("--analysis-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[205, 215, 225, 235, 245],
    )
    parser.add_argument(
        "--horizontal-dilate-ratio",
        type=float,
        default=0.0015,
    )
    parser.add_argument("--blank-ink-ratio", type=float, default=0.01)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_contact_sheet(
    records: list[dict[str, object]],
    output_directory: Path,
) -> None:
    """每行一个灰度阈值，依次展示二值、扩张后和白带结果。"""

    card_width, card_height = 460, 340
    columns = 3
    sheet = Image.new(
        "RGB",
        (card_width * columns, card_height * len(records)),
        (225, 225, 225),
    )
    font = load_font(22)
    for row, record in enumerate(records):
        threshold = int(record["gray_threshold"])
        line_count = int(record["horizontal_white_line_count"])
        files = [
            Path(str(record["binary_image"])),
            Path(str(record["dilated_image"])),
            Path(str(record["overlay_image"])),
        ]
        labels = [
            f"gray < {threshold}",
            f"3x1 dilated, gray < {threshold}",
            f"white rows = {line_count}",
        ]
        for column, (path, label) in enumerate(zip(files, labels)):
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(
                    (card_width - 20, card_height - 55),
                    Image.Resampling.LANCZOS,
                )
            card = Image.new("RGB", (card_width, card_height), "white")
            card.paste(
                image,
                (
                    (card_width - image.width) // 2,
                    45 + (card_height - 55 - image.height) // 2,
                ),
            )
            ImageDraw.Draw(card).text(
                (10, 10), label, fill="black", font=font
            )
            sheet.paste(
                card,
                (column * card_width, row * card_height),
            )
    sheet.save(output_directory / "灰度梯度联系图.jpg", quality=90)


def main() -> None:
    args = parse_args()
    if any(not 0 <= value <= 255 for value in args.thresholds):
        raise ValueError("灰度阈值必须位于 0 到 255 之间")
    if args.horizontal_dilate_ratio <= 0:
        raise ValueError("横向扩张比例必须大于 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.analysis_image) as source:
        analysis_image = source.convert("RGB")
    gray = np.asarray(analysis_image.convert("L"))
    analysis_image.save(args.output_dir / "001_固定分析区域原图.png")

    records: list[dict[str, object]] = []
    for threshold in args.thresholds:
        threshold_directory = args.output_dir / f"gray_{threshold:03d}"
        threshold_directory.mkdir(exist_ok=True)
        ink = gray < threshold
        config = TableConfig(
            whitespace_blank_ratio=args.blank_ink_ratio,
            whitespace_min_band=1,
            whitespace_dilate_ratio=0.004,
            whitespace_horizontal_dilate_ratio=(
                args.horizontal_dilate_ratio
            ),
            whitespace_vertical_dilate_ratio=0.004,
        )
        (
            for_rows,
            _,
            row_ratios,
            _,
            horizontal_kernel,
            _,
        ) = whitespace_debug_data(ink, config)
        rows, _ = _whitespace_centers(ink, config)

        binary_path = threshold_directory / "001_扩张前二值墨水.png"
        dilated_path = threshold_directory / "002_3x1横向扩张后.png"
        overlay_path = threshold_directory / "003_检测出的横向白带.png"
        binary_preview(ink).save(binary_path)
        binary_preview(for_rows).save(dilated_path)
        overlay = analysis_image.copy()
        draw_full_lines(
            ImageDraw.Draw(overlay),
            Box(0, 0, overlay.width, overlay.height),
            rows,
            [],
            (255, 128, 0),
            1,
        )
        overlay.save(overlay_path)
        record = {
            "gray_threshold": threshold,
            "horizontal_dilate_kernel": [horizontal_kernel, 1],
            "blank_maximum_ink_ratio": args.blank_ink_ratio,
            "horizontal_white_line_count": len(rows),
            "horizontal_white_line_positions": rows,
            "minimum_row_ink_ratio_after_dilation": float(
                row_ratios.min()
            ),
            "binary_image": str(binary_path.resolve()),
            "dilated_image": str(dilated_path.resolve()),
            "overlay_image": str(overlay_path.resolve()),
        }
        records.append(record)
        (threshold_directory / "数据.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    make_contact_sheet(records, args.output_dir)
    report = {
        "analysis_image": str(args.analysis_image.resolve()),
        "analysis_size": list(analysis_image.size),
        "only_variable": "gray_threshold",
        "horizontal_dilate_ratio": args.horizontal_dilate_ratio,
        "blank_maximum_ink_ratio": args.blank_ink_ratio,
        "records": records,
    }
    (args.output_dir / "实验报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"灰度梯度实验完成：{args.output_dir}")


if __name__ == "__main__":
    main()
