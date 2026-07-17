"""输出无模型墨水轮廓定位的逐步实验图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from afac_pipeline.table.步骤001_墨水密度定位 import (  # noqa: E402
    density_visualization,
    detect_ink_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无模型表格墨水轮廓定位实验")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--yolo-manifest", type=Path)
    parser.add_argument("--preview-max-side", type=int, default=1600)
    parser.add_argument("--coarse-max-side", type=int, default=384)
    return parser.parse_args()


def map_original_box(
    raw: dict, original_size: tuple[int, int], preview: Image.Image
) -> tuple[int, int, int, int]:
    box = raw["box"]
    original_width, original_height = original_size
    return (
        round(box["x1"] * preview.width / original_width),
        round(box["y1"] * preview.height / original_height),
        round(box["x2"] * preview.width / original_width),
        round(box["y2"] * preview.height / original_height),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as source:
        original_size = source.size
        preview = source.convert("RGB")
        preview.thumbnail(
            (args.preview_max_side, args.preview_max_side),
            Image.Resampling.LANCZOS,
        )
    preview.save(args.output_dir / "001_原始缩略图.png")

    result = detect_ink_regions(preview, coarse_max_side=args.coarse_max_side)
    density_visualization(result.coarse_density).resize(
        preview.size,
        Image.Resampling.NEAREST,
    ).save(args.output_dir / "002_墨水密度图.png")
    # 显示为“白色背景包围黑色内容块”，与原图的视觉关系保持一致。
    Image.fromarray(255 - result.connected_mask, mode="L").resize(
        preview.size,
        Image.Resampling.NEAREST,
    ).save(args.output_dir / "003_模糊连通区域.png")

    contour_image = preview.copy()
    contour_draw = ImageDraw.Draw(contour_image)
    for index, region in enumerate(result.regions):
        if len(region.contour) >= 3:
            contour_draw.line(
                [*region.contour, region.contour[0]],
                fill=(0, 200, 0),
                width=5,
            )
        contour_draw.rectangle(
            (region.box.x1, region.box.y1, region.box.x2, region.box.y2),
            outline=(255, 0, 0),
            width=4,
        )
        contour_draw.text(
            (region.box.x1 + 6, region.box.y1 + 6),
            f"ink-{index + 1}",
            fill=(255, 0, 0),
        )
    contour_image.save(args.output_dir / "004_最终墨水轮廓.png")

    comparison = contour_image.copy()
    comparison_draw = ImageDraw.Draw(comparison)
    yolo_regions: list[dict] = []
    if args.yolo_manifest is not None:
        manifest = json.loads(args.yolo_manifest.read_text(encoding="utf-8"))
        yolo_regions = manifest.get("regions", [])
        for index, region in enumerate(yolo_regions):
            coordinates = map_original_box(region, original_size, preview)
            comparison_draw.rectangle(coordinates, outline=(0, 80, 255), width=5)
            comparison_draw.text(
                (coordinates[0] + 6, coordinates[1] + 24),
                f"yolo-{index + 1}",
                fill=(0, 80, 255),
            )
    comparison.save(args.output_dir / "005_墨水轮廓与YOLO对比.png")

    report = {
        "image": str(args.image.resolve()),
        "original_size": list(original_size),
        "preview_size": list(result.preview_size),
        "coarse_size": list(result.coarse_size),
        "ink_regions": [region.to_dict() for region in result.regions],
        "yolo_regions": yolo_regions,
        "note": "内部行列尚未参与本实验；正式实现将先检测表格线，无线时才检测长空白带。",
    }
    (args.output_dir / "实验报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"实验完成：{args.output_dir}")


if __name__ == "__main__":
    main()
