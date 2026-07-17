"""校验 prepare-tables 产物，并可生成检测框联系图。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验图表准备清单")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--only-fallback", action="store_true", help="联系图只显示 fallback 图片")
    return parser.parse_args()


def make_contact_sheet(records: list[dict], output_path: Path) -> None:
    """把检测预览缩成联系图，便于一次检查异常框。"""

    if not records:
        print("没有符合条件的预览图，跳过联系图")
        return
    columns = 3
    cell_width, cell_height = 420, 340
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        with Image.open(record["preview"]) as source:
            preview = ImageOps.contain(source.convert("RGB"), (cell_width - 20, cell_height - 55))
        sheet.paste(preview, (x + (cell_width - preview.width) // 2, y + 25))
        draw.text((x + 8, y + 6), record["file_name"][:28], fill="black")
        draw.text(
            (x + 8, y + cell_height - 22),
            f"regions={record['regions']} tiles={record['tiles']} {record['sources']}",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=82, optimize=True)
    print(f"检测联系图：{output_path}")


def main() -> None:
    args = parse_args()
    dataset = load_json(args.manifest)
    unique_items = [item for item in dataset["items"] if item["duplicate_of"] is None]
    records: list[dict] = []
    tile_sizes: list[tuple[int, int]] = []
    sources: Counter[str] = Counter()
    grid_sources: Counter[str] = Counter()
    tiling_modes: Counter[str] = Counter()
    for item in unique_items:
        image_manifest_path = Path(item["image_manifest"])
        image_manifest = load_json(image_manifest_path)
        regions = image_manifest["regions"]
        region_sources = [region["detector_source"] for region in regions]
        sources.update(region_sources)
        grid_sources.update(region.get("grid_source", "unavailable") for region in regions)
        tile_count = 0
        for region in regions:
            for tile in region["tiles"]:
                tiling_modes.update([tile.get("tiling_mode", "pixel_overlap")])
                tile_path = image_manifest_path.parent / "tiles" / tile["file_name"]
                with Image.open(tile_path) as image:
                    tile_sizes.append(image.size)
                tile_count += 1
        is_fallback = any("fallback" in source for source in region_sources)
        if not args.only_fallback or is_fallback:
            records.append(
                {
                    "file_name": item["file_name"],
                    "preview": image_manifest_path.parent / "preview_detected.png",
                    "regions": len(regions),
                    "tiles": tile_count,
                    "sources": ",".join(sorted(set(region_sources))),
                }
            )

    report = {
        "image_count": dataset["image_count"],
        "unique_image_count": dataset["unique_image_count"],
        "duplicate_reuse_count": dataset["duplicate_reuse_count"],
        "detector_sources": dict(sources),
        "grid_sources": dict(grid_sources),
        "tiling_modes": dict(tiling_modes),
        "tile_count": len(tile_sizes),
        "max_tile_width": max(width for width, _ in tile_sizes),
        "max_tile_height": max(height for _, height in tile_sizes),
        "over_4096": sum(max(size) > 4096 for size in tile_sizes),
        "selected_previews": [record["file_name"] for record in records],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.contact_sheet:
        make_contact_sheet(records, args.contact_sheet)


if __name__ == "__main__":
    main()
