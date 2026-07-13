"""绘制长图滑窗检测框，辅助调整 Title/Text 规则。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


COLORS = {
    "Title": (230, 20, 20),
    "Text": (20, 90, 230),
    "Table": (20, 170, 40),
    "Figure": (180, 30, 180),
    "Equation": (255, 120, 0),
    "Caption": (0, 160, 160),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制长图检测调试图")
    parser.add_argument("--manifest", required=True, type=Path, help="单张图片 manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-window", default=0, type=int)
    parser.add_argument("--max-windows", default=4, type=int)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    windows = data["windows"][args.start_window : args.start_window + args.max_windows]
    blocks = data["layout_blocks"]
    rendered: list[Image.Image] = []
    for window in windows:
        image_path = args.manifest.parent / "detection_windows" / window["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for block in blocks:
            box = block["box"]
            if box["y2"] <= window["start_y"] or box["y1"] >= window["end_y"]:
                continue
            color = COLORS.get(block["label"], (80, 80, 80))
            local = (
                box["x1"],
                box["y1"] - window["start_y"],
                box["x2"],
                box["y2"] - window["start_y"],
            )
            draw.rectangle(local, outline=color, width=3)
            draw.text((local[0] + 3, max(0, local[1] + 3)), block["label"], fill=color)
        rendered.append(ImageOps.contain(image, (750, 1024)))

    columns = 2
    cell_width, cell_height = 770, 1060
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, image in enumerate(rendered):
        row, column = divmod(index, columns)
        sheet.paste(image, (column * cell_width + 10, row * cell_height + 25))
        ImageDraw.Draw(sheet).text(
            (column * cell_width + 10, row * cell_height + 5),
            f"window {windows[index]['index']} y={windows[index]['start_y']}",
            fill="black",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="JPEG", quality=78, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
