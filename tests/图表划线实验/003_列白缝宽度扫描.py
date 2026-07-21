"""列白缝最小宽度 1～40 像素扫描实验。

只处理0f372a06、1829aea8、d8b59365三个代表图。
黑线和横向白缝不改；每张图固定当前V2的50%主体和列晕染结果，
只改变列白缝的最小宽度，便于观察哪个阈值合适。
"""

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
Image.MAX_IMAGE_PIXELS = None

V2_PATH = Path(__file__).with_name("002_自适应50划线V2.py")
spec = importlib.util.spec_from_file_location("adaptive_line_v2", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(V2_PATH)
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)

RAW_ROOT = Path(
    "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A"
)
OUTPUT = Path("work/验证/列白缝宽度扫描V1")
CASES = (("0f372a06", 0), ("1829aea8", 0), ("d8b59365", 0))


def find_source(prefix):
    matches = sorted(RAW_ROOT.rglob(f"{prefix}*.jpg"))
    if not matches:
        raise FileNotFoundError(prefix)
    return matches[0]


def bands(profile):
    return [
        (a, b)
        for a, b in v2.BASE.runs(profile <= 0.01)
        if a > 0 and b < len(profile)
    ]


def make_column_mask(erased, ratio):
    kernel = max(3, round(erased.shape[0] * ratio))
    return cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel)),
    ).astype(bool)


def first_stable_threshold(raw_bands, maximum=20, repeat=3):
    """寻找第一个稳定区间；找到后立即停止，不再扫描后续阈值。"""
    counts = {}
    for threshold in range(1, maximum + 1):
        counts[threshold] = sum(
            end - start >= threshold for start, end in raw_bands
        )
        if threshold >= repeat:
            values = [
                counts[threshold - repeat + 1 + offset]
                for offset in range(repeat)
            ]
            if values[0] > 0 and len(set(values)) == 1:
                return {
                    "threshold": threshold - repeat + 1,
                    "band_count": values[0],
                    "counts_until_stop": counts,
                    "fatal": False,
                }
    return {
        "threshold": None,
        "band_count": 0,
        "counts_until_stop": counts,
        "fatal": True,
    }


def draw_columns(body, bands_at_threshold):
    output = body.convert("RGB")
    draw = ImageDraw.Draw(output)
    for start, end in bands_at_threshold:
        position = round((start + end - 1) / 2)
        draw.line(
            (position, 0, position, output.height - 1),
            fill=(0, 180, 180),
            width=3,
        )
    return output


def make_sheet(body, all_bands, output_path):
    font = ImageFont.load_default()
    card_w, card_h = 480, 330
    sheet = Image.new("RGB", (card_w * 5, card_h * 8), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, threshold in enumerate(range(1, 41)):
        kept = [
            band for band in all_bands
            if band[1] - band[0] >= threshold
        ]
        picture = draw_columns(body, kept)
        scale = min(450 / picture.width, 275 / picture.height, 1)
        picture = picture.resize(
            (
                max(1, round(picture.width * scale)),
                max(1, round(picture.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        x = index % 5 * card_w
        y = index // 5 * card_h
        draw.text(
            (x + 10, y + 8),
            f"minimum={threshold}px  bands={len(kept)}",
            fill=(0, 0, 0),
            font=font,
        )
        sheet.paste(
            picture,
            (x + (card_w - picture.width) // 2, y + 35),
        )
        draw.rectangle(
            (x, y, x + card_w - 1, y + card_h - 1),
            outline=(180, 180, 180),
            width=2,
        )
    sheet.save(output_path, quality=90)


def run_case(prefix, region_index):
    source = find_source(prefix)
    splitter = v2.SPLITTER
    regions, split_debug = v2.BASE.split_boxes(source, splitter)
    region_box = regions[region_index]
    with Image.open(source) as image:
        region = image.convert("RGB").crop(region_box)
    analysis = v2.BASE.half_image(region)
    body_box, full_ink, smeared, adaptive = v2.adaptive_body(
        analysis
    )
    body = analysis.crop(body_box)
    black = v2.black_with_continuity(body)
    erased = v2.BASE.erase_lines(
        black["ink"], black["rows"], black["columns"]
    )
    ratio = adaptive["white_dilate_ratio"]
    column_mask = make_column_mask(erased, ratio)
    raw_bands = bands(column_mask.mean(axis=0))

    case_dir = OUTPUT / f"{prefix}_region_{region_index:03d}"
    threshold_dir = case_dir / "阈值图"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    analysis.save(case_dir / "00_子表原图50.png")
    v2.BASE.mask_image(smeared).save(case_dir / "01_自适应晕染.png")
    body.save(case_dir / "02_主体块50.png")
    v2.BASE.mask_image(column_mask).save(case_dir / "03_列方向晕染.png")
    v2.BASE.draw_lines(
        body,
        [],
        v2.BASE.centers(raw_bands),
        row_color=(0, 0, 0),
        col_color=(0, 180, 180),
    ).save(case_dir / "04_原始列白缝.png")

    counts = {}
    for threshold in range(1, 41):
        kept = [
            band for band in raw_bands
            if band[1] - band[0] >= threshold
        ]
        counts[str(threshold)] = {
            "column_count": len(kept),
            "bands": [list(item) for item in kept],
        }
        draw_columns(body, kept).save(
            threshold_dir / f"最小宽度_{threshold:02d}px.png"
        )

    make_sheet(body, raw_bands, case_dir / "1到40像素汇总.jpg")
    result = {
        "image_name": source.name,
        "region_index": region_index,
        "region_box": list(region_box),
        "split_debug": split_debug,
        "analysis_size": list(analysis.size),
        "body_box": list(body_box),
        "adaptive": adaptive,
        "column_dilate_ratio": ratio,
        "raw_band_count": len(raw_bands),
        "raw_bands": [list(item) for item in raw_bands],
        "thresholds": counts,
        "first_stable": first_stable_threshold(raw_bands),
    }
    (case_dir / "列白缝扫描数据.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for prefix, region in CASES:
        print(f"[scan] {prefix} region={region}", flush=True)
        result = run_case(prefix, region)
        results.append(result)
        print(
            f"  raw={result['raw_band_count']} "
            f"ratio={result['column_dilate_ratio']:.3f}",
            flush=True,
        )
    (OUTPUT / "汇总.json").write_text(
        json.dumps(
            {"cases": [list(item) for item in CASES], "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
