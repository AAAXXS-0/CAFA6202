"""表体纵向滑动窗口实验。

窗口内列白缝在1px向上扫描，第一次出现连续3个阈值数量相同就停止。
随后比较相邻窗口的白缝数量和位置，选择最长的稳定窗口段作为表体。
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
spec = importlib.util.spec_from_file_location("adaptive_v2_window", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(V2_PATH)
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)

RAW_ROOT = Path(
    "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A"
)
OUTPUT = Path("work/验证/表体滑动窗口V1")
CASES = (
    ("d8b59365", 0, "原始d8"),
    ("0f372a06", 0, "稀疏对照0f"),
    ("1829aea8", 0, "密集对照1829"),
    ("man/process1.jpg", None, "人工切图"),
)


def source_for(name):
    if name == "man/process1.jpg":
        return Path(name)
    matches = sorted(RAW_ROOT.rglob(f"{name}*.jpg"))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def split_or_manual(source, region_index):
    if region_index is None:
        with Image.open(source) as image:
            return image.convert("RGB")
    regions, _ = v2.BASE.split_boxes(source, v2.SPLITTER)
    with Image.open(source) as image:
        return image.convert("RGB").crop(regions[region_index])


def make_column_mask(image):
    analysis = v2.BASE.half_image(image)
    gray = np.asarray(analysis.convert("L"))
    ink = gray < 225
    _, ratio = v2.adaptive_ratios(float(ink.mean()))
    black = v2.black_with_continuity(analysis)
    erased = v2.BASE.erase_lines(
        black["ink"], black["rows"], black["columns"]
    )
    kernel = max(3, round(analysis.height * ratio))
    mask = cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel)),
    ).astype(bool)
    return analysis, mask, ratio, black


def raw_bands(profile):
    return [
        (a, b)
        for a, b in v2.BASE.runs(profile <= 0.01)
        if a > 0 and b < len(profile)
    ]


def first_stable_threshold(bands):
    counts = {}
    for threshold in range(1, 21):
        counts[threshold] = sum(
            end - start >= threshold for start, end in bands
        )
        # 已达到第一个三连稳定区间，后面的阈值不再计算。
        if threshold >= 3:
            values = [
                counts[threshold - 2],
                counts[threshold - 1],
                counts[threshold],
            ]
            if values[0] > 0 and values[0] == values[1] == values[2]:
                return threshold - 2, values[0], counts
    return None, None, counts


def window_result(mask, start, end):
    bands = raw_bands(mask[start:end].mean(axis=0))
    threshold, count, counts = first_stable_threshold(bands)
    if threshold is None:
        return {
            "start": start,
            "end": end,
            "stable": False,
            "threshold": None,
            "band_count": 0,
            "bands": [],
            "counts_until_stop": counts,
        }
    chosen = [
        band for band in bands if band[1] - band[0] >= threshold
    ]
    return {
        "start": start,
        "end": end,
        "stable": True,
        "threshold": threshold,
        "band_count": count,
        "bands": [list(item) for item in chosen],
        "counts_until_stop": counts,
    }


def compatible(previous, current):
    """判断相邻窗口是否仍属于同一个表体。

    标题/表头边缘常会多出几条很短的候选白带，因此不能要求两块的
    数量完全相同。这里保留主体列的位置作为主依据：数量最多相差4条，
    且较少的一侧至少有80%的中心位置能在另一侧找到对应列。
    """
    if not previous["stable"] or not current["stable"]:
        return False
    if abs(previous["band_count"] - current["band_count"]) > 4:
        return False
    left = np.asarray(
        [item[0] + (item[1] - item[0]) / 2 for item in previous["bands"]]
    )
    right = np.asarray(
        [item[0] + (item[1] - item[0]) / 2 for item in current["bands"]]
    )
    if left.size == 0 or right.size == 0:
        return False
    smaller, larger = (
        (left, right) if left.size <= right.size else (right, left)
    )
    spacing = np.median(np.diff(smaller)) if smaller.size >= 3 else 20
    tolerance = max(8.0, float(spacing) * 0.25)
    matched = sum(
        np.min(np.abs(larger - value)) <= tolerance for value in smaller
    )
    return matched / smaller.size >= 0.80


def choose_body_windows(window_results):
    """选连续最长稳定段；允许少量边缘候选列波动。"""

    best = (0, 0)
    start = None
    for index, current in enumerate(window_results):
        if start is None:
            start = index if current["stable"] else None
            continue
        if compatible(window_results[index - 1], current):
            continue
        if index - start > best[1] - best[0]:
            best = (start, index)
        start = index if current["stable"] else None
    if start is not None and len(window_results) - start > best[1] - best[0]:
        best = (start, len(window_results))
    if best[1] - best[0] < 3:
        return None
    return best


def draw_windows(image, results, selected):
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    selected_indices = (
        set(range(selected[0], selected[1])) if selected else set()
    )
    for index, item in enumerate(results):
        color = (0, 180, 0) if index in selected_indices else (220, 60, 60)
        draw.rectangle(
            (0, item["start"], output.width - 1, item["end"] - 1),
            outline=color,
            width=5,
        )
        draw.text(
            (10, item["start"] + 5),
            f"window={index} bands={item['band_count']} "
            f"stable={item['stable']}",
            fill=color,
        )
    return output


def run_case(name, region_index, label):
    source = source_for(name)
    region = split_or_manual(source, region_index)
    analysis, mask, ratio, black = make_column_mask(region)
    height = analysis.height
    window_height = min(height, max(120, round(height * 0.18)))
    step = max(20, round(height * 0.05))
    starts = list(range(0, max(1, height - window_height + 1), step))
    if starts and starts[-1] != height - window_height:
        starts.append(height - window_height)
    results = [
        window_result(mask, start, start + window_height)
        for start in starts
    ]
    selected = choose_body_windows(results)
    if selected is None:
        body_box = None
    else:
        body_box = (
            0,
            results[selected[0]]["start"],
            analysis.width,
            results[selected[1] - 1]["end"],
        )

    out = OUTPUT / label
    out.mkdir(parents=True, exist_ok=True)
    analysis.save(out / "01_整张子表50.png")
    v2.BASE.mask_image(mask).save(out / "02_列方向晕染.png")
    draw_windows(analysis, results, selected).save(
        out / "03_窗口稳定性.png"
    )
    if body_box is not None:
        analysis.crop(body_box).save(out / "04_选中的表体.png")

    report = {
        "source": str(source.resolve()),
        "label": label,
        "region_index": region_index,
        "analysis_size": list(analysis.size),
        "column_dilate_ratio": ratio,
        "window_height": window_height,
        "window_step": step,
        "stable_repeat_count": 3,
        "formal_maximum_width": 20,
        "windows": results,
        "selected_window_range": None if selected is None else list(selected),
        "selected_body_box": None if body_box is None else list(body_box),
        "fatal": selected is None,
    }
    (out / "窗口诊断.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        label,
        "windows=", len(results),
        "selected=", selected,
        "body=", body_box,
        "fatal=", selected is None,
        flush=True,
    )
    return report


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = [
        run_case(name, region, label)
        for name, region, label in CASES
    ]
    (OUTPUT / "汇总.json").write_text(
        json.dumps({"results": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
