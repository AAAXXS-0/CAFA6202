"""横向分表 V2 实验。

在 V1 的基础上只改“上下分表”判定：

* 先把被短标题/表头打断的相邻白缝合并；
* 放宽白缝宽度门槛，但最终仍要求两侧是足够大的主体；
* 对没有纯白行的情况，增加低密度谷值作为候选。

本文件仍是独立实验，不接入正式流水线。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

V1_PATH = Path(__file__).with_name("001_横向分表V1.py")
spec = importlib.util.spec_from_file_location("horizontal_v1", V1_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载 V1：{V1_PATH}")
v1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v1
spec.loader.exec_module(v1)


@dataclass(frozen=True)
class V2Config(v1.SplitConfig):
    """V2 参数；数值集中放在这里，方便下一轮调参。"""

    # V1 的 2 倍过严；普通行间小缝会在主体复核阶段被过滤。
    separator_width_multiplier: float = 1.0
    minimum_separator_width: int = 3
    # 两条白缝中间允许存在一段很窄的表头墨迹。
    bridge_max_height_ratio: float = 0.02
    bridge_side_min_width: int = 2
    # 没有纯白行时，只把低密度谷值交给后面的主体复核。
    soft_gap_density: float = 0.05
    soft_gap_max_width: int = 8


def _merge_gap_fragments(
    raw_gaps: list[tuple[int, int]],
    row_density: np.ndarray,
    config: V2Config,
) -> list[tuple[int, int]]:
    """合并被窄表头打断的两段白缝。"""

    if not raw_gaps:
        return []
    max_bridge = max(2, round(len(row_density) * config.bridge_max_height_ratio))
    typical_side = float(np.median([end - start for start, end in raw_gaps]))
    side_min = max(config.bridge_side_min_width, round(typical_side * 0.75))
    merged: list[tuple[int, int]] = []
    index = 0
    while index < len(raw_gaps):
        start, end = raw_gaps[index]
        while index + 1 < len(raw_gaps):
            next_start, next_end = raw_gaps[index + 1]
            bridge_start, bridge_end = end, next_start
            bridge_width = bridge_end - bridge_start
            left_width = end - start
            right_width = next_end - next_start
            bridge_density = (
                float(row_density[bridge_start:bridge_end].max())
                if bridge_width
                else 0.0
            )
            if (
                len(raw_gaps) <= 12
                and typical_side >= 4.0
                and left_width >= side_min
                and right_width >= side_min
                and 0 < bridge_width <= max_bridge
                # 不用墨迹浓度判断表头，避免粗体标题把白缝再次打断。
                and bridge_density >= 0.0
            ):
                end = next_end
                index += 1
            else:
                break
        merged.append((start, end))
        index += 1
    return merged


def _soft_gap_ranges(
    row_density: np.ndarray,
    hard_content: np.ndarray,
    config: V2Config,
) -> list[tuple[int, int]]:
    """找被表头连接时仍然存在的窄低密度谷值。"""

    soft_blank = (row_density <= config.soft_gap_density) & hard_content
    runs = v1.boolean_runs(soft_blank)
    return [
        (start, end)
        for start, end in runs
        if end - start <= config.soft_gap_max_width
    ]


def _normalize_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1]:
            previous_start, previous_end = normalized[-1]
            normalized[-1] = (previous_start, max(previous_end, end))
        else:
            normalized.append((start, end))
    return normalized

def _safe_cut_position(
    gap: tuple[int, int],
    raw_gaps: list[tuple[int, int]],
) -> int:
    # 表头打断白缝时从上方白缝落刀，让表头完整归入下表。
    contained = [
        raw
        for raw in raw_gaps
        if gap[0] <= raw[0] and raw[1] <= gap[1]
    ]
    selected = contained[0] if len(contained) >= 2 else gap
    return round((selected[0] + selected[1] - 1) / 2)

def detect_horizontal_tables_v2(
    preview: Image.Image,
    config: V2Config,
    *,
    image_name: str,
    expected_count: int | None,
    original_size: tuple[int, int],
):
    gray = np.asarray(preview.convert("L"))
    ink = (gray < config.gray_threshold).astype(np.uint8)
    smear_width = max(3, round(preview.width * config.horizontal_smear_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (smear_width, 1))
    smeared = cv2.dilate(ink, kernel)
    row_density = smeared.mean(axis=1)
    hard_content = row_density > config.blank_row_ratio

    content_runs = v1.boolean_runs(hard_content)
    if not content_runs:
        result = v1.SplitResult(
            image_name=image_name,
            original_size=original_size,
            analysis_size=preview.size,
            smear_width=smear_width,
            typical_inner_gap=0.0,
            minimum_separator_width=config.minimum_separator_width,
            minimum_body_height=config.body_min_height_pixels,
            minimum_body_active_rows=config.body_min_active_rows,
            content_start=0,
            content_end=preview.height,
            gaps=(),
            segments=(),
            split_boxes=((0, 0, preview.width, preview.height),),
            expected_count=expected_count,
        )
        return result, {
            "gray": gray, "ink": ink, "smeared": smeared,
            "row_density": row_density, "content_rows": hard_content,
            "raw_gaps": [], "candidate_gaps": [], "final_gaps": [],
        }

    content_start, content_end = content_runs[0][0], content_runs[-1][1]
    raw_gaps = [
        (start, end)
        for start, end in v1.boolean_runs(~hard_content)
        if content_start < start and end < content_end
    ]
    merged_gaps = _merge_gap_fragments(raw_gaps, row_density, config)
    soft_gaps = _soft_gap_ranges(row_density, hard_content, config)
    enabled_soft_gaps = [
        gap for gap in soft_gaps
        if len(raw_gaps) > 12 and content_start < gap[0] and gap[1] < content_end
    ]
    all_gaps = _normalize_ranges(merged_gaps + enabled_soft_gaps)
    base_gaps = merged_gaps or enabled_soft_gaps
    widths = [end - start for start, end in base_gaps]
    typical_gap = float(np.median(widths)) if widths else 0.0
    relaxed = len(raw_gaps) <= 12 and typical_gap >= 5.0
    width_multiplier = 1.0 if relaxed else 2.0
    minimum_separator = max(
        config.minimum_separator_width,
        round(typical_gap * width_multiplier),
    )
    candidate_ranges = [
        gap for gap in all_gaps if gap[1] - gap[0] >= minimum_separator
    ]

    edges = [content_start, *[v for gap in candidate_ranges for v in gap], content_end]
    segment_ranges = [
        (edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)
    ]
    minimum_body_height = max(
        config.body_min_height_pixels,
        round(preview.height * config.body_min_height_ratio),
    )
    minimum_body_active_rows = max(
        config.body_min_active_rows, minimum_body_height // 3,
    )
    raw_segments = []
    for start, end in segment_ranges:
        height = end - start
        active_rows = int(hard_content[start:end].sum())
        is_body = height >= minimum_body_height and active_rows >= minimum_body_active_rows
        raw_segments.append((start, end, height, active_rows, is_body))

    owners = v1._segment_owners([item[4] for item in raw_segments])
    final_ranges = [
        candidate_ranges[i]
        for i in range(len(candidate_ranges))
        if owners[i] is not None
        and owners[i + 1] is not None
        and owners[i] != owners[i + 1]
    ]
    # 主体判断后再复核一次：小残表可能只因高度差几像素而被当成页脚。
    # 只有小块的横向墨迹密度接近相邻大表时才强制切，稀疏页脚不会通过。
    forced_ranges = []
    for index, gap in enumerate(candidate_ranges):
        if gap in final_ranges:
            continue
        left = raw_segments[index]
        right = raw_segments[index + 1]
        pairs = ((left, right), (right, left))
        for small, body in pairs:
            if small[4] or not body[4]:
                continue
            small_density = float(row_density[small[0]:small[1]].mean())
            body_density = float(row_density[body[0]:body[1]].mean())
            enough_height = small[2] >= max(
                config.body_min_height_pixels,
                round(minimum_body_height * 0.5),
            )
            enough_ink = small[3] >= minimum_body_active_rows
            table_like_density = (
                small_density >= 0.15
                and small_density >= body_density * 0.55
            )
            if enough_height and enough_ink and table_like_density:
                final_ranges.append(gap)
                forced_ranges.append(gap)
                break
    final_ranges.sort()
    final_set = set(final_ranges)
    forced_set = set(forced_ranges)
    candidate_set = set(candidate_ranges)
    gaps = []
    for start, end in all_gaps:
        interval = (start, end)
        is_candidate = interval in candidate_set
        is_final = interval in final_set
        if is_final and interval in forced_set:
            reason = "V2：候选白缝二次复核后切出小残表"
        elif is_final:
            reason = "V2：白缝两侧是不同表格主体"
        elif is_candidate:
            reason = "V2：候选但主体归属相同，保留不切"
        elif interval in set(soft_gaps):
            reason = "V2：低密度谷值，宽度或主体条件不足"
        else:
            reason = "V2：普通短白缝"
        gaps.append(v1.Gap(start, end, end - start, is_candidate, is_final, reason))

    segments = tuple(
        v1.Segment(start, end, height, active_rows, is_body, owner)
        for (start, end, height, active_rows, is_body), owner
        in zip(raw_segments, owners)
    )
    cut_positions = [
        _safe_cut_position(gap, raw_gaps)
        for gap in final_ranges
    ]
    boundaries = [content_start, *cut_positions, content_end]
    split_boxes = tuple(
        (0, y1, preview.width, y2)
        for y1, y2 in zip(boundaries, boundaries[1:]) if y2 > y1
    )
    result = v1.SplitResult(
        image_name=image_name,
        original_size=original_size,
        analysis_size=preview.size,
        smear_width=smear_width,
        typical_inner_gap=typical_gap,
        minimum_separator_width=minimum_separator,
        minimum_body_height=minimum_body_height,
        minimum_body_active_rows=minimum_body_active_rows,
        content_start=content_start,
        content_end=content_end,
        gaps=tuple(gaps),
        segments=segments,
        split_boxes=split_boxes,
        expected_count=expected_count,
    )
    return result, {
        "gray": gray, "ink": ink, "smeared": smeared,
        "row_density": row_density, "content_rows": hard_content,
        "raw_gaps": raw_gaps, "merged_gaps": merged_gaps,
        "soft_gaps": soft_gaps, "candidate_gaps": candidate_ranges,
        "final_gaps": final_ranges,
        "forced_gaps": forced_ranges,
        "cut_positions": cut_positions,
    }


def final_split_visualization_v2(preview: Image.Image, result):
    overlay = preview.copy()
    draw = ImageDraw.Draw(overlay)
    for box in result.split_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 80, 255), width=4)
    for box in result.split_boxes[1:]:
        y = box[1]
        draw.line((0, y, preview.width - 1, y), fill=(255, 0, 0), width=3)
    return overlay

def save_case_v2(image_path: Path, output_dir: Path, config: V2Config):
    """复用 V1 的九张中间图保存逻辑，但使用 V2 检测函数。"""
    original = Image.open(image_path).convert("RGB")
    preview = v1.make_analysis_image(image_path, config)
    result, debug = detect_horizontal_tables_v2(
        preview, config, image_name=image_path.name,
        expected_count=None, original_size=original.size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preview.save(output_dir / "01_原始20%分析图.png")
    v1.mask_image(debug["ink"]).save(output_dir / "02_灰度225墨迹图.png")
    v1.mask_image(debug["smeared"]).save(output_dir / "03_横向晕染图.png")
    v1.density_chart(debug["row_density"], config.blank_row_ratio).save(output_dir / "04_逐行密度曲线.png")
    v1.gap_overlay(preview, debug["raw_gaps"], color=(0,180,255,70), label_prefix="raw-").save(output_dir / "05_全部原始白缝.png")
    v1.gap_overlay(preview, debug["merged_gaps"], color=(0,180,255,70), label_prefix="merged-").save(output_dir / "06_V2合并后白缝.png")
    v1.gap_overlay(preview, debug["candidate_gaps"], color=(255,0,0,70), label_prefix="candidate-").save(output_dir / "07_V2候选白缝.png")
    v1.segment_ownership_visualization(preview, result).save(output_dir / "08_主体归属.png")
    final_split_visualization_v2(preview, result).save(output_dir / "09_V2最终分表框.png")
    pieces = output_dir / "实际切出的表格"
    pieces.mkdir(exist_ok=True)
    for i, box in enumerate(result.split_boxes):
        preview.crop(box).save(pieces / f"table_{i:03d}_20percent.png")
    (output_dir / "分表参数与所有白缝数据.json").write_text(
        json.dumps({"config": asdict(config), "result": result.to_dict(), "debug": {k: v for k, v in debug.items() if k.endswith('gaps')}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


TEST_PREFIXES = ("d1752e16", "deb8de95", "3792d522", "4c2588e5", "e4e9de30", "8a150db7", "9212016c")


def main() -> int:
    parser = argparse.ArgumentParser(description="测试横向分表 V2")
    parser.add_argument("--raw-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--output", type=Path, default=Path("work/验证/横向分表V2_问题图"))
    args = parser.parse_args()
    config = V2Config()
    results = []
    for prefix in TEST_PREFIXES:
        source = v1.find_source_image(args.raw_root, prefix)
        print(f"[V2] {source.name}")
        results.append(save_case_v2(source, args.output / prefix, config))
        print(f"  -> blocks={results[-1].actual_count}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "汇总.json").write_text(
        json.dumps({"config": asdict(config), "results": [r.to_dict() for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
