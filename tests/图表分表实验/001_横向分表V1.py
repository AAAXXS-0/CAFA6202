"""只测试“同图上下分表”的独立实验，不接入正式图表流水线。

目标：
1. 固定使用原图20%分析图，避免5%图把真实表间白缝压成1像素。
2. 只允许水平分割线，永远不产生左右分表。
3. 横向晕染只连接同一行的分散文字，不在竖直方向扩张墨迹。
4. 已知答案只用于最终断言，不参与任何图片的分表判断。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class SplitConfig:
    """横向分表V1的全部公共参数。"""

    analysis_scale: float = 0.20
    gray_threshold: int = 225
    horizontal_smear_ratio: float = 0.01
    blank_row_ratio: float = 0.01
    separator_width_multiplier: float = 2.0
    minimum_separator_width: int = 3
    body_min_height_ratio: float = 0.02
    body_min_height_pixels: int = 8
    body_min_active_rows: int = 6


@dataclass(frozen=True)
class Gap:
    start: int
    end: int
    width: int
    is_candidate: bool
    is_final: bool
    reason: str

    @property
    def center(self) -> int:
        return round((self.start + self.end - 1) / 2)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    height: int
    active_rows: int
    is_table_body: bool
    owner: int | None


@dataclass(frozen=True)
class SplitResult:
    image_name: str
    original_size: tuple[int, int]
    analysis_size: tuple[int, int]
    smear_width: int
    typical_inner_gap: float
    minimum_separator_width: int
    minimum_body_height: int
    minimum_body_active_rows: int
    content_start: int
    content_end: int
    gaps: tuple[Gap, ...]
    segments: tuple[Segment, ...]
    split_boxes: tuple[tuple[int, int, int, int], ...]
    expected_count: int | None

    @property
    def actual_count(self) -> int:
        return len(self.split_boxes)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["actual_count"] = self.actual_count
        value["matched_expected"] = (
            None
            if self.expected_count is None
            else self.actual_count == self.expected_count
        )
        return value


# 这些答案只在脚本末尾做验收，不会传给 detect_horizontal_tables。
KNOWN_EXPECTED_COUNTS: dict[str, int] = {
    "0cd74f08": 3,
    "0f372a06": 3,
    "1829aea8": 1,
    "3bfd625b": 7,
    "b1044f0e": 1,
    "185a2337": 3,
    "5b93ec6f": 3,
    "d8b59365": 1,
}

# afa837只要求禁止竖切；当前横向白缝实验得到一个完整表，暂不硬写数量断言。
EXTRA_PREFIXES = ("afa837b9",)


def boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """把一维布尔数组转成左闭右开的连续区间。"""

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks] + 1, indices[-1] + 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def make_analysis_image(image_path: Path, config: SplitConfig) -> Image.Image:
    """按固定20%生成RGB分析图。"""

    with Image.open(image_path) as source:
        size = (
            max(1, round(source.width * config.analysis_scale)),
            max(1, round(source.height * config.analysis_scale)),
        )
        return source.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _segment_owners(table_flags: list[bool]) -> list[int | None]:
    """标题优先归给下面主体，末尾页码和注释归给上面主体。"""

    body_indices = [index for index, value in enumerate(table_flags) if value]
    if not body_indices:
        return [0 for _ in table_flags]
    owners: list[int | None] = []
    for index, is_body in enumerate(table_flags):
        if is_body:
            owners.append(index)
            continue
        next_body = next((item for item in body_indices if item > index), None)
        previous_body = next(
            (item for item in reversed(body_indices) if item < index),
            None,
        )
        owners.append(next_body if next_body is not None else previous_body)
    return owners


def detect_horizontal_tables(
    preview: Image.Image,
    config: SplitConfig,
    *,
    image_name: str,
    expected_count: int | None,
    original_size: tuple[int, int],
) -> tuple[SplitResult, dict[str, object]]:
    """只依据水平白缝返回上下排列的表格区域。"""

    gray = np.asarray(preview.convert("L"))
    ink = (gray < config.gray_threshold).astype(np.uint8)

    # 只向左右扩张文字；核高度固定为1，不会把上下表格糊在一起。
    smear_width = max(3, round(preview.width * config.horizontal_smear_ratio))
    smear_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (smear_width, 1),
    )
    smeared = cv2.dilate(ink, smear_kernel)

    row_density = smeared.mean(axis=1)
    content_rows = row_density > config.blank_row_ratio
    content_runs = boolean_runs(content_rows)
    if not content_runs:
        result = SplitResult(
            image_name=image_name,
            original_size=preview.size,
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
            "gray": gray,
            "ink": ink,
            "smeared": smeared,
            "row_density": row_density,
            "content_rows": content_rows,
            "raw_gaps": [],
            "candidate_gaps": [],
            "final_gaps": [],
        }

    content_start = content_runs[0][0]
    content_end = content_runs[-1][1]
    raw_gap_ranges = [
        (start, end)
        for start, end in boolean_runs(~content_rows)
        if content_start < start and end < content_end
    ]
    gap_widths = [end - start for start, end in raw_gap_ranges]
    typical_gap = float(np.median(gap_widths)) if gap_widths else 0.0
    minimum_separator = max(
        config.minimum_separator_width,
        round(typical_gap * config.separator_width_multiplier),
    )
    candidate_ranges = [
        (start, end)
        for start, end in raw_gap_ranges
        if end - start >= minimum_separator
    ]

    # 候选白缝先把内容切成若干段；此时尚未决定哪些白缝真正落刀。
    edges = [
        content_start,
        *[
            value
            for start, end in candidate_ranges
            for value in (start, end)
        ],
        content_end,
    ]
    segment_ranges = [
        (edges[index], edges[index + 1])
        for index in range(0, len(edges), 2)
    ]

    minimum_body_height = max(
        config.body_min_height_pixels,
        round(preview.height * config.body_min_height_ratio),
    )
    minimum_body_active_rows = max(
        config.body_min_active_rows,
        minimum_body_height // 3,
    )
    raw_segments: list[tuple[int, int, int, int, bool]] = []
    for start, end in segment_ranges:
        height = end - start
        active_rows = int(content_rows[start:end].sum())
        is_body = (
            height >= minimum_body_height
            and active_rows >= minimum_body_active_rows
        )
        raw_segments.append((start, end, height, active_rows, is_body))

    owners = _segment_owners([item[4] for item in raw_segments])
    final_ranges = [
        candidate_ranges[index]
        for index in range(len(candidate_ranges))
        if owners[index] is not None
        and owners[index + 1] is not None
        and owners[index] != owners[index + 1]
    ]
    final_set = set(final_ranges)
    candidate_set = set(candidate_ranges)

    gaps: list[Gap] = []
    for start, end in raw_gap_ranges:
        interval = (start, end)
        is_candidate = interval in candidate_set
        is_final = interval in final_set
        if is_final:
            reason = "白缝两侧属于不同表格主体"
        elif is_candidate:
            reason = "标题、页码或注释与相邻主体属于同一张表"
        else:
            reason = "宽度接近本图典型表内行距，按短白缝填平"
        gaps.append(
            Gap(
                start=start,
                end=end,
                width=end - start,
                is_candidate=is_candidate,
                is_final=is_final,
                reason=reason,
            )
        )

    segments = tuple(
        Segment(
            start=start,
            end=end,
            height=height,
            active_rows=active_rows,
            is_table_body=is_body,
            owner=owner,
        )
        for (start, end, height, active_rows, is_body), owner in zip(
            raw_segments,
            owners,
        )
    )

    cut_positions = [
        round((start + end - 1) / 2)
        for start, end in final_ranges
    ]
    y_boundaries = [content_start, *cut_positions, content_end]
    split_boxes = tuple(
        (0, y1, preview.width, y2)
        for y1, y2 in zip(y_boundaries, y_boundaries[1:])
        if y2 > y1
    )
    result = SplitResult(
        image_name=image_name,
        original_size=preview.size,
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
        "gray": gray,
        "ink": ink,
        "smeared": smeared,
        "row_density": row_density,
        "content_rows": content_rows,
        "raw_gaps": raw_gap_ranges,
        "candidate_gaps": candidate_ranges,
        "final_gaps": final_ranges,
    }


def mask_image(mask: np.ndarray) -> Image.Image:
    """保存为白底黑墨的二值调试图。"""

    return Image.fromarray(
        np.where(mask, 0, 255).astype(np.uint8),
        mode="L",
    )


def gap_overlay(
    preview: Image.Image,
    gaps: list[tuple[int, int]],
    *,
    color: tuple[int, int, int, int],
    label_prefix: str,
) -> Image.Image:
    overlay = preview.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, (start, end) in enumerate(gaps):
        draw.rectangle(
            (0, start, preview.width - 1, max(start, end - 1)),
            fill=color,
        )
        draw.text(
            (5, start),
            f"{label_prefix}{index + 1}: y={start}:{end}, w={end - start}",
            fill=(0, 0, 0, 255),
        )
    return overlay.convert("RGB")


def density_chart(profile: np.ndarray, threshold: float) -> Image.Image:
    """生成纵向密度折线图；红线是空白判定阈值。"""

    chart_width = 640
    chart = Image.new("RGB", (chart_width, len(profile)), "white")
    draw = ImageDraw.Draw(chart)
    reference = max(
        threshold * 4,
        float(np.quantile(profile, 0.98)),
        1e-6,
    )
    threshold_x = round(min(1.0, threshold / reference) * (chart_width - 1))
    draw.line((threshold_x, 0, threshold_x, len(profile) - 1), fill="red", width=2)
    for y, value in enumerate(profile):
        x = round(min(1.0, float(value) / reference) * (chart_width - 1))
        draw.line((0, y, x, y), fill="black")
    return chart


def filled_short_gap_visualization(
    preview: Image.Image,
    result: SplitResult,
) -> Image.Image:
    """蓝色表示已经按表内短白缝填平，红色表示继续参与分表复核。"""

    overlay = preview.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for gap in result.gaps:
        color = (255, 0, 0, 75) if gap.is_candidate else (0, 100, 255, 75)
        draw.rectangle(
            (0, gap.start, preview.width - 1, max(gap.start, gap.end - 1)),
            fill=color,
        )
    return overlay.convert("RGB")


def segment_ownership_visualization(
    preview: Image.Image,
    result: SplitResult,
) -> Image.Image:
    """绿色为表格主体，橙色为归入相邻主体的标题/页码/注释。"""

    overlay = preview.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, segment in enumerate(result.segments):
        color = (
            (0, 200, 0, 55)
            if segment.is_table_body
            else (255, 140, 0, 80)
        )
        draw.rectangle(
            (0, segment.start, preview.width - 1, max(segment.start, segment.end - 1)),
            fill=color,
        )
        draw.text(
            (5, segment.start + 2),
            (
                f"segment-{index + 1} body={segment.is_table_body} "
                f"owner={segment.owner} active={segment.active_rows}"
            ),
            fill=(0, 0, 0, 255),
        )
    return overlay.convert("RGB")


def final_split_visualization(
    preview: Image.Image,
    result: SplitResult,
) -> Image.Image:
    overlay = preview.copy()
    draw = ImageDraw.Draw(overlay)
    for index, (x1, y1, x2, y2) in enumerate(result.split_boxes):
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 80, 255), width=4)
        draw.text(
            (x1 + 6, y1 + 6),
            f"table-{index + 1}",
            fill=(0, 80, 255),
        )
    for gap in result.gaps:
        if gap.is_final:
            draw.line(
                (0, gap.center, preview.width - 1, gap.center),
                fill=(255, 0, 0),
                width=3,
            )
    return overlay


def save_case(
    image_path: Path,
    output_dir: Path,
    config: SplitConfig,
    expected_count: int | None,
) -> SplitResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = make_analysis_image(image_path, config)
    with Image.open(image_path) as source:
        original_size = source.size
    result, debug = detect_horizontal_tables(
        preview,
        config,
        image_name=image_path.name,
        expected_count=expected_count,
        original_size=original_size,
    )

    preview.save(output_dir / "01_原始20%分析图.png")
    mask_image(debug["ink"]).save(output_dir / "02_灰度225墨迹图.png")
    mask_image(debug["smeared"]).save(output_dir / "03_横向晕染图.png")
    density_chart(
        debug["row_density"],
        config.blank_row_ratio,
    ).save(output_dir / "04_逐行密度曲线.png")
    gap_overlay(
        preview,
        debug["raw_gaps"],
        color=(0, 180, 255, 70),
        label_prefix="gap-",
    ).save(output_dir / "05_全部原始白缝.png")
    filled_short_gap_visualization(
        preview,
        result,
    ).save(output_dir / "06_填平表内短白缝后.png")
    gap_overlay(
        preview,
        debug["candidate_gaps"],
        color=(255, 0, 0, 70),
        label_prefix="candidate-",
    ).save(output_dir / "07_候选表间白缝.png")
    segment_ownership_visualization(
        preview,
        result,
    ).save(output_dir / "08_标题与页脚归属结果.png")
    final_split_visualization(
        preview,
        result,
    ).save(output_dir / "09_最终水平分表框.png")

    pieces_dir = output_dir / "实际切出的表格"
    pieces_dir.mkdir(parents=True, exist_ok=True)
    for index, box in enumerate(result.split_boxes):
        preview.crop(box).save(
            pieces_dir / f"table_{index:03d}_20percent.png"
        )

    with (output_dir / "分表参数与所有白缝数据.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "config": asdict(config),
                "result": result.to_dict(),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return result


def find_source_image(root: Path, prefix: str) -> Path:
    matches = sorted(root.rglob(f"{prefix}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"没有找到测试图片：{prefix}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只测试图表上下分表")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("raw_data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/验证/横向分表V1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SplitConfig()
    prefixes = [*KNOWN_EXPECTED_COUNTS, *EXTRA_PREFIXES]
    results: list[SplitResult] = []
    for index, prefix in enumerate(prefixes, start=1):
        source = find_source_image(args.raw_root, prefix)
        print(f"[横向分表 {index:02d}/{len(prefixes):02d}] {source.name}")
        results.append(
            save_case(
                source,
                args.output / prefix,
                config,
                KNOWN_EXPECTED_COUNTS.get(prefix),
            )
        )

    summary = {
        "config": asdict(config),
        "results": [result.to_dict() for result in results],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "汇总.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    mismatches = [
        result
        for result in results
        if result.expected_count is not None
        and result.actual_count != result.expected_count
    ]
    for result in results:
        expected = (
            "待人工确认"
            if result.expected_count is None
            else str(result.expected_count)
        )
        print(
            f"  {result.image_name[:8]}：实际{result.actual_count}，预期{expected}"
        )
    if mismatches:
        names = ", ".join(item.image_name[:8] for item in mismatches)
        raise AssertionError(f"已知答案不匹配：{names}")
    print(f"横向分表已知答案全部通过，输出：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
