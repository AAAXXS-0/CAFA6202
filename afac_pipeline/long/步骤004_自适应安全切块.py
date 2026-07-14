"""根据全图版面保护框和原图墨水投影生成最终 VLM 安全切块。

检测窗口仍保持固定尺寸，本模块只负责最终发送给 FinixDoc-VL 的原图裁块。
正常情况下从连续空白带中间无重叠切开；只有找不到安全空白时，才使用少量
物理重叠保护接缝文字。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import LongConfig
from .步骤001_数据定义 import DetectionWindow, LayoutBlock, SafeCutChunk
from ..common.models import Box


@dataclass(frozen=True)
class BlankBand:
    """原图中连续若干行近似空白的纵向区间。"""

    start_y: int
    end_y: int
    mean_ink_ratio: float

    @property
    def height(self) -> int:
        return self.end_y - self.start_y

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def build_row_ink_projection(
    window_paths: list[Path],
    windows: list[DetectionWindow],
    image_height: int,
    *,
    sample_width: int,
    white_threshold: int,
) -> list[float]:
    """利用检测窗口的责任区计算原图逐行墨水比例。

    直接再次解码十万像素高的原图会增加大量内存。检测窗口已经完整覆盖原图，
    因此这里只读取每个窗口独占的 ownership 区域，每个全局像素行恰好统计一次。
    横向缩到 sample_width 后再计算深色像素比例，可以显著减少计算量。
    """

    if len(window_paths) != len(windows):
        raise ValueError("检测窗口图片和窗口元数据数量不一致")
    if image_height <= 0:
        raise ValueError("image_height 必须大于 0")

    projection = np.ones(image_height, dtype=np.float32)
    covered = np.zeros(image_height, dtype=np.bool_)
    for path, window in zip(window_paths, windows):
        global_start = max(0, window.ownership_start_y)
        global_end = min(image_height, window.ownership_end_y)
        if global_end <= global_start:
            continue
        local_start = global_start - window.start_y
        local_end = local_start + (global_end - global_start)
        with Image.open(path) as source:
            gray = source.convert("L").crop((0, local_start, source.width, local_end))
            target_width = min(sample_width, gray.width)
            if target_width != gray.width:
                gray = gray.resize(
                    (target_width, gray.height),
                    Image.Resampling.BOX,
                )
            pixels = np.asarray(gray, dtype=np.uint8)
        ratios = np.mean(pixels < white_threshold, axis=1, dtype=np.float32)
        projection[global_start:global_end] = ratios
        covered[global_start:global_end] = True

    if not bool(np.all(covered)):
        missing = np.flatnonzero(~covered)
        raise RuntimeError(
            f"检测窗口责任区没有覆盖原图全部行，首个缺口位于 y={int(missing[0])}"
        )
    return projection.tolist()


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def build_protection_intervals(
    blocks: list[LayoutBlock],
    image_height: int,
    *,
    minimum_confidence: float,
    padding: int,
) -> list[tuple[int, int]]:
    """将所有可信版面框转换成最终切割时不可穿越的纵向区间。"""

    intervals = [
        (
            max(0, block.box.y1 - padding),
            min(image_height, block.box.y2 + padding),
        )
        for block in blocks
        if block.confidence >= minimum_confidence
    ]
    return _merge_intervals(
        [(start, end) for start, end in intervals if end > start]
    )


def find_blank_bands(
    projection: list[float],
    *,
    blank_ratio: float,
    minimum_height: int,
) -> list[BlankBand]:
    """从逐行墨水比例中找出足够高的连续空白带。"""

    bands: list[BlankBand] = []
    start: int | None = None
    for y, ratio in enumerate([*projection, blank_ratio + 1.0]):
        if ratio <= blank_ratio:
            if start is None:
                start = y
            continue
        if start is not None and y - start >= minimum_height:
            mean_ratio = float(sum(projection[start:y]) / (y - start))
            bands.append(BlankBand(start, y, mean_ratio))
        start = None
    return bands


def _is_protected(y: int, intervals: list[tuple[int, int]]) -> bool:
    for start, end in intervals:
        if y < start:
            return False
        if start <= y < end:
            return True
    return False


def _nearest_unprotected(
    preferred: int,
    lower: int,
    upper: int,
    intervals: list[tuple[int, int]],
) -> int | None:
    """在闭区间内寻找离 preferred 最近且不在保护区中的像素行。"""

    if lower > upper:
        return None
    preferred = min(max(preferred, lower), upper)
    maximum_distance = max(preferred - lower, upper - preferred)
    for distance in range(maximum_distance + 1):
        left = preferred - distance
        if left >= lower and not _is_protected(left, intervals):
            return left
        right = preferred + distance
        if distance and right <= upper and not _is_protected(right, intervals):
            return right
    return None


def _select_blank_boundary(
    bands: list[BlankBand],
    intervals: list[tuple[int, int]],
    projection: list[float],
    *,
    lower: int,
    upper: int,
    target: int,
    blank_ratio: float,
    search_radius: int,
) -> tuple[int, float] | None:
    best: tuple[float, int, float] | None = None
    distance_scale = max(1, search_radius)
    for band in bands:
        band_lower = max(lower, band.start_y)
        band_upper = min(upper, band.end_y - 1)
        if band_lower > band_upper:
            continue
        candidate = _nearest_unprotected(
            target,
            band_lower,
            band_upper,
            intervals,
        )
        if candidate is None:
            continue
        whiteness = 1.0 - min(1.0, band.mean_ink_ratio / max(blank_ratio, 1e-6))
        band_score = min(1.0, band.height / 64.0)
        distance_penalty = abs(candidate - target) / distance_scale
        score = 3.0 * whiteness + 1.5 * band_score - distance_penalty
        row_ratio = float(projection[candidate])
        current = (score, candidate, row_ratio)
        if best is None or current[0] > best[0]:
            best = current
    return None if best is None else (best[1], best[2])


def _select_fallback_boundary(
    projection: list[float],
    intervals: list[tuple[int, int]],
    *,
    lower: int,
    upper: int,
    target: int,
) -> tuple[int, float]:
    """没有连续空白带时，选择墨水最少且尽量不穿过保护框的像素行。"""

    if lower > upper:
        raise ValueError("兜底切割区间为空")
    candidates = list(range(lower, upper + 1))
    candidates.sort(
        key=lambda y: (
            _is_protected(y, intervals),
            projection[y],
            abs(y - target),
        )
    )
    boundary = candidates[0]
    return boundary, float(projection[boundary])


def build_adaptive_chunks(
    image_width: int,
    image_height: int,
    projection: list[float],
    blocks: list[LayoutBlock],
    config: LongConfig,
) -> tuple[list[SafeCutChunk], dict[str, object]]:
    """生成覆盖整张长图、且每块不超过 VLM 高度限制的自适应裁块。"""

    if len(projection) != image_height:
        raise ValueError("墨水投影长度必须等于原图高度")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("原图尺寸必须大于 0")

    protection = build_protection_intervals(
        blocks,
        image_height,
        minimum_confidence=config.cut_protection_confidence,
        padding=config.cut_protection_padding,
    )
    blank_bands = find_blank_bands(
        projection,
        blank_ratio=config.projection_blank_ratio,
        minimum_height=config.minimum_blank_band,
    )

    # 先保存原始范围，最后统一计算每块与前后块的真实重叠高度。
    records: list[tuple[int, int, str, int | None, float | None]] = []
    cursor = 0
    while image_height - cursor > config.max_vlm_height:
        remaining_height = image_height - cursor
        # 当剩余高度略高于上限时，不可能同时得到两个 2200 高的块。
        # 此时把最小高度降到剩余区域的一半，优先保证不超 4096 且无遗漏。
        effective_min_height = min(
            config.adaptive_min_height,
            remaining_height // 2,
        )
        target = min(
            cursor + config.adaptive_target_height,
            cursor + config.max_vlm_height,
        )
        lower = cursor + effective_min_height
        upper = min(
            cursor + config.max_vlm_height,
            image_height - effective_min_height,
        )
        if lower > upper:
            break

        search_lower = max(lower, target - config.safe_cut_search)
        search_upper = min(upper, target + config.safe_cut_search)
        selected = _select_blank_boundary(
            blank_bands,
            protection,
            projection,
            lower=search_lower,
            upper=search_upper,
            target=target,
            blank_ratio=config.projection_blank_ratio,
            search_radius=config.safe_cut_search,
        )
        # 目标附近没有空白时，再检查当前合法高度范围内是否存在稍远的空白带。
        if selected is None:
            selected = _select_blank_boundary(
                blank_bands,
                protection,
                projection,
                lower=lower,
                upper=upper,
                target=target,
                blank_ratio=config.projection_blank_ratio,
                search_radius=max(config.safe_cut_search, upper - lower),
            )

        if selected is not None:
            boundary, ink_ratio = selected
            end_y = boundary
            next_cursor = boundary
            method = "blank_band"
        else:
            half_overlap = config.vlm_overlap // 2
            fallback_lower = max(
                cursor + effective_min_height - half_overlap,
                cursor + 1,
            )
            fallback_upper = min(
                cursor + config.max_vlm_height - half_overlap,
                image_height - effective_min_height + half_overlap,
            )
            fallback_target = min(
                cursor + config.adaptive_target_height,
                fallback_upper,
            )
            boundary, ink_ratio = _select_fallback_boundary(
                projection,
                protection,
                lower=fallback_lower,
                upper=fallback_upper,
                target=fallback_target,
            )
            end_y = min(image_height, boundary + half_overlap)
            next_cursor = max(cursor + 1, boundary - half_overlap)
            method = "fallback_overlap"

        if end_y <= cursor or next_cursor <= cursor:
            raise RuntimeError("自适应切块没有向下推进")
        if end_y - cursor > config.max_vlm_height:
            raise RuntimeError("自适应切块超过 max_vlm_height")
        records.append((cursor, end_y, method, boundary, ink_ratio))
        cursor = next_cursor

    if cursor < image_height:
        records.append((cursor, image_height, "document_end", None, None))

    chunks: list[SafeCutChunk] = []
    for index, (start_y, end_y, method, boundary, ink_ratio) in enumerate(records):
        previous_end = records[index - 1][1] if index else start_y
        next_start = records[index + 1][0] if index + 1 < len(records) else end_y
        chunks.append(
            SafeCutChunk(
                id=f"adaptive_chunk_{index:05d}",
                index=index,
                source_box=Box(0, start_y, image_width, end_y),
                cut_method=method,
                boundary_y=boundary,
                boundary_ink_ratio=ink_ratio,
                overlap_top=max(0, previous_end - start_y),
                overlap_bottom=max(0, end_y - next_start),
            )
        )

    debug: dict[str, object] = {
        "projection": {
            "sample_width": config.projection_sample_width,
            "white_threshold": config.projection_white_threshold,
            "blank_ratio": config.projection_blank_ratio,
            "minimum_blank_band": config.minimum_blank_band,
        },
        "protection_intervals": [
            {"start_y": start, "end_y": end} for start, end in protection
        ],
        "blank_bands": [band.to_dict() for band in blank_bands],
        "safe_cut_count": sum(chunk.cut_method == "blank_band" for chunk in chunks),
        "fallback_overlap_count": sum(
            chunk.cut_method == "fallback_overlap" for chunk in chunks
        ),
    }
    return chunks, debug
