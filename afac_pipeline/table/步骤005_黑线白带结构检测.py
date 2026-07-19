"""图表 v6：固定比例密度分表与黑线优先、白带兜底的结构检测。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..common.models import Box, DetectedBox
from .config import TableConfig
from .步骤002_低密度分表 import DensityBand, _runs, boxes_from_bands, find_density_bands
from .步骤004_网格与白带检测 import (
    GridStructure,
    _whitespace_centers,
    _whitespace_dilate_kernels,
)
from .步骤001_墨水密度定位 import InkRegionResult, detect_ink_regions


@dataclass(frozen=True)
class LineSegment:
    """一条黑线及其在梯形表格包络内的有效跨度。"""

    position: int
    start: int
    end: int


@dataclass(frozen=True)
class RejectedColumnLine:
    """被网格整体规律否决的竖线候选。"""

    line: LineSegment
    reason: str
    left_gap: int
    right_gap: int
    typical_gap: float
    minimum_gap: int

    def to_dict(self) -> dict[str, object]:
        return {
            **self.line.__dict__,
            "reason": self.reason,
            "left_gap": self.left_gap,
            "right_gap": self.right_gap,
            "typical_gap": self.typical_gap,
            "minimum_gap": self.minimum_gap,
        }


@dataclass(frozen=True)
class WhiteColumnBand:
    """一条纵向白带在20%分析图中的完整像素范围。"""

    start: int
    end: int

    @property
    def position(self) -> int:
        return round((self.start + self.end - 1) / 2)

    @property
    def width(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        return {
            "position": self.position,
            "start": self.start,
            "end": self.end,
            "width": self.width,
        }


@dataclass(frozen=True)
class RejectedWhiteColumnBand:
    """因本表自适应列宽复核被删除的纵向白带。"""

    band: WhiteColumnBand
    reason: str
    selected_minimum: int

    def to_dict(self) -> dict[str, object]:
        return {
            **self.band.to_dict(),
            "reason": self.reason,
            "selected_minimum": self.selected_minimum,
        }


@dataclass(frozen=True)
class V6RegionResult:
    """分表阶段的全部结果，既供正式流程使用，也供审计图保存。"""

    ink_result: InkRegionResult
    horizontal_bands: tuple[DensityBand, ...]
    vertical_bands: tuple[DensityBand, ...]
    split_boxes: tuple[Box, ...]
    analysis_boxes: tuple[Box, ...]


@dataclass(frozen=True)
class V6GridDiagnostics:
    """单表边界检测的可审计数据。"""

    row_source: str
    column_source: str
    black_rows: tuple[LineSegment, ...]
    black_columns: tuple[LineSegment, ...]
    white_rows: tuple[int, ...]
    white_columns: tuple[int, ...]
    row_reliability: str = ""
    column_reliability: str = ""
    black_rows_at_whitespace_scale: tuple[LineSegment, ...] = ()
    black_columns_at_whitespace_scale: tuple[LineSegment, ...] = ()
    used_black_columns: tuple[LineSegment, ...] = ()
    rejected_black_columns: tuple[RejectedColumnLine, ...] = ()
    column_cleanup: str = ""
    raw_white_column_bands: tuple[WhiteColumnBand, ...] = ()
    used_white_column_bands: tuple[WhiteColumnBand, ...] = ()
    rejected_white_column_bands: tuple[RejectedWhiteColumnBand, ...] = ()
    white_column_min_band: int = 1
    white_column_regularity_before: float = 0.0
    white_column_regularity_after: float = 0.0
    white_column_cleanup: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "row_source": self.row_source,
            "column_source": self.column_source,
            "black_rows": [line.__dict__ for line in self.black_rows],
            "black_columns": [line.__dict__ for line in self.black_columns],
            "white_rows": list(self.white_rows),
            "white_columns": list(self.white_columns),
            "black_row_count": len(self.black_rows),
            "black_column_count": len(self.black_columns),
            "black_row_count_at_whitespace_scale": len(
                self.black_rows_at_whitespace_scale
            ),
            "black_column_count_at_whitespace_scale": len(
                self.black_columns_at_whitespace_scale
            ),
            "used_black_columns": [
                line.__dict__ for line in self.used_black_columns
            ],
            "used_black_column_count": len(self.used_black_columns),
            "rejected_black_columns": [
                line.to_dict() for line in self.rejected_black_columns
            ],
            "rejected_black_column_count": len(self.rejected_black_columns),
            "column_cleanup": self.column_cleanup,
            "raw_white_column_bands": [
                band.to_dict() for band in self.raw_white_column_bands
            ],
            "used_white_column_bands": [
                band.to_dict() for band in self.used_white_column_bands
            ],
            "rejected_white_column_bands": [
                item.to_dict() for item in self.rejected_white_column_bands
            ],
            "raw_white_column_count": len(self.raw_white_column_bands),
            "used_white_column_count": len(self.used_white_column_bands),
            "rejected_white_column_count": len(
                self.rejected_white_column_bands
            ),
            "white_column_min_band": self.white_column_min_band,
            "white_column_regularity_before": self.white_column_regularity_before,
            "white_column_regularity_after": self.white_column_regularity_after,
            "white_column_cleanup": self.white_column_cleanup,
            "row_reliability": self.row_reliability,
            "column_reliability": self.column_reliability,
            "black_rows_at_whitespace_scale": [
                line.__dict__ for line in self.black_rows_at_whitespace_scale
            ],
            "black_columns_at_whitespace_scale": [
                line.__dict__ for line in self.black_columns_at_whitespace_scale
            ],
        }


def map_box(box: Box, from_size: tuple[int, int], to_size: tuple[int, int]) -> Box:
    """在两个等比例图像坐标系之间映射矩形。"""

    from_width, from_height = from_size
    to_width, to_height = to_size
    return Box(
        round(box.x1 * to_width / from_width),
        round(box.y1 * to_height / from_height),
        round(box.x2 * to_width / from_width),
        round(box.y2 * to_height / from_height),
    ).clamp(to_width, to_height)


def dense_content_box(ink: np.ndarray, projection_ratio: float = 0.01) -> Box:
    """保守删除分表块四周空白，宁可多留也不切掉稀疏边缘数据。"""

    if not ink.any():
        return Box(0, 0, ink.shape[1], ink.shape[0])
    row_ratio = ink.mean(axis=1)
    column_ratio = ink.mean(axis=0)
    rows = np.flatnonzero(
        row_ratio >= max(0.0005, float(row_ratio.max()) * projection_ratio)
    )
    columns = np.flatnonzero(
        column_ratio >= max(0.0005, float(column_ratio.max()) * projection_ratio)
    )
    if rows.size == 0 or columns.size == 0:
        return Box(0, 0, ink.shape[1], ink.shape[0])
    padding = max(4, round(min(ink.shape) * 0.01))
    return Box(
        max(0, int(columns[0]) - padding),
        max(0, int(rows[0]) - padding),
        min(ink.shape[1], int(columns[-1]) + 1 + padding),
        min(ink.shape[0], int(rows[-1]) + 1 + padding),
    )


def detect_v6_regions(preview: Image.Image, config: TableConfig) -> V6RegionResult:
    """在原图 20% 分析图上，以原图 5% 密度图切开同图异表。"""

    ink_result = detect_ink_regions(
        preview,
        coarse_scale=config.table_density_scale,
        ink_threshold=config.ink_threshold,
        minimum_density=config.ink_minimum_density,
        blur_ratio=config.ink_blur_ratio,
        closing_ratio=config.ink_closing_ratio,
        minimum_box_area_ratio=config.ink_minimum_box_area_ratio,
    )
    density = ink_result.coarse_density
    horizontal, vertical = find_density_bands(density)
    coarse_boxes = boxes_from_bands(
        density.shape[1], density.shape[0], horizontal, vertical, density
    )
    if not coarse_boxes:
        coarse_boxes = [Box(0, 0, density.shape[1], density.shape[0])]

    split_boxes: list[Box] = []
    analysis_boxes: list[Box] = []
    gray = np.asarray(preview.convert("L"))
    for coarse_box in coarse_boxes:
        split_box = map_box(
            coarse_box,
            (density.shape[1], density.shape[0]),
            preview.size,
        )
        split_boxes.append(split_box)
        local_ink = (
            gray[split_box.y1 : split_box.y2, split_box.x1 : split_box.x2]
            < config.ink_threshold
        )
        local = dense_content_box(local_ink)
        analysis_boxes.append(
            Box(
                split_box.x1 + local.x1,
                split_box.y1 + local.y1,
                split_box.x1 + local.x2,
                split_box.y1 + local.y2,
            ).clamp(preview.width, preview.height)
        )
    return V6RegionResult(
        ink_result=ink_result,
        horizontal_bands=tuple(horizontal),
        vertical_bands=tuple(vertical),
        split_boxes=tuple(split_boxes),
        analysis_boxes=tuple(analysis_boxes),
    )


def detected_boxes(result: V6RegionResult) -> list[DetectedBox]:
    """转换为正式流水线统一使用的检测框。"""

    return [
        DetectedBox(box, confidence=1.0, source="density-v6")
        for box in result.analysis_boxes
        if box.width > 0 and box.height > 0
    ]


def content_envelope_mask(ink: np.ndarray) -> np.ndarray:
    """估计梯形表格的二维外形，作为 90% 黑线覆盖率的独立分母。"""

    height, width = ink.shape
    blurred = cv2.GaussianBlur(
        ink.astype(np.float32),
        (0, 0),
        sigmaX=max(1.0, width * 0.008),
        sigmaY=max(1.0, height * 0.008),
    )
    mask = (blurred >= 0.015).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            max(3, round(width * 0.015)) | 1,
            max(3, round(height * 0.015)) | 1,
        ),
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)


def adaptive_line_segments(
    black_ink: np.ndarray,
    envelope_mask: np.ndarray,
    axis: int,
    minimum_ratio: float = 0.90,
    minimum_span_ratio: float = 0.20,
    *,
    grayscale: np.ndarray | None = None,
    endpoint_trim_ratio: float = 0.0,
    minimum_contrast: float = 0.0,
    contrast_offset_scale: float = 1.0,
    contrast_bypass_ratio: float | None = None,
) -> list[LineSegment]:
    """按覆盖率找整线，并可用中段覆盖率和邻域对比排除数字竖笔画。"""

    data = black_ink if axis == 0 else black_ink.T
    envelope = envelope_mask if axis == 0 else envelope_mask.T
    gray_data = None if grayscale is None else (grayscale if axis == 0 else grayscale.T)
    scores = np.zeros(len(data), dtype=np.float32)
    starts = np.zeros(len(data), dtype=np.int32)
    ends = np.zeros(len(data), dtype=np.int32)
    for index, (line, expected) in enumerate(zip(data, envelope)):
        positions = np.flatnonzero(expected)
        if positions.size == 0:
            continue
        start = int(positions[0])
        end = int(positions[-1]) + 1
        if (end - start) / len(line) < minimum_span_ratio:
            continue
        starts[index] = start
        ends[index] = end
        trim = min(
            round((end - start) * endpoint_trim_ratio),
            max(0, (end - start - 1) // 2),
        )
        measure_start = start + trim
        measure_end = end - trim
        scores[index] = float(data[index, measure_start:measure_end].mean())
        if gray_data is not None and minimum_contrast > 0:
            # 20% 图原来使用左右 2～3 像素。黑线图提高到 50% 后，按
            # 分辨率同比例把取样位置移到 5～8 像素外，避免仍取在线芯里。
            offsets = sorted(
                {
                    int(round(offset * contrast_offset_scale))
                    for offset in (-3, -2, 2, 3)
                }
            )
            shoulder_indices = [
                neighbor
                for offset in offsets
                if (neighbor := index + offset) != index
                if 0 <= neighbor < len(gray_data)
            ]
            if not shoulder_indices:
                scores[index] = 0.0
                continue
            center_mean = float(gray_data[index, measure_start:measure_end].mean())
            shoulder_mean = float(
                gray_data[shoulder_indices, measure_start:measure_end].mean()
            )
            # 覆盖率极高的连续物理线不再依赖邻域灰度差。邻域可能落在灰底、
            # 文字或缩放后的抗锯齿上，但这些因素不会改变“整列几乎全黑”。
            bypass_contrast = (
                contrast_bypass_ratio is not None
                and scores[index] >= contrast_bypass_ratio
            )
            if not bypass_contrast and shoulder_mean - center_mean < minimum_contrast:
                scores[index] = 0.0

    segments: list[LineSegment] = []
    for run_start, run_end in _runs(scores >= minimum_ratio):
        segments.append(
            LineSegment(
                position=round((run_start + run_end - 1) / 2),
                start=int(starts[run_start:run_end].min()),
                end=int(ends[run_start:run_end].max()),
            )
        )
    return segments


def _white_column_spacing_regularity(
    bands: list[WhiteColumnBand],
) -> float:
    """计算多数相邻白带是否遵循同一列距；允许少量宽列和合并列。"""

    if len(bands) < 3:
        return 0.0
    positions = np.asarray([band.position for band in bands], dtype=np.int32)
    gaps = np.diff(positions)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return 0.0
    typical = float(np.median(positive))
    tolerance = max(2.0, typical * 0.18)
    return float(np.mean(np.abs(positive - typical) <= tolerance))


def select_adaptive_white_column_bands(
    ink: np.ndarray,
    config: TableConfig,
) -> tuple[
    list[WhiteColumnBand],
    list[WhiteColumnBand],
    list[RejectedWhiteColumnBand],
    int,
    float,
    float,
    str,
]:
    """只为列白带选择固定最小宽度，横向行白带仍严格保持1像素能力。"""

    _, vertical_kernel = _whitespace_dilate_kernels(ink, config)
    expanded = cv2.dilate(
        ink.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel)),
    )
    raw_bands = [
        WhiteColumnBand(start, end)
        for start, end in _runs(
            expanded.mean(axis=0) <= config.whitespace_blank_ratio
        )
        if end - start >= config.whitespace_min_band
    ]
    base_minimum = config.whitespace_min_band
    before = _white_column_spacing_regularity(raw_bands)
    if (
        len(raw_bands) < 6
        or config.whitespace_column_max_min_band == base_minimum
        or before >= config.whitespace_column_regular_spacing_ratio
    ):
        message = (
            f"原始{len(raw_bands)}根，间距稳定度{before:.1%}，"
            f"列白带保持{base_minimum}px"
        )
        return (
            raw_bands,
            raw_bands,
            [],
            base_minimum,
            before,
            before,
            message,
        )

    required_count = max(
        5,
        int(np.ceil(
            len(raw_bands) * config.whitespace_column_min_retention_ratio
        )),
    )
    best_bands = raw_bands
    best_minimum = base_minimum
    best_regularity = before
    for minimum in range(
        base_minimum + 1,
        config.whitespace_column_max_min_band + 1,
    ):
        kept = [band for band in raw_bands if band.width >= minimum]
        removed = [band for band in raw_bands if band.width < minimum]
        if len(kept) < required_count or not removed:
            continue
        kept_width = float(np.median([band.width for band in kept]))
        removed_width = float(np.median([band.width for band in removed]))
        if (
            kept_width
            < removed_width
            * config.whitespace_column_min_width_separation_ratio
        ):
            continue
        regularity = _white_column_spacing_regularity(kept)
        if regularity > best_regularity + 1e-9:
            best_bands = kept
            best_minimum = minimum
            best_regularity = regularity

    gain = best_regularity - before
    if (
        best_minimum == base_minimum
        or gain < config.whitespace_column_min_regularity_gain
    ):
        message = (
            f"原始{len(raw_bands)}根，尝试至"
            f"{config.whitespace_column_max_min_band}px后改善不足，保持"
            f"{base_minimum}px；稳定度{before:.1%}"
        )
        return (
            raw_bands,
            raw_bands,
            [],
            base_minimum,
            before,
            before,
            message,
        )

    kept_set = set(best_bands)
    rejected = [
        RejectedWhiteColumnBand(
            band=band,
            reason=(
                f"白带宽{band.width}px，小于本区域选择的"
                f"{best_minimum}px；删除后列间距更稳定"
            ),
            selected_minimum=best_minimum,
        )
        for band in raw_bands
        if band not in kept_set
    ]
    message = (
        f"原始{len(raw_bands)}根，列白带固定最小宽度"
        f"{base_minimum}→{best_minimum}px，保留{len(best_bands)}根，"
        f"间距稳定度{before:.1%}→{best_regularity:.1%}"
    )
    return (
        raw_bands,
        best_bands,
        rejected,
        best_minimum,
        before,
        best_regularity,
        message,
    )


def clean_suspicious_column_lines(
    lines: list[LineSegment], config: TableConfig
) -> tuple[list[LineSegment], list[RejectedColumnLine], str]:
    """删除孤立拥挤候选，但保留整张表稳定存在的窄列规律。

    单根中文竖笔画可能在短行高表格中恰好首尾相接，覆盖率、连续性和灰度
    对比都会与真实竖线相似。此处不再重复判断单根线，而是检查候选线放进
    整张网格后是否连续制造了异常窄格。

    规则刻意保持保守：只有至少两个连续窄间距组成的局部拥挤簇才处理；若
    窄间距在全表反复出现，则把它视为真实的密集列规格，不做清理。被删除
    的候选会完整写入诊断和 manifest，供识别后复核或恢复备选网格。
    """

    ordered = sorted(lines, key=lambda line: line.position)
    if len(ordered) < 5:
        return ordered, [], "候选不足5根，不做网格间距清理"

    positions = np.asarray([line.position for line in ordered], dtype=np.int32)
    gaps = np.diff(positions)
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.size == 0:
        return ordered, [], "候选没有有效间距，不做网格间距清理"

    typical_gap = float(np.median(positive_gaps))
    minimum_gap = max(
        2,
        min(
            config.grid_min_cell_size,
            round(typical_gap * config.grid_black_column_min_gap_ratio),
        ),
    )
    close_mask = gaps < minimum_gap
    close_count = int(np.count_nonzero(close_mask))
    if close_count == 0:
        return (
            ordered,
            [],
            f"典型间距{typical_gap:.1f}px，最小可信间距{minimum_gap}px，无拥挤候选",
        )

    close_fraction = close_count / len(gaps)
    if close_fraction > config.grid_black_column_close_gap_max_fraction:
        return (
            ordered,
            [],
            f"窄间距占{close_fraction:.1%}，在全表反复出现，视为真实密集列",
        )

    rejected_indices: set[int] = set()
    rejected: list[RejectedColumnLine] = []
    for gap_start, gap_end in _runs(close_mask):
        # 一个拥挤簇包含 gap_start 到 gap_end 两端的候选线。只有连续两个
        # 以上窄间距才足以说明它不是普通的列宽变化。
        if gap_end - gap_start < 2:
            continue
        first_line_index = gap_start
        last_line_index = gap_end
        if positions[last_line_index] - positions[first_line_index] < minimum_gap:
            continue
        for line_index in range(first_line_index + 1, last_line_index):
            rejected_indices.add(line_index)
            rejected.append(
                RejectedColumnLine(
                    line=ordered[line_index],
                    reason="位于孤立拥挤候选簇内部，会连续制造异常窄列",
                    left_gap=int(gaps[line_index - 1]),
                    right_gap=int(gaps[line_index]),
                    typical_gap=typical_gap,
                    minimum_gap=minimum_gap,
                )
            )

    kept = [
        line for index, line in enumerate(ordered) if index not in rejected_indices
    ]
    message = (
        f"原始{len(ordered)}根，典型间距{typical_gap:.1f}px，"
        f"最小可信间距{minimum_gap}px，删除{len(rejected)}根孤立拥挤候选"
    )
    return kept, rejected, message


def _map_centers(
    centers: list[int],
    preview_length: int,
    source_start: int,
    source_length: int,
    *,
    include_outer: bool,
) -> tuple[int, ...]:
    """映射检测边界；只有白带模式才使用内容框外沿补齐首尾。"""

    detected = sorted(set(centers))
    local = [0, *detected, preview_length] if include_outer else detected
    mapped = [
        source_start + round(value * source_length / preview_length) for value in local
    ]
    if include_outer:
        mapped[0] = source_start
        mapped[-1] = source_start + source_length
    return tuple(dict.fromkeys(mapped))


def _erase_perpendicular_lines(
    ink: np.ndarray, lines: list[LineSegment], axis: int
) -> np.ndarray:
    """找另一方向白带前，擦掉已经确认的交叉表格线。

    例如横线可靠、竖线缺失时，完整横线会让每一列都带少量黑像素，直接找
    纵向白带就会失败。这里只在已确认线的有效跨度内擦除很窄的一条，不碰
    其余文字墨迹。
    """

    result = ink.copy()
    radius = max(1, round(min(ink.shape) * 0.001))
    for line in lines:
        lower = max(0, line.position - radius)
        upper = min(ink.shape[axis], line.position + radius + 1)
        if axis == 0:
            result[lower:upper, line.start : line.end] = False
        else:
            result[line.start : line.end, lower:upper] = False
    return result


def _black_lines_are_distributed(
    lines: list[LineSegment],
    length: int,
    minimum_count: int,
    interior_margin_ratio: float,
) -> tuple[bool, str]:
    """判断黑线数量和空间分布是否都像真实网格，而不是两侧边框。"""

    if len(lines) < minimum_count:
        return False, f"黑线只有{len(lines)}根，少于{minimum_count}根"
    margin = length * interior_margin_ratio
    interior = [line for line in lines if margin < line.position < length - margin]
    if not interior:
        return False, "黑线全部挤在表格两侧，中部没有物理边界"
    return True, f"{len(lines)}根黑线，含{len(interior)}根中部边界"


def _boundaries_have_reasonable_cells(
    boundaries: tuple[int, ...], maximum_span_ratio: float
) -> tuple[bool, str]:
    """拒绝某一格吞掉整张表的荒谬边界，避免错误网格流入切块。"""

    if len(boundaries) < 2:
        return False, "没有形成至少一个逻辑格"
    total = boundaries[-1] - boundaries[0]
    if total <= 0:
        return False, "边界总跨度为零"
    maximum = max(right - left for left, right in zip(boundaries, boundaries[1:]))
    ratio = maximum / total
    if ratio > maximum_span_ratio:
        return False, f"最大逻辑格占该方向{ratio:.1%}，超过{maximum_span_ratio:.0%}"
    return True, f"最大逻辑格占该方向{ratio:.1%}"


def detect_v6_grid(
    analysis_image: Image.Image,
    source_region: Box,
    config: TableConfig,
    *,
    black_analysis_image: Image.Image | None = None,
) -> tuple[GridStructure, V6GridDiagnostics]:
    """用高分辨率图找黑线、原 20% 图找白带，再映射回原图。"""

    # 白带链路保持原来的 20% 灰度图不变。
    white_gray = np.asarray(analysis_image.convert("L"))
    white_ink = white_gray < config.grid_white_threshold

    # 黑线可以单独使用分表区域的 50% 图。测试或历史调用没有传入时，
    # 仍退回同一张分析图，保证接口向后兼容。
    black_image = black_analysis_image or analysis_image
    black_gray = np.asarray(black_image.convert("L"))
    black_ink = black_gray < config.grid_white_threshold
    black_envelope = content_envelope_mask(black_ink)
    resolution_ratio = max(
        black_image.width / analysis_image.width,
        black_image.height / analysis_image.height,
    )
    black_rows = adaptive_line_segments(
        black_ink,
        black_envelope,
        0,
        config.grid_black_line_ratio,
    )
    raw_black_columns = adaptive_line_segments(
        black_ink,
        black_envelope,
        1,
        config.grid_black_column_line_ratio,
        grayscale=black_gray,
        endpoint_trim_ratio=config.grid_black_column_endpoint_trim_ratio,
        minimum_contrast=config.grid_black_column_min_contrast,
        contrast_offset_scale=resolution_ratio,
        contrast_bypass_ratio=config.grid_black_column_contrast_bypass_ratio,
    )
    black_columns, rejected_black_columns, column_cleanup = (
        clean_suspicious_column_lines(raw_black_columns, config)
    )

    # 白带分支以及它的交叉线擦除继续完全依据 20% 图自身结果，不能让
    # 新的 50% 黑线检测反过来改变已经稳定的白带。
    white_envelope = content_envelope_mask(white_ink)
    white_black_rows = adaptive_line_segments(
        white_ink,
        white_envelope,
        0,
        config.grid_black_line_ratio,
    )
    white_black_columns = adaptive_line_segments(
        white_ink,
        white_envelope,
        1,
        config.grid_black_column_line_ratio,
        grayscale=white_gray,
        endpoint_trim_ratio=config.grid_black_column_endpoint_trim_ratio,
        minimum_contrast=config.grid_black_column_min_contrast,
    )

    reliable = config.grid_reliable_line_count
    row_is_black, row_black_reason = _black_lines_are_distributed(
        black_rows,
        black_image.height,
        reliable,
        config.grid_interior_margin_ratio,
    )
    column_is_black, column_black_reason = _black_lines_are_distributed(
        black_columns,
        black_image.width,
        reliable,
        config.grid_interior_margin_ratio,
    )
    white_row_has_black = _black_lines_are_distributed(
        white_black_rows,
        analysis_image.height,
        reliable,
        config.grid_interior_margin_ratio,
    )[0]
    white_column_has_black = _black_lines_are_distributed(
        white_black_columns,
        analysis_image.width,
        reliable,
        config.grid_interior_margin_ratio,
    )[0]

    # 这是原有白带逻辑：找某方向白带前，只擦掉另一方向已确认的黑线。
    ink_for_rows = (
        _erase_perpendicular_lines(white_ink, white_black_columns, axis=1)
        if white_column_has_black
        else white_ink
    )
    ink_for_columns = (
        _erase_perpendicular_lines(white_ink, white_black_rows, axis=0)
        if white_row_has_black
        else white_ink
    )
    white_rows = _whitespace_centers(ink_for_rows, config)[0]
    (
        raw_white_column_bands,
        used_white_column_bands,
        rejected_white_column_bands,
        white_column_min_band,
        white_column_regularity_before,
        white_column_regularity_after,
        white_column_cleanup,
    ) = select_adaptive_white_column_bands(ink_for_columns, config)
    white_columns = [band.position for band in used_white_column_bands]

    row_centers = [line.position for line in black_rows] if row_is_black else white_rows
    column_centers = (
        [line.position for line in black_columns] if column_is_black else white_columns
    )
    row_source = (
        f"black-line-{config.grid_black_line_ratio:.2f}"
        if row_is_black
        else "white-band"
    )
    column_source = (
        f"black-line-{config.grid_black_column_line_ratio:.2f}-contrast"
        if column_is_black
        else "white-band"
    )
    if not row_centers or not column_centers:
        diagnostics = V6GridDiagnostics(
            row_source=row_source,
            column_source=column_source,
            black_rows=tuple(black_rows),
            black_columns=tuple(raw_black_columns),
            white_rows=tuple(white_rows),
            white_columns=tuple(white_columns),
            row_reliability=row_black_reason,
            column_reliability=column_black_reason,
            black_rows_at_whitespace_scale=tuple(white_black_rows),
            black_columns_at_whitespace_scale=tuple(white_black_columns),
            used_black_columns=tuple(black_columns),
            rejected_black_columns=tuple(rejected_black_columns),
            column_cleanup=column_cleanup,
            raw_white_column_bands=tuple(raw_white_column_bands),
            used_white_column_bands=tuple(used_white_column_bands),
            rejected_white_column_bands=tuple(rejected_white_column_bands),
            white_column_min_band=white_column_min_band,
            white_column_regularity_before=white_column_regularity_before,
            white_column_regularity_after=white_column_regularity_after,
            white_column_cleanup=white_column_cleanup,
        )
        return GridStructure("unavailable", (), ()), diagnostics

    rows = _map_centers(
        row_centers,
        black_image.height if row_is_black else analysis_image.height,
        source_region.y1,
        source_region.height,
        include_outer=not row_is_black,
    )
    columns = _map_centers(
        column_centers,
        black_image.width if column_is_black else analysis_image.width,
        source_region.x1,
        source_region.width,
        include_outer=not column_is_black,
    )
    raw_columns = (
        _map_centers(
            [line.position for line in raw_black_columns],
            black_image.width,
            source_region.x1,
            source_region.width,
            include_outer=False,
        )
        if column_is_black
        else _map_centers(
            [band.position for band in raw_white_column_bands],
            analysis_image.width,
            source_region.x1,
            source_region.width,
            include_outer=True,
        )
    )
    rejected_columns = (
        _map_centers(
            [item.line.position for item in rejected_black_columns],
            black_image.width,
            source_region.x1,
            source_region.width,
            include_outer=False,
        )
        if column_is_black
        else _map_centers(
            [item.band.position for item in rejected_white_column_bands],
            analysis_image.width,
            source_region.x1,
            source_region.width,
            include_outer=False,
        )
    )
    row_cells_ok, row_cell_reason = _boundaries_have_reasonable_cells(
        rows, config.grid_max_cell_span_ratio
    )
    column_cells_ok, column_cell_reason = _boundaries_have_reasonable_cells(
        columns, config.grid_max_cell_span_ratio
    )
    diagnostics = V6GridDiagnostics(
        row_source=row_source,
        column_source=column_source,
        black_rows=tuple(black_rows),
        black_columns=tuple(raw_black_columns),
        white_rows=tuple(white_rows),
        white_columns=tuple(white_columns),
        row_reliability=f"{row_black_reason}；{row_cell_reason}",
        column_reliability=f"{column_black_reason}；{column_cell_reason}",
        black_rows_at_whitespace_scale=tuple(white_black_rows),
        black_columns_at_whitespace_scale=tuple(white_black_columns),
        used_black_columns=tuple(black_columns),
        rejected_black_columns=tuple(rejected_black_columns),
        column_cleanup=column_cleanup,
        raw_white_column_bands=tuple(raw_white_column_bands),
        used_white_column_bands=tuple(used_white_column_bands),
        rejected_white_column_bands=tuple(rejected_white_column_bands),
        white_column_min_band=white_column_min_band,
        white_column_regularity_before=white_column_regularity_before,
        white_column_regularity_after=white_column_regularity_after,
        white_column_cleanup=white_column_cleanup,
    )
    if len(rows) < 2 or len(columns) < 2 or not row_cells_ok or not column_cells_ok:
        return (
            GridStructure(
                "unavailable", (), (), raw_columns, rejected_columns
            ),
            diagnostics,
        )
    return (
        GridStructure(
            source=f"v6:rows={row_source};columns={column_source}",
            row_boundaries=rows,
            column_boundaries=columns,
            raw_column_boundaries=raw_columns,
            rejected_column_boundaries=rejected_columns,
        ),
        diagnostics,
    )
