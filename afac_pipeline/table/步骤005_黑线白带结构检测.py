"""图表正式结构检测：横向分表、黑线优先与滑窗表体白带。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..common.models import Box, DetectedBox
from .config import TableConfig
from .步骤002_低密度分表 import (
    DensityBand,
    _runs,
    horizontal_table_split_boxes,
)
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
    """一条纵向白带在本区域实际采用的分析图中的完整像素范围。"""

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
    white_column_uses_black_scale: bool = False
    # 表体滑动窗口诊断；窗口为空时仍保留旧白带兜底结果。
    body_window_selected: tuple[int, int] | None = None
    body_window_box: tuple[int, int, int, int] | None = None
    body_window_height: int = 0
    body_window_step: int = 0
    body_window_results: tuple[dict[str, object], ...] = ()
    body_window_cleanup: str = ""

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
            "white_column_analysis": (
                "50%条件兜底图" if self.white_column_uses_black_scale else "20%常规图"
            ),
            "body_window_selected": (
                None
                if self.body_window_selected is None
                else list(self.body_window_selected)
            ),
            "body_window_box": (
                None if self.body_window_box is None else list(self.body_window_box)
            ),
            "body_window_height": self.body_window_height,
            "body_window_step": self.body_window_step,
            "body_window_results": list(self.body_window_results),
            "body_window_cleanup": self.body_window_cleanup,
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


def _merge_shallow_title_regions(
    split_boxes: list[Box],
    analysis_boxes: list[Box],
    preview_size: tuple[int, int],
    config: TableConfig,
) -> tuple[list[Box], list[Box]]:
    """把密度分表误切出的单行表题并回紧随其后的主表。

    这里只合并“占整页高度很小、又远矮于下一块”的上方区域。真正的小表
    即使位于大表上方，也通常不会同时满足这两个严格比例。合并后仍保留原
    像素和中间空白，后续由顶部候选区流程单独识别标题。
    """

    if len(split_boxes) < 2:
        return split_boxes, analysis_boxes
    page_height = preview_size[1]
    merged_split: list[Box] = []
    merged_analysis: list[Box] = []
    index = 0
    while index < len(split_boxes):
        if index + 1 < len(split_boxes):
            upper_split = split_boxes[index]
            lower_split = split_boxes[index + 1]
            upper = analysis_boxes[index]
            lower = analysis_boxes[index + 1]
            overlap = max(
                0,
                min(upper.x2, lower.x2) - max(upper.x1, lower.x1),
            )
            minimum_width = max(1, min(upper.width, lower.width))
            looks_like_title = (
                overlap / minimum_width >= 0.60
                and upper.height
                <= page_height * config.density_title_strip_max_page_height_ratio
                and upper.height
                <= lower.height * config.density_title_strip_max_next_height_ratio
            )
            if looks_like_title:
                merged_split.append(
                    Box(
                        min(upper_split.x1, lower_split.x1),
                        upper_split.y1,
                        max(upper_split.x2, lower_split.x2),
                        lower_split.y2,
                    )
                )
                merged_analysis.append(
                    Box(
                        min(upper.x1, lower.x1),
                        upper.y1,
                        max(upper.x2, lower.x2),
                        lower.y2,
                    )
                )
                index += 2
                continue
        merged_split.append(split_boxes[index])
        merged_analysis.append(analysis_boxes[index])
        index += 1
    return merged_split, merged_analysis


def detect_v6_regions(preview: Image.Image, config: TableConfig) -> V6RegionResult:
    """在原图20%分析图上只按上下方向切开同图异表。"""

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
    horizontal_boxes, preview_horizontal, _ = horizontal_table_split_boxes(
        preview,
        gray_threshold=config.ink_threshold,
    )
    if not horizontal_boxes:
        horizontal_boxes = [Box(0, 0, preview.width, preview.height)]
    horizontal = [
        DensityBand(
            axis="horizontal",
            start=round(band.start * density.shape[0] / preview.height),
            end=max(
                round(band.start * density.shape[0] / preview.height) + 1,
                round(band.end * density.shape[0] / preview.height),
            ),
            mean_density=band.mean_density,
            source=band.source,
        )
        for band in preview_horizontal
    ]
    vertical: list[DensityBand] = []

    split_boxes: list[Box] = []
    analysis_boxes: list[Box] = []
    gray = np.asarray(preview.convert("L"))
    for split_box in horizontal_boxes:
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
    # 表题与主表之间的空白有时比真正表间空白还明显。若上块只是很浅的
    # 单行标题，就把它并回下表；真正的三张表（例如185a）高度足够，不会
    # 命中该规则。
    split_boxes, analysis_boxes = _merge_shallow_title_regions(
        split_boxes,
        analysis_boxes,
        preview.size,
        config,
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


def _position_spacing_regularity(positions: list[int]) -> float:
    """计算多数相邻位置是否遵循同一间距；允许少量宽格和合并格。"""

    if len(positions) < 3:
        return 0.0
    values = np.asarray(positions, dtype=np.int32)
    gaps = np.diff(values)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return 0.0
    typical = float(np.median(positive))
    tolerance = max(2.0, typical * 0.18)
    return float(np.mean(np.abs(positive - typical) <= tolerance))


def _white_column_spacing_regularity(
    bands: list[WhiteColumnBand],
) -> float:
    """计算多数相邻列白带是否遵循同一列距。"""

    return _position_spacing_regularity([band.position for band in bands])


def _white_rows_reveal_partial_black_grid(
    black_rows: list[LineSegment],
    white_rows: list[int],
    config: TableConfig,
) -> tuple[bool, str]:
    """识别只有表头或分段处画横线、数据行本身没有横线的表。

    单纯“黑线不少于5根”无法区分完整网格与局部分隔线。这里要求白带数量
    至少达到固定下限、又达到黑线数量的若干倍，并且间距自身足够稳定。
    三个条件同时成立，才说明白带表达的是完整数据行，而黑线只是局部表头。
    """

    regularity = _position_spacing_regularity(white_rows)
    required = max(
        config.grid_partial_line_min_white_bands,
        int(np.ceil(
            len(black_rows) * config.grid_partial_line_white_band_multiplier
        )),
    )
    is_partial = (
        len(white_rows) >= required
        and regularity >= config.grid_partial_line_min_white_regularity
    )
    reason = (
        f"稳定行白带{len(white_rows)}根（采用黑线需少于{required}根），"
        f"间距稳定度{regularity:.1%}"
    )
    return is_partial, reason


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


def _remove_outer_white_column_bands(
    raw_bands: list[WhiteColumnBand],
    used_bands: list[WhiteColumnBand],
    rejected: list[RejectedWhiteColumnBand],
    width: int,
    selected_minimum: int,
) -> tuple[
    list[WhiteColumnBand],
    list[WhiteColumnBand],
    list[RejectedWhiteColumnBand],
]:
    """吸收与分析框外沿直接相连的空白，不让留白本身制造一列。

    这不是旧版“删除边缘3%”：只有 start==0 或 end==width 的白带会被
    吸附为外框，离外沿哪怕还有1像素也仍作为真实候选保留。
    """

    outer = {band for band in raw_bands if band.start == 0 or band.end == width}
    if not outer:
        return raw_bands, used_bands, rejected
    kept_raw = [band for band in raw_bands if band not in outer]
    kept_used = [band for band in used_bands if band not in outer]
    already_rejected = {item.band for item in rejected}
    extra = [
        RejectedWhiteColumnBand(
            band=band,
            reason="白带与分析框外沿直接相连，吸附为外框，不额外制造列",
            selected_minimum=selected_minimum,
        )
        for band in outer
        if band not in already_rejected
    ]
    return kept_raw, kept_used, [*rejected, *extra]


def _select_body_column_bands(
    ink: np.ndarray,
    white_rows: list[int],
    config: TableConfig,
) -> tuple[
    list[WhiteColumnBand],
    list[WhiteColumnBand],
    list[RejectedWhiteColumnBand],
    int,
    float,
    float,
    str,
] | None:
    """在长表的数据主体区复核列白带，避开表头与页脚文字干扰。"""

    if len(white_rows) < config.whitespace_column_body_min_row_bands:
        return None
    trim_count = max(
        2,
        round(len(white_rows) * config.whitespace_column_body_trim_ratio),
    )
    if trim_count * 2 + 2 >= len(white_rows):
        return None
    start = white_rows[trim_count]
    end = white_rows[-trim_count - 1]
    if end - start < ink.shape[0] * 0.40:
        return None
    result = list(select_adaptive_white_column_bands(ink[start:end], config))
    raw, used, rejected = _remove_outer_white_column_bands(
        result[0], result[1], result[2], ink.shape[1], result[3]
    )
    before = _white_column_spacing_regularity(raw)
    after = _white_column_spacing_regularity(used)
    result[0] = raw
    result[1] = used
    result[2] = rejected
    result[4] = before
    result[5] = after
    result[6] = (
        f"主体区y={start}:{end}（首尾各略过{trim_count}根行白带）；"
        f"{result[6]}；吸附外沿白带后保留{len(used)}根，"
        f"间距稳定度{after:.1%}"
    )
    return tuple(result)  # type: ignore[return-value]


def _erase_confirmed_grid_lines(ink, rows, columns):
    """擦除已确认黑线，避免污染白缝投影。"""
    result=ink.copy()
    radius=max(1,round(min(ink.shape)*0.001))
    for line in rows:
        y1=max(0,line.position-radius); y2=min(ink.shape[0],line.position+radius+1)
        result[y1:y2,max(0,line.start):min(ink.shape[1],line.end)]=False
    for line in columns:
        x1=max(0,line.position-radius); x2=min(ink.shape[1],line.position+radius+1)
        result[max(0,line.start):min(ink.shape[0],line.end),x1:x2]=False
    return result

def _first_stable_column_bands(profile, config):
    """列白缝从1px向上寻找，第一次连续稳定就停止。"""
    raw=[WhiteColumnBand(a,b) for a,b in _runs(profile<=config.whitespace_blank_ratio)
         if a>0 and b<len(profile) and b-a>=config.whitespace_min_band]
    counts={}; repeat=config.body_column_stable_repeat
    for minimum in range(1,config.body_column_stable_max_width+1):
        count=sum(band.width>=minimum for band in raw); counts[str(minimum)]=count
        if minimum>=repeat:
            values=[counts[str(minimum-repeat+1+offset)] for offset in range(repeat)]
            if values[0]>0 and len(set(values))==1:
                return raw,[band for band in raw if band.width>=minimum-repeat+1],minimum-repeat+1,counts
    return raw,[],None,counts

def _compatible_body_windows(previous,current,config):
    if not previous["stable"] or not current["stable"]: return False
    if abs(int(previous["band_count"])-int(current["band_count"]))>config.body_column_max_count_delta: return False
    left=np.asarray([a+(b-a)/2 for a,b in previous["bands"]])
    right=np.asarray([a+(b-a)/2 for a,b in current["bands"]])
    if left.size==0 or right.size==0: return False
    smaller,larger=(left,right) if left.size<=right.size else (right,left)
    spacing=np.median(np.diff(smaller)) if smaller.size>=3 else 20
    tolerance=max(float(config.body_column_position_tolerance_px),float(spacing)*0.25)
    return sum(np.min(np.abs(larger-value))<=tolerance for value in smaller)/smaller.size>=config.body_column_min_position_match

def _choose_body_window_range(results,config):
    best=(0,0); start=None
    def update(end,begin):
        nonlocal best
        if begin is not None and end-begin>best[1]-best[0]: best=(begin,end)
    for index,current in enumerate(results):
        if not current["stable"]: update(index,start); start=None; continue
        if start is None: start=index; continue
        if not _compatible_body_windows(results[index-1],current,config): update(index,start); start=index
    update(len(results),start)
    return best if best[1]-best[0]>=config.body_window_min_count else None

def _select_sliding_body_columns(black_ink,black_rows,black_columns,config):
    """在50%图中用列结构稳定性定位表体，再在整段表体上取列白缝。"""
    height,width=black_ink.shape
    erased=_erase_confirmed_grid_lines(black_ink,black_rows,black_columns)
    density=max(0.01,float(erased.mean()))
    dilate_ratio=float(np.clip(0.01*0.15/density,config.body_column_dilate_min_ratio,config.body_column_dilate_max_ratio))
    kernel=max(3,round(height*dilate_ratio))
    mask=cv2.dilate(erased.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_RECT,(1,kernel))).astype(bool)
    window_height=min(height,max(config.body_window_min_height,round(height*config.body_window_height_ratio)))
    step=max(20,round(height*config.body_window_step_ratio))
    starts=list(range(0,max(1,height-window_height+1),step))
    if starts and starts[-1]!=height-window_height: starts.append(height-window_height)
    results=[]
    for start in starts:
        end=start+window_height
        raw,selected,threshold,counts=_first_stable_column_bands(mask[start:end].mean(axis=0),config)
        results.append({"start":start,"end":end,"stable":threshold is not None,"threshold":threshold,"band_count":len(selected),"bands":[[b.start,b.end] for b in selected],"counts_until_stop":counts,"raw_band_count":len(raw)})
    selected_range=_choose_body_window_range(results,config)
    base={"selected":selected_range,"window_height":window_height,"step":step,"dilate_ratio":dilate_ratio,"mask":mask,"windows":results}
    if selected_range is None:
        return {**base,"body_box":None,"raw":[],"used":[],"threshold":None,"message":"没有连续稳定的表体窗口"}
    y1=int(results[selected_range[0]]["start"])
    y2=int(results[selected_range[1]-1]["end"])
    raw,used,threshold,counts=_first_stable_column_bands(mask[y1:y2].mean(axis=0),config)
    if threshold is None or len(used)<1:
        return {**base,"body_box":(0,y1,width,y2),"raw":raw,"used":[],"threshold":threshold,"message":"选中的表体整体没有形成稳定列白缝"}
    return {**base,"body_box":(0,y1,width,y2),"raw":raw,"used":used,"threshold":threshold,"counts_until_stop":counts,"message":f"表体窗口{selected_range[0]}～{selected_range[1]-1}，y={y1}:{y2}，列白缝首个稳定阈值{threshold}px，保留{len(used)}根"}

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


def _sparse_black_grid_centers(
    lines: list[LineSegment],
    length: int,
    minimum_cell_size: int,
    maximum_span_ratio: float,
    source_length: int,
    maximum_source_cell_size: int,
) -> tuple[list[int], bool, str]:
    """让两三列的小表也能使用严格黑线，而不要求至少出现5根线。

    靠近分析框两端的线吸附为外框，只保留真正位于内部的线。这样
    “左外框+一根中间线”可以表达两列，同时不会额外制造一个极窄边缘列。
    """

    if len(lines) < 2:
        return [], False, f"严格黑线只有{len(lines)}根，无法确认小表网格"
    edge_tolerance = max(6, minimum_cell_size)
    interior = sorted(
        {
            line.position
            for line in lines
            if edge_tolerance < line.position < length - edge_tolerance
        }
    )
    local_boundaries = tuple([0, *interior, length])
    if len(local_boundaries) < 3:
        return [], False, "严格黑线没有形成至少两列"
    ok, cell_reason = _boundaries_have_reasonable_cells(
        local_boundaries,
        maximum_span_ratio,
    )
    maximum_local_span = max(
        right - left
        for left, right in zip(local_boundaries, local_boundaries[1:])
    )
    maximum_source_span = round(maximum_local_span * source_length / length)
    if maximum_source_span > maximum_source_cell_size:
        ok = False
        cell_reason = (
            f"最大格映射回原图为{maximum_source_span}px，"
            f"超过切片上限{maximum_source_cell_size}px"
        )
    return (
        interior,
        ok,
        f"{len(lines)}根严格黑线吸附外框后形成{len(local_boundaries) - 1}格；"
        f"{cell_reason}",
    )


def _hybrid_sparse_row_centers(
    black_rows: list[LineSegment],
    white_rows: list[int],
) -> tuple[list[int], str]:
    """用稀疏横线固定表头，再用白带细分其间的大块数据区。

    少量横线常常只包住多行表头和整段数据区。若直接只用横线，几十行数据
    会被吞成一格；若直接只用所有白带，多行表头中的文字行又会被误拆。
    因此短黑线间隔保持为一个逻辑行，只有明显超过典型行距的大间隔才插入
    内部白带。
    """

    black_positions = sorted({line.position for line in black_rows})
    ordered_white = sorted(set(white_rows))
    if len(black_positions) < 2 or len(ordered_white) < 3:
        return [], "稀疏横线或行白带数量不足，无法组合"
    gaps = np.diff(np.asarray(ordered_white, dtype=np.int32))
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return [], "行白带没有有效间距，无法组合"
    typical = float(np.median(positive))
    edge_guard = max(2, round(typical * 0.35))
    minimum_split_span = typical * 2.5
    used_white: list[int] = []
    for left, right in zip(black_positions, black_positions[1:]):
        if right - left <= minimum_split_span:
            continue
        used_white.extend(
            center
            for center in ordered_white
            if left + edge_guard < center < right - edge_guard
        )
    combined = sorted(set([*black_positions, *used_white]))
    message = (
        f"{len(black_positions)}根稀疏横线固定表头/外框，"
        f"典型行距{typical:.1f}px，在大间隔内采用{len(used_white)}根行白带"
    )
    return combined, message


def detect_v6_grid(
    analysis_image: Image.Image,
    source_region: Box,
    config: TableConfig,
    *,
    black_analysis_image: Image.Image | None = None,
) -> tuple[GridStructure, V6GridDiagnostics]:
    """用50%图找黑线和稳定表体列缝，20%图保留行白带兜底，再映射回原图。"""

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
    column_is_black, column_black_reason = _black_lines_are_distributed(
        black_columns,
        black_image.width,
        reliable,
        config.grid_interior_margin_ratio,
    )
    (
        sparse_column_centers,
        sparse_columns_ok,
        sparse_column_reason,
    ) = _sparse_black_grid_centers(
        black_columns,
        black_image.width,
        config.grid_min_cell_size,
        config.grid_max_cell_span_ratio,
        source_region.width,
        config.max_vlm_side,
    )
    column_uses_sparse_black = not column_is_black and sparse_columns_ok
    if column_uses_sparse_black:
        column_black_reason = (
            f"{column_black_reason}；小列数表复核通过：{sparse_column_reason}"
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
    row_is_black, row_black_reason = _black_lines_are_distributed(
        black_rows,
        black_image.height,
        reliable,
        config.grid_interior_margin_ratio,
    )

    (
        raw_white_column_bands,
        used_white_column_bands,
        rejected_white_column_bands,
        white_column_min_band,
        white_column_regularity_before,
        white_column_regularity_after,
        white_column_cleanup,
    ) = select_adaptive_white_column_bands(ink_for_columns, config)
    # 分析框会保留少量安全留白；与外沿直接相连的白带属于外框，不应
    # 再额外制造一列。这里只吸收“碰到外沿”的带，不恢复旧版3%删除。
    (
        raw_white_column_bands,
        used_white_column_bands,
        rejected_white_column_bands,
    ) = _remove_outer_white_column_bands(
        raw_white_column_bands,
        used_white_column_bands,
        rejected_white_column_bands,
        analysis_image.width,
        white_column_min_band,
    )
    white_column_regularity_before = _white_column_spacing_regularity(
        raw_white_column_bands
    )
    white_column_regularity_after = _white_column_spacing_regularity(
        used_white_column_bands
    )

    # d8b、0f372一类长无框表的表头文字会横跨真实列缝，若在整张区域上
    # 统计，会把一个列缝切成许多碎白带。行很多时，再用去掉少量表头页脚
    # 的数据主体区复核；只有间距达到高稳定度且显著优于全图时才采用。
    body_columns = _select_body_column_bands(
        ink_for_columns,
        white_rows,
        config,
    )
    if body_columns is not None:
        (
            body_raw,
            body_used,
            body_rejected,
            body_min_band,
            body_before,
            body_after,
            body_cleanup,
        ) = body_columns
        body_boundaries = tuple(
            dict.fromkeys([0, *[band.position for band in body_used], analysis_image.width])
        )
        body_cells_ok = _boundaries_have_reasonable_cells(
            body_boundaries,
            config.grid_max_cell_span_ratio,
        )[0]
        improvement = body_after - white_column_regularity_after
        # 0f372只漏了一条真实列缝：主体区稳定度与全图接近，但候选数恰好
        # 多1根。此类“小幅补回”不要求提升8个百分点，只要求主体本身仍
        # 足够规律，且最多增加2根，防止重新引入几十条表头碎白带。
        small_recovery = (
            0 < len(body_used) - len(used_white_column_bands) <= 2
            and body_after >= white_column_regularity_after
            and body_after
            >= config.whitespace_column_regular_spacing_ratio - 0.05
        )
        if (
            body_cells_ok
            and len(body_used) >= 2
            and (
                improvement >= config.whitespace_column_min_regularity_gain
                or small_recovery
            )
        ):
            full_cleanup = white_column_cleanup
            raw_white_column_bands = body_raw
            used_white_column_bands = body_used
            rejected_white_column_bands = body_rejected
            white_column_min_band = body_min_band
            white_column_regularity_before = body_before
            white_column_regularity_after = body_after
            white_column_cleanup = (
                f"全图列白带稳定度"
                f"{white_column_regularity_after - improvement:.1%}；"
                f"改用{body_cleanup}；原全图处理：{full_cleanup}"
            )

    # 新主体算法优先使用50%图。黑线可靠时不介入，避免白缝干扰有线网格。
    body_selection = None
    body_window_selected = None
    body_window_box = None
    body_window_height = 0
    body_window_step = 0
    body_window_results = ()
    body_window_cleanup = ""
    white_column_uses_black_scale = False
    if (
        not column_is_black
        and not column_uses_sparse_black
        and black_image.size != analysis_image.size
    ):
        body_selection = _select_sliding_body_columns(
            black_ink,
            black_rows,
            black_columns,
            config,
        )
        if body_selection is not None:
            body_window_selected = body_selection.get("selected")
            body_window_box = body_selection.get("body_box")
            body_window_height = int(body_selection.get("window_height", 0))
            body_window_step = int(body_selection.get("step", 0))
            body_window_results = tuple(body_selection.get("windows", ()))
            body_window_cleanup = str(body_selection.get("message", ""))
            body_used = list(body_selection.get("used", ()))
            body_raw = list(body_selection.get("raw", ()))
            if body_used:
                raw_white_column_bands = body_raw
                used_white_column_bands = body_used
                rejected_white_column_bands = []
                white_column_min_band = int(body_selection["threshold"])
                white_column_regularity_before = _white_column_spacing_regularity(body_raw)
                white_column_regularity_after = _white_column_spacing_regularity(body_used)
                white_column_cleanup = (
                    f"{body_window_cleanup}；"
                    "列白缝坐标采用50%表体图，后续映射回原图"
                )
                white_column_uses_black_scale = True

    # 20%图在超宽密集表上可能把细列缝压没。只有常规列白带本身无法形成
    # 可信网格时，才在50%图上重试；重试前必须擦掉贯穿全宽的横线，否则
    # 几根粗横线就足以让每一列的墨水比例超过1%。
    low_column_boundaries = tuple(
        dict.fromkeys(
            [
                0,
                *[band.position for band in used_white_column_bands],
                (
                    black_image.width
                    if white_column_uses_black_scale
                    else analysis_image.width
                ),
            ]
        )
    )
    low_columns_ok = _boundaries_have_reasonable_cells(
        low_column_boundaries,
        config.grid_max_cell_span_ratio,
    )[0]
    if (
        not column_is_black
        and not column_uses_sparse_black
        and not low_columns_ok
        and not white_column_uses_black_scale
        and black_image.size != analysis_image.size
    ):
        high_ink_for_columns = (
            _erase_perpendicular_lines(black_ink, black_rows, axis=0)
            if black_rows
            else black_ink
        )
        (
            high_raw_bands,
            high_used_bands,
            high_rejected_bands,
            high_min_band,
            high_regularity_before,
            high_regularity_after,
            high_cleanup,
        ) = select_adaptive_white_column_bands(high_ink_for_columns, config)
        high_column_boundaries = tuple(
            dict.fromkeys(
                [
                    0,
                    *[band.position for band in high_used_bands],
                    black_image.width,
                ]
            )
        )
        high_columns_ok = _boundaries_have_reasonable_cells(
            high_column_boundaries,
            config.grid_max_cell_span_ratio,
        )[0]
        if high_columns_ok and len(high_used_bands) > len(used_white_column_bands):
            low_cleanup = white_column_cleanup
            raw_white_column_bands = high_raw_bands
            used_white_column_bands = high_used_bands
            rejected_white_column_bands = high_rejected_bands
            white_column_min_band = high_min_band
            white_column_regularity_before = high_regularity_before
            white_column_regularity_after = high_regularity_after
            white_column_cleanup = (
                f"20%常规图不可信：{low_cleanup}；"
                f"改用50%图并擦除{len(black_rows)}根横线：{high_cleanup}"
            )
            white_column_uses_black_scale = True

    partial_rows, partial_row_reason = _white_rows_reveal_partial_black_grid(
        black_rows,
        white_rows,
        config,
    )
    if row_is_black and partial_rows:
        row_is_black = False
        row_black_reason = (
            f"{row_black_reason}；{partial_row_reason}，"
            "判定黑线只是局部表头/分段线，正式行网格改用白带"
        )

    row_uses_sparse_hybrid = False
    hybrid_row_centers: list[int] = []
    if not row_is_black and white_black_columns and len(white_black_rows) >= 2:
        current_row_boundaries = tuple(
            dict.fromkeys([0, *white_rows, analysis_image.height])
        )
        current_rows_ok = _boundaries_have_reasonable_cells(
            current_row_boundaries,
            config.grid_max_cell_span_ratio,
        )[0]
        if not current_rows_ok:
            # 少量严格竖线不足以单独证明完整列网格，却仍会堵住所有横向
            # 白带。仅在当前行结构已经不可信时擦除它们，再让稀疏横线
            # 固定表头/外框、白带细分中间的大数据区。
            retry_row_ink = _erase_perpendicular_lines(
                white_ink,
                white_black_columns,
                axis=1,
            )
            retry_white_rows = _whitespace_centers(retry_row_ink, config)[0]
            hybrid_row_centers, hybrid_reason = _hybrid_sparse_row_centers(
                white_black_rows,
                retry_white_rows,
            )
            hybrid_boundaries = tuple(hybrid_row_centers)
            hybrid_ok = _boundaries_have_reasonable_cells(
                hybrid_boundaries,
                config.grid_max_cell_span_ratio,
            )[0]
            if hybrid_ok and len(hybrid_row_centers) > len(white_rows):
                white_rows = retry_white_rows
                row_uses_sparse_hybrid = True
                row_black_reason = (
                    f"{row_black_reason}；常规行白带不可信，擦除"
                    f"{len(white_black_columns)}根严格竖线后：{hybrid_reason}"
                )

    white_columns = [band.position for band in used_white_column_bands]
    white_column_preview_width = (
        black_image.width
        if white_column_uses_black_scale
        else analysis_image.width
    )

    row_centers = (
        [line.position for line in black_rows]
        if row_is_black
        else (hybrid_row_centers if row_uses_sparse_hybrid else white_rows)
    )
    column_centers = (
        [line.position for line in black_columns]
        if column_is_black
        else (sparse_column_centers if column_uses_sparse_black else white_columns)
    )
    row_source = (
        f"black-line-{config.grid_black_line_ratio:.2f}"
        if row_is_black
        else (
            "hybrid-sparse-lines-white-bands"
            if row_uses_sparse_hybrid
            else "white-band"
        )
    )
    column_source = (
        f"black-line-{config.grid_black_column_line_ratio:.2f}-contrast"
        if column_is_black
        else (
            "sparse-black-lines-0.95-contrast"
            if column_uses_sparse_black
            else (
                "white-band-50%-fallback"
                if white_column_uses_black_scale
                else "white-band"
            )
        )
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
            white_column_uses_black_scale=white_column_uses_black_scale,
            body_window_selected=body_window_selected,
            body_window_box=body_window_box,
            body_window_height=body_window_height,
            body_window_step=body_window_step,
            body_window_results=body_window_results,
            body_window_cleanup=body_window_cleanup,
        )
        return GridStructure("unavailable", (), ()), diagnostics

    rows = _map_centers(
        row_centers,
        black_image.height if row_is_black else analysis_image.height,
        source_region.y1,
        source_region.height,
        include_outer=not row_is_black and not row_uses_sparse_hybrid,
    )
    columns = _map_centers(
        column_centers,
        (
            black_image.width
            if column_is_black or column_uses_sparse_black
            else white_column_preview_width
        ),
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
            include_outer=column_uses_sparse_black,
        )
        if column_is_black or column_uses_sparse_black
        else _map_centers(
            [band.position for band in raw_white_column_bands],
            white_column_preview_width,
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
            include_outer=column_uses_sparse_black,
        )
        if column_is_black or column_uses_sparse_black
        else _map_centers(
            [item.band.position for item in rejected_white_column_bands],
            white_column_preview_width,
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
        white_column_uses_black_scale=white_column_uses_black_scale,
        body_window_selected=body_window_selected,
        body_window_box=body_window_box,
        body_window_height=body_window_height,
        body_window_step=body_window_step,
        body_window_results=body_window_results,
        body_window_cleanup=body_window_cleanup,
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
