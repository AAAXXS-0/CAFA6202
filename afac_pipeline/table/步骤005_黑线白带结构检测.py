"""图表 v6：固定比例密度分表与黑线优先、白带兜底的结构检测。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..common.models import Box, DetectedBox
from .config import TableConfig
from .步骤002_低密度分表 import DensityBand, _runs, boxes_from_bands, find_density_bands
from .步骤004_网格与白带检测 import GridStructure, _whitespace_centers
from .步骤001_墨水密度定位 import InkRegionResult, detect_ink_regions


@dataclass(frozen=True)
class LineSegment:
    """一条黑线及其在梯形表格包络内的有效跨度。"""

    position: int
    start: int
    end: int


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

    def to_dict(self) -> dict[str, object]:
        return {
            "row_source": self.row_source,
            "column_source": self.column_source,
            "black_rows": [line.__dict__ for line in self.black_rows],
            "black_columns": [line.__dict__ for line in self.black_columns],
            "white_rows": list(self.white_rows),
            "white_columns": list(self.white_columns),
            "row_reliability": self.row_reliability,
            "column_reliability": self.column_reliability,
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
            shoulder_indices = [
                neighbor
                for neighbor in (index - 3, index - 2, index + 2, index + 3)
                if 0 <= neighbor < len(gray_data)
            ]
            if not shoulder_indices:
                scores[index] = 0.0
                continue
            center_mean = float(gray_data[index, measure_start:measure_end].mean())
            shoulder_mean = float(
                gray_data[shoulder_indices, measure_start:measure_end].mean()
            )
            if shoulder_mean - center_mean < minimum_contrast:
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
) -> tuple[GridStructure, V6GridDiagnostics]:
    """对一张已分开的表检测逻辑行列边界，并映射回原图坐标。"""

    gray = np.asarray(analysis_image.convert("L"))
    ink = gray < config.grid_white_threshold
    envelope = content_envelope_mask(ink)
    black_rows = adaptive_line_segments(ink, envelope, 0, config.grid_black_line_ratio)
    black_columns = adaptive_line_segments(
        ink,
        envelope,
        1,
        config.grid_black_column_line_ratio,
        grayscale=gray,
        endpoint_trim_ratio=config.grid_black_column_endpoint_trim_ratio,
        minimum_contrast=config.grid_black_column_min_contrast,
    )
    reliable = config.grid_reliable_line_count
    row_is_black, row_black_reason = _black_lines_are_distributed(
        black_rows,
        analysis_image.height,
        reliable,
        config.grid_interior_margin_ratio,
    )
    column_is_black, column_black_reason = _black_lines_are_distributed(
        black_columns,
        analysis_image.width,
        reliable,
        config.grid_interior_margin_ratio,
    )
    # 某个方向使用白带兜底时，先擦掉另一方向已经确认的黑线。否则水平
    # 表格线会贯穿所有列（或竖线贯穿所有行），把真实白带堵死。
    ink_for_rows = (
        _erase_perpendicular_lines(ink, black_columns, axis=1)
        if column_is_black
        else ink
    )
    ink_for_columns = (
        _erase_perpendicular_lines(ink, black_rows, axis=0) if row_is_black else ink
    )
    white_rows = _whitespace_centers(ink_for_rows, config)[0]
    white_columns = _whitespace_centers(ink_for_columns, config)[1]
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
            row_source,
            column_source,
            tuple(black_rows),
            tuple(black_columns),
            tuple(white_rows),
            tuple(white_columns),
            row_black_reason,
            column_black_reason,
        )
        return GridStructure("unavailable", (), ()), diagnostics
    rows = _map_centers(
        row_centers,
        analysis_image.height,
        source_region.y1,
        source_region.height,
        include_outer=not row_is_black,
    )
    columns = _map_centers(
        column_centers,
        analysis_image.width,
        source_region.x1,
        source_region.width,
        include_outer=not column_is_black,
    )
    row_cells_ok, row_cell_reason = _boundaries_have_reasonable_cells(
        rows, config.grid_max_cell_span_ratio
    )
    column_cells_ok, column_cell_reason = _boundaries_have_reasonable_cells(
        columns, config.grid_max_cell_span_ratio
    )
    diagnostics = V6GridDiagnostics(
        row_source,
        column_source,
        tuple(black_rows),
        tuple(black_columns),
        tuple(white_rows),
        tuple(white_columns),
        f"{row_black_reason}；{row_cell_reason}",
        f"{column_black_reason}；{column_cell_reason}",
    )
    if len(rows) < 2 or len(columns) < 2 or not row_cells_ok or not column_cells_ok:
        return GridStructure("unavailable", (), ()), diagnostics
    return (
        GridStructure(f"v6:rows={row_source};columns={column_source}", rows, columns),
        diagnostics,
    )
