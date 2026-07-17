"""图表 v6：固定比例密度分表与黑线优先、白带兜底的结构检测。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..common.models import Box, DetectedBox
from .config import TableConfig
from .density_split import DensityBand, _runs, boxes_from_bands, find_density_bands
from .grid import GridStructure, _whitespace_centers
from .ink_region import InkRegionResult, detect_ink_regions


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

    def to_dict(self) -> dict[str, object]:
        return {
            "row_source": self.row_source,
            "column_source": self.column_source,
            "black_rows": [line.__dict__ for line in self.black_rows],
            "black_columns": [line.__dict__ for line in self.black_columns],
            "white_rows": list(self.white_rows),
            "white_columns": list(self.white_columns),
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
        local_ink = gray[split_box.y1:split_box.y2, split_box.x1:split_box.x2] < config.ink_threshold
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
) -> list[LineSegment]:
    """只接受在完整表格跨度内黑像素覆盖率达到 90% 的候选线。"""

    data = black_ink if axis == 0 else black_ink.T
    envelope = envelope_mask if axis == 0 else envelope_mask.T
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
        if axis == 0:
            scores[index] = float(black_ink[index, start:end].mean())
        else:
            scores[index] = float(black_ink[start:end, index].mean())

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
    centers: list[int], preview_length: int, source_start: int, source_length: int,
    *, include_outer: bool,
) -> tuple[int, ...]:
    """映射检测边界；只有白带模式才使用内容框外沿补齐首尾。"""

    detected = sorted(set(centers))
    local = [0, *detected, preview_length] if include_outer else detected
    mapped = [
        source_start + round(value * source_length / preview_length)
        for value in local
    ]
    if include_outer:
        mapped[0] = source_start
        mapped[-1] = source_start + source_length
    return tuple(dict.fromkeys(mapped))


def detect_v6_grid(
    analysis_image: Image.Image,
    source_region: Box,
    config: TableConfig,
) -> tuple[GridStructure, V6GridDiagnostics]:
    """对一张已分开的表检测逻辑行列边界，并映射回原图坐标。"""

    gray = np.asarray(analysis_image.convert("L"))
    ink = gray < config.grid_white_threshold
    envelope = content_envelope_mask(ink)
    black_rows = adaptive_line_segments(
        ink, envelope, 0, config.grid_black_line_ratio
    )
    black_columns = adaptive_line_segments(
        ink, envelope, 1, config.grid_black_line_ratio
    )
    white_rows, white_columns = _whitespace_centers(ink, config)
    reliable = config.grid_reliable_line_count
    row_is_black = len(black_rows) >= reliable
    column_is_black = len(black_columns) >= reliable
    row_centers = [line.position for line in black_rows] if row_is_black else white_rows
    column_centers = [line.position for line in black_columns] if column_is_black else white_columns
    row_source = f"black-line-{config.grid_black_line_ratio:.2f}" if row_is_black else "white-band"
    column_source = f"black-line-{config.grid_black_line_ratio:.2f}" if column_is_black else "white-band"
    diagnostics = V6GridDiagnostics(
        row_source,
        column_source,
        tuple(black_rows),
        tuple(black_columns),
        tuple(white_rows),
        tuple(white_columns),
    )
    if not row_centers or not column_centers:
        return GridStructure("unavailable", (), ()), diagnostics
    rows = _map_centers(
        row_centers, analysis_image.height, source_region.y1, source_region.height,
        include_outer=not row_is_black,
    )
    columns = _map_centers(
        column_centers, analysis_image.width, source_region.x1, source_region.width,
        include_outer=not column_is_black,
    )
    if len(rows) < 2 or len(columns) < 2:
        return GridStructure("unavailable", (), ()), diagnostics
    return (
        GridStructure(
            f"v6:rows={row_source};columns={column_source}", rows, columns
        ),
        diagnostics,
    )
