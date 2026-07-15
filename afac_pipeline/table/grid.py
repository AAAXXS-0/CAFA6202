"""从表格区域分析图中提取横线、竖线和逻辑行列边界。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..common.models import Box
from .config import TableConfig


@dataclass(frozen=True)
class GridStructure:
    """使用原图绝对坐标保存表格的逻辑网格。

    boundaries 采用左闭右开坐标。相邻两个边界之间是一行或一列，因此
    行数等于 len(row_boundaries) - 1。
    """

    source: str
    row_boundaries: tuple[int, ...]
    column_boundaries: tuple[int, ...]

    @property
    def available(self) -> bool:
        return len(self.row_boundaries) >= 2 and len(self.column_boundaries) >= 2

    @property
    def row_count(self) -> int:
        return max(0, len(self.row_boundaries) - 1)

    @property
    def column_count(self) -> int:
        return max(0, len(self.column_boundaries) - 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "row_boundaries": list(self.row_boundaries),
            "column_boundaries": list(self.column_boundaries),
            "row_count": self.row_count,
            "column_count": self.column_count,
        }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks] + 1, indices[-1] + 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _line_centers(ratio: np.ndarray, minimum_ratio: float) -> list[int]:
    """把有宽度的印刷线压缩成单个中心坐标，避免一条粗线被算成多条。"""

    return [
        round((start + end - 1) / 2)
        for start, end in _runs(ratio >= minimum_ratio)
    ]


def _blank_band_centers(
    ratio: np.ndarray, maximum_ratio: float, minimum_band: int
) -> list[int]:
    """只保留足够长的连续空白带，短空隙通常只是字内或单元格内间距。"""

    return [
        round((start + end - 1) / 2)
        for start, end in _runs(ratio <= maximum_ratio)
        if end - start >= minimum_band
    ]


def _whitespace_dilate_kernels(
    ink: np.ndarray, config: TableConfig
) -> tuple[int, int]:
    """分别计算找横向、纵向白带时的文字扩张长度。"""

    horizontal_ratio = (
        config.whitespace_horizontal_dilate_ratio
        if config.whitespace_horizontal_dilate_ratio is not None
        else config.whitespace_dilate_ratio
    )
    vertical_ratio = (
        config.whitespace_vertical_dilate_ratio
        if config.whitespace_vertical_dilate_ratio is not None
        else config.whitespace_dilate_ratio
    )
    return (
        max(3, round(ink.shape[1] * horizontal_ratio)),
        max(3, round(ink.shape[0] * vertical_ratio)),
    )


def _whitespace_centers(
    ink: np.ndarray, config: TableConfig
) -> tuple[list[int], list[int]]:
    """无线表格兜底：扩张文字后，再寻找贯穿整行或整列的长空白带。"""

    binary = ink.astype(np.uint8)
    horizontal_kernel, vertical_kernel = _whitespace_dilate_kernels(
        ink, config
    )
    for_rows = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel, 1)),
    )
    for_columns = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel)),
    )
    rows = _blank_band_centers(
        for_rows.mean(axis=1), config.whitespace_blank_ratio, config.whitespace_min_band
    )
    columns = _blank_band_centers(
        for_columns.mean(axis=0), config.whitespace_blank_ratio, config.whitespace_min_band
    )
    return rows, columns


def _local_boundaries(
    line_centers: list[int],
    length: int,
    minimum_cell_size: int,
) -> list[int]:
    """将网格线中心转换成完整边界，并吸附靠近区域边缘的外框线。"""

    edge_tolerance = max(6, minimum_cell_size)
    interior = [
        center
        for center in line_centers
        if edge_tolerance < center < length - edge_tolerance
    ]
    candidates = [0, *interior, length]
    filtered = [candidates[0]]
    for value in candidates[1:]:
        if value - filtered[-1] >= minimum_cell_size:
            filtered.append(value)
        elif value == length:
            filtered[-1] = value
    return filtered


def _map_boundaries(
    local: list[int],
    preview_length: int,
    source_start: int,
    source_length: int,
) -> tuple[int, ...]:
    mapped = [
        source_start + round(value * source_length / preview_length)
        for value in local
    ]
    mapped[0] = source_start
    mapped[-1] = source_start + source_length
    return tuple(dict.fromkeys(mapped))


def detect_grid_structure(
    analysis_image: Image.Image,
    source_region: Box,
    config: TableConfig,
) -> GridStructure:
    """检测同时贯穿较大区域的横线和竖线。

    每个方向都先尝试传统长直线；某个方向没有可靠直线时，才对该方向
    使用长空白带。这样有线表格不会被行列间空白干扰，无线表格仍有兜底。
    """

    gray = np.asarray(analysis_image.convert("L"))
    ink = gray < config.grid_white_threshold
    horizontal = _line_centers(ink.mean(axis=1), config.grid_line_min_ratio)
    vertical = _line_centers(ink.mean(axis=0), config.grid_line_min_ratio)
    horizontal_lines = len(horizontal) >= config.grid_min_line_count
    vertical_lines = len(vertical) >= config.grid_min_line_count
    whitespace_rows, whitespace_columns = _whitespace_centers(ink, config)
    row_centers = horizontal if horizontal_lines else whitespace_rows
    column_centers = vertical if vertical_lines else whitespace_columns
    if not row_centers or not column_centers:
        return GridStructure("unavailable", (), ())

    rows = _local_boundaries(row_centers, analysis_image.height, config.grid_min_cell_size)
    columns = _local_boundaries(column_centers, analysis_image.width, config.grid_min_cell_size)
    if len(rows) < 2 or len(columns) < 2:
        return GridStructure("unavailable", (), ())
    if horizontal_lines and vertical_lines:
        source = "ruled-lines"
    elif horizontal_lines or vertical_lines:
        source = "hybrid-lines-whitespace"
    else:
        source = "whitespace"
    return GridStructure(
        source=source,
        row_boundaries=_map_boundaries(
            rows, analysis_image.height, source_region.y1, source_region.height
        ),
        column_boundaries=_map_boundaries(
            columns, analysis_image.width, source_region.x1, source_region.width
        ),
    )
