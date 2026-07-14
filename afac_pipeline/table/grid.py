"""从表格区域分析图中提取横线、竖线和逻辑行列边界。"""

from __future__ import annotations

from dataclasses import dataclass

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

    只有横纵两个方向都达到最少线数时才声明逻辑网格可靠。无边框表格、
    背景噪声很重的扫描件会返回 unavailable，调用方随后使用像素重叠
    兜底，不会凭不可靠的文字间隙伪造行列坐标。
    """

    gray = np.asarray(analysis_image.convert("L"))
    ink = gray < config.grid_white_threshold
    horizontal = _line_centers(ink.mean(axis=1), config.grid_line_min_ratio)
    vertical = _line_centers(ink.mean(axis=0), config.grid_line_min_ratio)
    if (
        len(horizontal) < config.grid_min_line_count
        or len(vertical) < config.grid_min_line_count
    ):
        return GridStructure("unavailable", (), ())

    rows = _local_boundaries(horizontal, analysis_image.height, config.grid_min_cell_size)
    columns = _local_boundaries(vertical, analysis_image.width, config.grid_min_cell_size)
    if len(rows) < 2 or len(columns) < 2:
        return GridStructure("unavailable", (), ())
    return GridStructure(
        source="ruled-lines",
        row_boundaries=_map_boundaries(
            rows, analysis_image.height, source_region.y1, source_region.height
        ),
        column_boundaries=_map_boundaries(
            columns, analysis_image.width, source_region.x1, source_region.width
        ),
    )
