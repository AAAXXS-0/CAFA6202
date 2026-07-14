"""在低清墨水密度图上寻找同图异表之间的宽低密度区域。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.models import Box


@dataclass(frozen=True)
class DensityBand:
    axis: str
    start: int
    end: int
    mean_density: float

    @property
    def center(self) -> int:
        return round((self.start + self.end - 1) / 2)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks] + 1, indices[-1] + 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _fill_short_interruptions(low: np.ndarray, maximum_length: int) -> np.ndarray:
    """允许表标题等少量墨水穿过大空白区，不让它把空白带截成两段。"""

    result = low.copy()
    for start, end in _runs(~low):
        if (
            end - start <= maximum_length
            and start > 0
            and end < len(low)
            and low[start - 1]
            and low[end]
        ):
            result[start:end] = True
    return result


def _profile_bands(
    profile: np.ndarray,
    axis: str,
    *,
    maximum_density: float,
    minimum_band_ratio: float,
    maximum_interrupt_ratio: float,
    edge_margin_ratio: float,
    content_density: float,
) -> list[DensityBand]:
    length = len(profile)
    minimum_band = max(3, round(length * minimum_band_ratio))
    maximum_interrupt = max(1, round(length * maximum_interrupt_ratio))
    edge_margin = max(1, round(length * edge_margin_ratio))
    low = _fill_short_interruptions(profile <= maximum_density, maximum_interrupt)
    bands: list[DensityBand] = []
    for start, end in _runs(low):
        if end - start < minimum_band:
            continue
        if start < edge_margin or end > length - edge_margin:
            continue
        # 上一步允许少量高密度位置（例如跨过空白区的表标题）把空白带短暂
        # 打断。但“打断很短”不代表整条带仍然足够空，因此合并之后必须再
        # 看一次平均密度。这里允许最高为逐行阈值的两倍：真正的表间空白可
        # 以容纳少量标题，而正文行、正文列不会仅凭若干低密度点被误判成分表带。
        band_mean = float(profile[start:end].mean())
        if band_mean > maximum_density * 2:
            continue
        support = max(3, round(length * 0.05))
        before = profile[max(0, start - support) : start]
        after = profile[end : min(length, end + support)]
        if before.size == 0 or after.size == 0:
            continue
        if float(before.mean()) < content_density or float(after.mean()) < content_density:
            continue
        bands.append(
            DensityBand(
                axis=axis,
                start=start,
                end=end,
                mean_density=band_mean,
            )
        )
    return bands


def find_density_bands(
    density: np.ndarray,
    *,
    maximum_density: float = 0.01,
    minimum_band_ratio: float = 0.02,
    maximum_interrupt_ratio: float = 0.012,
    edge_margin_ratio: float = 0.03,
    content_density: float = 0.02,
) -> tuple[list[DensityBand], list[DensityBand]]:
    """返回横向和纵向的宽低密度带；它们是分表候选，不是单元格空白。"""

    rows = _profile_bands(
        density.mean(axis=1),
        "horizontal",
        maximum_density=maximum_density,
        minimum_band_ratio=minimum_band_ratio,
        maximum_interrupt_ratio=maximum_interrupt_ratio,
        edge_margin_ratio=edge_margin_ratio,
        content_density=content_density,
    )
    columns = _profile_bands(
        density.mean(axis=0),
        "vertical",
        maximum_density=maximum_density,
        minimum_band_ratio=minimum_band_ratio,
        maximum_interrupt_ratio=maximum_interrupt_ratio,
        edge_margin_ratio=edge_margin_ratio,
        content_density=content_density,
    )
    return rows, columns


def boxes_from_bands(
    width: int,
    height: int,
    horizontal: list[DensityBand],
    vertical: list[DensityBand],
    density: np.ndarray,
    minimum_content_density: float = 0.01,
) -> list[Box]:
    """按候选带中心形成网格，并删除几乎没有内容的格子。"""

    xs = [0, *sorted({band.center for band in vertical}), width]
    ys = [0, *sorted({band.center for band in horizontal}), height]
    boxes: list[Box] = []
    for y1, y2 in zip(ys, ys[1:]):
        for x1, x2 in zip(xs, xs[1:]):
            if x2 <= x1 or y2 <= y1:
                continue
            if float(density[y1:y2, x1:x2].mean()) < minimum_content_density:
                continue
            boxes.append(Box(x1, y1, x2, y2))
    return boxes
