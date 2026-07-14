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
    maximum_band_mean: float,
    support_ratio: float,
) -> list[DensityBand]:
    length = len(profile)
    minimum_band = max(3, round(length * minimum_band_ratio))
    # 低清图中标题造成的中断通常只有 1～2 像素。超过 2 像素后更可能已经
    # 进入正文或相邻单元格，因此即使图片更长也不继续放宽。
    maximum_interrupt = max(
        1, min(2, round(length * maximum_interrupt_ratio))
    )
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
        # 看一次平均密度。真正的表间空白可以容纳少量标题，但整条带的平均
        # 墨水仍应很低；否则正文行、正文列会被误判成分表带。
        band_mean = float(profile[start:end].mean())
        if band_mean > maximum_band_mean:
            continue
        support = max(3, round(length * support_ratio))
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


def _cluster_nearby_bands(
    bands: list[DensityBand], maximum_span: int
) -> list[list[DensityBand]]:
    """把同一处宽空白附近的数条窄低谷归为一个候选区。

    分表带附近偶尔还会有标题下方的短空行。这里以每组第一条低谷为锚点，
    避免使用链式合并把一整排单元格空隙串成一个超宽候选区。
    """

    if not bands:
        return []
    groups: list[list[DensityBand]] = []
    current = [bands[0]]
    anchor = bands[0].center
    for band in bands[1:]:
        if band.center - anchor <= maximum_span:
            current.append(band)
        else:
            groups.append(current)
            current = [band]
            anchor = band.center
    groups.append(current)
    return groups


def _select_axis_bands(
    bands: list[DensityBand],
    *,
    coarse_max_side: int,
    cluster_ratio: float,
    minimum_region_ratio: float,
) -> list[DensityBand]:
    """删除明显属于表内密集行列空隙的候选，并保留每个分隔区的主低谷。"""

    groups = _cluster_nearby_bands(
        bands, max(4, round(coarse_max_side * cluster_ratio))
    )
    selected = [
        max(group, key=lambda band: (band.end - band.start, -band.mean_density))
        for group in groups
    ]
    if len(selected) >= 2:
        spacing = float(np.median(np.diff([band.center for band in selected])))
        # 如果候选低谷像单元格一样密集重复，它们是内部行列而不是同图异表。
        if spacing < coarse_max_side * minimum_region_ratio:
            return []

    # 三块呈“小块—大块—小块”且两条低谷只差一个像素时，较窄低谷很容易
    # 是表头与正文之间的内部留白。只在带宽确实不同的情况下保留更强的一条；
    # 两条等宽低谷（0cd74f08）仍然都是真正分表线。
    if len(selected) == 2:
        widths = [band.end - band.start for band in selected]
        if widths[0] != widths[1]:
            selected = [selected[int(widths[1] > widths[0])]]
    return selected


def find_density_bands(
    density: np.ndarray,
    *,
    maximum_density: float = 0.03,
    minimum_band_ratio: float = 0.007,
    maximum_interrupt_ratio: float = 0.008,
    edge_margin_ratio: float = 0.025,
    content_density: float = 0.03,
    maximum_band_mean: float = 0.02,
    support_ratio: float = 0.03,
    cluster_ratio: float = 0.08,
    minimum_region_ratio: float = 0.12,
) -> tuple[list[DensityBand], list[DensityBand]]:
    """返回单一主方向上的分表带，不把密集单元格空隙当成同图异表。"""

    rows = _profile_bands(
        density.mean(axis=1),
        "horizontal",
        maximum_density=maximum_density,
        minimum_band_ratio=minimum_band_ratio,
        maximum_interrupt_ratio=maximum_interrupt_ratio,
        edge_margin_ratio=edge_margin_ratio,
        content_density=content_density,
        maximum_band_mean=maximum_band_mean,
        support_ratio=support_ratio,
    )
    columns = _profile_bands(
        density.mean(axis=0),
        "vertical",
        maximum_density=maximum_density,
        minimum_band_ratio=minimum_band_ratio,
        maximum_interrupt_ratio=maximum_interrupt_ratio,
        edge_margin_ratio=edge_margin_ratio,
        content_density=content_density,
        maximum_band_mean=maximum_band_mean,
        support_ratio=support_ratio,
    )
    coarse_max_side = max(density.shape)
    rows = _select_axis_bands(
        rows,
        coarse_max_side=coarse_max_side,
        cluster_ratio=cluster_ratio,
        minimum_region_ratio=minimum_region_ratio,
    )
    columns = _select_axis_bands(
        columns,
        coarse_max_side=coarse_max_side,
        cluster_ratio=cluster_ratio,
        minimum_region_ratio=minimum_region_ratio,
    )
    row_score = sum(
        (band.end - band.start) * (maximum_band_mean - band.mean_density)
        for band in rows
    )
    column_score = sum(
        (band.end - band.start) * (maximum_band_mean - band.mean_density)
        for band in columns
    )
    if row_score >= column_score:
        return rows, []
    return [], columns


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
