"""把原图表格区域规划为符合视觉模型尺寸限制的切片。"""

from __future__ import annotations

import math

from ..common.models import Box, TilePlan


def _axis_segments(length: int, tile_length: int, overlap: int) -> list[tuple[int, int]]:
    """用最少块覆盖一条轴，并尽量保留目标重叠。

    当最少块无法同时容纳目标重叠时，自动降低重叠；不会为了满足重叠而
    生成一个几乎与前块完全相同的尾块。
    """

    if length <= tile_length:
        return [(0, length)]
    if overlap < 0 or overlap >= tile_length:
        raise ValueError("切片重叠不能大于等于切片长度")
    count = math.ceil(length / tile_length)
    maximum_overlap = max(0, (count * tile_length - length) // (count - 1))
    actual_overlap = min(overlap, maximum_overlap)
    segment_length = math.ceil(
        (length + (count - 1) * actual_overlap) / count
    )
    stride = segment_length - actual_overlap
    segments: list[tuple[int, int]] = []
    for index in range(count):
        start = min(index * stride, length - segment_length)
        end = min(length, start + segment_length)
        segments.append((start, end))
    return list(dict.fromkeys(segments))


def plan_region_tiles(
    region: Box,
    region_index: int,
    max_side: int,
    overlap: int,
    single_tile_min_scale: float,
) -> list[TilePlan]:
    """规划切片。

    中等尺寸表格优先整体等比缩小，避免破坏表格拓扑；缩放比例过低时才进行
    二维切片。切片坐标始终保留在原图坐标系中，便于后续审计和重新切图。
    """

    whole_scale = min(1.0, max_side / max(region.width, region.height))
    if whole_scale >= single_tile_min_scale:
        return [
            TilePlan(
                region_index=region_index,
                row_index=0,
                column_index=0,
                row_count=1,
                column_count=1,
                source_box=region,
                output_width=max(1, round(region.width * whole_scale)),
                output_height=max(1, round(region.height * whole_scale)),
                scale=whole_scale,
                file_name=f"region_{region_index:03d}_r000_c000.png",
            )
        ]

    x_segments = _axis_segments(region.width, max_side, overlap)
    y_segments = _axis_segments(region.height, max_side, overlap)
    plans: list[TilePlan] = []
    for row_index, (local_y1, local_y2) in enumerate(y_segments):
        for column_index, (local_x1, local_x2) in enumerate(x_segments):
            source_box = Box(
                region.x1 + local_x1,
                region.y1 + local_y1,
                region.x1 + local_x2,
                region.y1 + local_y2,
            )
            plans.append(
                TilePlan(
                    region_index=region_index,
                    row_index=row_index,
                    column_index=column_index,
                    row_count=len(y_segments),
                    column_count=len(x_segments),
                    source_box=source_box,
                    output_width=source_box.width,
                    output_height=source_box.height,
                    scale=1.0,
                    file_name=(
                        f"region_{region_index:03d}_r{row_index:03d}_c{column_index:03d}.png"
                    ),
                )
            )
    return plans
