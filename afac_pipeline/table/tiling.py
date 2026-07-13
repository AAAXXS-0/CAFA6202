"""把原图表格区域规划为符合视觉模型尺寸限制的切片。"""

from __future__ import annotations

import math

from ..common.models import Box, TilePlan


def _axis_starts(length: int, tile_length: int, overlap: int) -> list[int]:
    if length <= tile_length:
        return [0]
    stride = tile_length - overlap
    if stride <= 0:
        raise ValueError("切片重叠不能大于等于切片长度")
    count = 1 + math.ceil((length - tile_length) / stride)
    starts = [min(index * stride, length - tile_length) for index in range(count)]
    # 最后一块被回推到末端后可能与前一块起点相同，按顺序去重。
    return list(dict.fromkeys(starts))


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

    x_starts = _axis_starts(region.width, max_side, overlap)
    y_starts = _axis_starts(region.height, max_side, overlap)
    plans: list[TilePlan] = []
    for row_index, local_y in enumerate(y_starts):
        for column_index, local_x in enumerate(x_starts):
            width = min(max_side, region.width - local_x)
            height = min(max_side, region.height - local_y)
            source_box = Box(
                region.x1 + local_x,
                region.y1 + local_y,
                region.x1 + local_x + width,
                region.y1 + local_y + height,
            )
            plans.append(
                TilePlan(
                    region_index=region_index,
                    row_index=row_index,
                    column_index=column_index,
                    row_count=len(y_starts),
                    column_count=len(x_starts),
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
