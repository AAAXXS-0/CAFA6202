"""沿表格逻辑行列边界规划结构化请求图片。"""

from __future__ import annotations

from ..common.models import Box, TilePlan


def _axis_groups(
    boundaries: tuple[int, ...], max_side: int, repeated_count: int,
    maximum_logical_count: int | None = None,
) -> list[tuple[int, int, int]]:
    logical_count = len(boundaries) - 1
    groups: list[tuple[int, int, int]] = []
    start = 0
    while start < logical_count:
        context_count = min(repeated_count, logical_count) if start > 0 else 0
        available = max_side - (boundaries[context_count] - boundaries[0])
        end = start + 1
        while (
            end <= logical_count
            and boundaries[end] - boundaries[start] <= available
            and (
                maximum_logical_count is None
                or end - start <= maximum_logical_count
            )
        ):
            end += 1
        end -= 1
        if available <= 0 or end <= start:
            return []
        groups.append((start, end, context_count))
        start = end
    return groups




def plan_grid_tiles(
    region: Box,
    region_index: int,
    row_boundaries: tuple[int, ...],
    column_boundaries: tuple[int, ...],
    max_side: int,
    single_tile_min_scale: float,
    repeat_header_rows: int,
    repeat_stub_columns: int,
    max_logical_cells_per_tile: int = 320,
) -> list[TilePlan]:
    """优先整表缩放；必须切分时只在完整的逻辑行列边界落刀。"""

    if (row_boundaries[0], row_boundaries[-1]) != (region.y1, region.y2):
        raise ValueError("行边界没有覆盖完整表格区域")
    if (column_boundaries[0], column_boundaries[-1]) != (region.x1, region.x2):
        raise ValueError("列边界没有覆盖完整表格区域")
    logical_rows = len(row_boundaries) - 1
    logical_columns = len(column_boundaries) - 1
    whole_scale = min(1.0, max_side / max(region.width, region.height))
    logical_cell_count = logical_rows * logical_columns
    if (
        whole_scale >= single_tile_min_scale
        and logical_cell_count <= max_logical_cells_per_tile
    ):
        return [
            TilePlan(
                region_index, 0, 0, 1, 1, region,
                max(1, round(region.width * whole_scale)),
                max(1, round(region.height * whole_scale)), whole_scale,
                f"region_{region_index:03d}_r000_c000.png",
                logical_row_end=logical_rows,
                logical_column_end=logical_columns,
                tiling_mode="logical_grid",
            )
        ]

    # 超过输出预算时同时沿完整行列边界切块，绝不通过非等比例拉伸凑尺寸。
    row_cap = max(1, int(max_logical_cells_per_tile ** 0.5))
    column_cap = max(1, max_logical_cells_per_tile // row_cap)
    row_groups = _axis_groups(
        row_boundaries, max_side, repeat_header_rows, row_cap
    )
    column_groups = _axis_groups(
        column_boundaries, max_side, repeat_stub_columns, column_cap
    )
    if not row_groups or not column_groups:
        return []
    plans: list[TilePlan] = []
    for row_index, (row_start, row_end, header_rows) in enumerate(row_groups):
        for column_index, (column_start, column_end, stub_columns) in enumerate(column_groups):
            body = Box(
                column_boundaries[column_start], row_boundaries[row_start],
                column_boundaries[column_end], row_boundaries[row_end],
            )
            context_width = column_boundaries[stub_columns] - column_boundaries[0]
            context_height = row_boundaries[header_rows] - row_boundaries[0]
            plans.append(
                TilePlan(
                    region_index, row_index, column_index,
                    len(row_groups), len(column_groups), body,
                    body.width + context_width, body.height + context_height, 1.0,
                    f"region_{region_index:03d}_r{row_index:03d}_c{column_index:03d}.png",
                    logical_row_start=row_start,
                    logical_row_end=row_end,
                    logical_column_start=column_start,
                    logical_column_end=column_end,
                    header_context_rows=header_rows,
                    stub_context_columns=stub_columns,
                    tiling_mode="logical_grid",
                )
            )
    return plans
