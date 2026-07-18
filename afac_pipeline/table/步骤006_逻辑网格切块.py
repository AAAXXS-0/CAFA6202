"""沿表格逻辑行列边界规划结构化请求图片。"""

from __future__ import annotations

import math

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


def _balanced_axis_groups(
    boundaries: tuple[int, ...],
    max_side: int,
    repeated_count: int,
    group_count: int,
) -> list[tuple[int, int, int]]:
    """把余数均匀分到各组，避免最后只剩一两行或一两列。"""

    logical_count = len(boundaries) - 1
    if not 1 <= group_count <= logical_count:
        return []
    base, remainder = divmod(logical_count, group_count)
    sizes = [base + 1] * remainder + [base] * (group_count - remainder)
    groups: list[tuple[int, int, int]] = []
    start = 0
    for size in sizes:
        end = start + size
        context_count = min(repeated_count, logical_count) if start > 0 else 0
        context_size = boundaries[context_count] - boundaries[0]
        if boundaries[end] - boundaries[start] + context_size > max_side:
            return []
        groups.append((start, end, context_count))
        start = end
    return groups


def _balanced_grid_groups(
    row_boundaries: tuple[int, ...],
    column_boundaries: tuple[int, ...],
    max_side: int,
    repeat_header_rows: int,
    repeat_stub_columns: int,
    max_cells: int,
    preferred_min_cells: int,
    max_aspect_ratio: float,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """搜索切片最少且没有极端宽高比、小余数块尽量少的均衡方案。"""

    logical_rows = len(row_boundaries) - 1
    logical_columns = len(column_boundaries) - 1
    whole_width = column_boundaries[-1] - column_boundaries[0]
    whole_height = row_boundaries[-1] - row_boundaries[0]
    whole_aspect_ratio = max(
        whole_width / whole_height,
        whole_height / whole_width,
    )
    # 一张极端细长表即使格子数很少，也至少沿长边切到
    # “整体宽高比 / 允许宽高比”个子块。这里只规定最少块数，
    # 具体沿行还是列切，仍由下面的宽高比和小碎片评分决定。
    minimum_aspect_tile_count = max(
        1,
        math.ceil(whole_aspect_ratio / max_aspect_ratio),
    )
    row_options = [
        groups
        for count in range(1, logical_rows + 1)
        if (groups := _balanced_axis_groups(
            row_boundaries, max_side, repeat_header_rows, count
        ))
    ]
    column_options = [
        groups
        for count in range(1, logical_columns + 1)
        if (groups := _balanced_axis_groups(
            column_boundaries, max_side, repeat_stub_columns, count
        ))
    ]

    best_score: tuple[int, int, int, int, float, int] | None = None
    best_groups: tuple[
        list[tuple[int, int, int]],
        list[tuple[int, int, int]],
    ] | None = None
    for row_groups in row_options:
        row_counts = [end - start for start, end, _ in row_groups]
        row_heights = [
            row_boundaries[end] - row_boundaries[start]
            + row_boundaries[context] - row_boundaries[0]
            for start, end, context in row_groups
        ]
        for column_groups in column_options:
            column_counts = [end - start for start, end, _ in column_groups]
            if max(row_counts) * max(column_counts) > max_cells:
                continue
            column_widths = [
                column_boundaries[end] - column_boundaries[start]
                + column_boundaries[context] - column_boundaries[0]
                for start, end, context in column_groups
            ]
            aspect_ratio = max(
                max(column_widths) / min(row_heights),
                max(row_heights) / min(column_widths),
            )
            aspect_violation = int(aspect_ratio > max_aspect_ratio)
            tile_count = len(row_groups) * len(column_groups)
            if tile_count < minimum_aspect_tile_count:
                continue
            if (
                best_score is not None
                and (aspect_violation, tile_count) > best_score[:2]
            ):
                continue
            cell_counts = [
                row_count * column_count
                for row_count in row_counts
                for column_count in column_counts
            ]
            tiny_count = sum(
                count < preferred_min_cells for count in cell_counts
            )
            narrow_count = sum(
                min(row_count, column_count) < 8
                for row_count in row_counts
                for column_count in column_counts
            )
            score = (
                aspect_violation,
                tile_count,
                tiny_count,
                narrow_count,
                round(aspect_ratio, 6),
                max(cell_counts) - min(cell_counts),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_groups = (row_groups, column_groups)
    return best_groups or ([], [])




def plan_grid_tiles(
    region: Box,
    region_index: int,
    row_boundaries: tuple[int, ...],
    column_boundaries: tuple[int, ...],
    max_side: int,
    single_tile_min_scale: float,
    repeat_header_rows: int,
    repeat_stub_columns: int,
    max_logical_cells_per_tile: int = 280,
    preferred_min_logical_cells_per_tile: int = 80,
    max_tile_aspect_ratio: float = 8.0,
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
    whole_aspect_ratio = max(
        region.width / region.height, region.height / region.width
    )
    if (
        whole_scale >= single_tile_min_scale
        and logical_cell_count <= max_logical_cells_per_tile
        and whole_aspect_ratio <= max_tile_aspect_ratio
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

    # 优先均匀分配行列余数，避免“前面装满、最后剩 6×6”。
    row_groups, column_groups = _balanced_grid_groups(
        row_boundaries,
        column_boundaries,
        max_side,
        repeat_header_rows,
        repeat_stub_columns,
        max_logical_cells_per_tile,
        preferred_min_logical_cells_per_tile,
        max_tile_aspect_ratio,
    )
    if not row_groups or not column_groups:
        # 某些单元格本身就超过像素上限。均衡规划失败时保留旧贪心算法
        # 作为最后兜底，若仍无法规划则由上层改用像素重叠切片。
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
