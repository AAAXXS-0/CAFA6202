import unittest

import numpy as np
from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤004_网格与白带检测 import (
    _whitespace_dilate_kernels,
    detect_grid_structure,
)
from afac_pipeline.table.步骤006_逻辑网格切块 import plan_grid_tiles


class TableGridTest(unittest.TestCase):
    def test_horizontal_and_vertical_whitespace_dilation_can_differ(self) -> None:
        ink = np.zeros((900, 1200), dtype=bool)
        horizontal, vertical = _whitespace_dilate_kernels(
            ink,
            TableConfig(
                whitespace_dilate_ratio=0.004,
                whitespace_horizontal_dilate_ratio=0.0015,
                whitespace_vertical_dilate_ratio=0.004,
            ),
        )
        self.assertEqual(horizontal, 3)
        self.assertEqual(vertical, 4)

    def test_ruled_grid_is_mapped_to_source_coordinates(self) -> None:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        for y in (0, 100, 200, 300, 399):
            draw.line((0, y, 599, y), fill="black", width=3)
        for x in (0, 150, 300, 450, 599):
            draw.line((x, 0, x, 399), fill="black", width=3)
        grid = detect_grid_structure(
            image,
            Box(1000, 2000, 7000, 6000),
            TableConfig(grid_line_min_ratio=0.8),
        )
        self.assertTrue(grid.available)
        self.assertEqual(grid.row_count, 4)
        self.assertEqual(grid.column_count, 4)
        self.assertEqual(grid.row_boundaries[0], 2000)
        self.assertEqual(grid.column_boundaries[-1], 7000)

    def test_large_grid_repeats_header_and_stub_without_exceeding_limit(self) -> None:
        boundaries = tuple(range(0, 6001, 1000))
        plans = plan_grid_tiles(
            Box(0, 0, 6000, 6000),
            0,
            boundaries,
            boundaries,
            max_side=3000,
            repeat_header_rows=1,
            repeat_stub_columns=1,
        )
        self.assertGreater(len(plans), 1)
        self.assertTrue(any(plan.header_context_rows == 1 for plan in plans))
        self.assertTrue(any(plan.stub_context_columns == 1 for plan in plans))
        self.assertTrue(all(plan.output_width <= 3000 for plan in plans))
        self.assertTrue(all(plan.output_height <= 3000 for plan in plans))

    def test_output_budget_splits_wide_table_in_two_dimensions(self) -> None:
        rows = tuple(range(0, 1001, 100))
        columns = tuple(range(0, 4001, 100))
        plans = plan_grid_tiles(
            Box(0, 0, 4000, 1000),
            0,
            rows,
            columns,
            max_side=3900,
            repeat_header_rows=0,
            repeat_stub_columns=0,
            max_logical_cells_per_tile=320,
        )
        self.assertGreater(len(plans), 1)
        self.assertTrue(all(plan.column_count > 1 for plan in plans))
        self.assertTrue(any(plan.logical_column_end < 40 for plan in plans))
        self.assertTrue(all(plan.header_context_rows == 0 for plan in plans))
        self.assertTrue(all(plan.output_width <= 3900 for plan in plans))

    def test_default_output_budget_matches_firered_safe_limit(self) -> None:
        self.assertEqual(TableConfig().max_logical_cells_per_tile, 140)
        boundaries = tuple(range(0, 1801, 100))
        plans = plan_grid_tiles(
            Box(0, 0, 1800, 1800),
            0,
            boundaries,
            boundaries,
            max_side=3900,
            repeat_header_rows=0,
            repeat_stub_columns=0,
        )
        self.assertGreater(len(plans), 1)
        self.assertTrue(
            all(
                (plan.logical_row_end - plan.logical_row_start)
                * (plan.logical_column_end - plan.logical_column_start)
                <= 140
                for plan in plans
            )
        )

    def test_balanced_split_avoids_small_remainder_tiles(self) -> None:
        rows = tuple(round(index * 350 / 23) for index in range(24))
        columns = tuple(round(index * 1315 / 24) for index in range(25))
        plans = plan_grid_tiles(
            Box(0, 0, 1315, 350),
            0,
            rows,
            columns,
            max_side=3900,
            repeat_header_rows=0,
            repeat_stub_columns=0,
        )
        shapes = [
            (
                plan.logical_row_end - plan.logical_row_start,
                plan.logical_column_end - plan.logical_column_start,
            )
            for plan in plans
        ]
        self.assertEqual(shapes, [(23, 6), (23, 6), (23, 6), (23, 6)])

    def test_extreme_aspect_ratio_is_split_even_below_cell_budget(self) -> None:
        rows = tuple(round(index * 130 / 8) for index in range(9))
        columns = tuple(round(index * 1679 / 24) for index in range(25))
        plans = plan_grid_tiles(
            Box(0, 0, 1679, 130),
            0,
            rows,
            columns,
            max_side=3900,
            repeat_header_rows=0,
            repeat_stub_columns=0,
        )
        shapes = [
            (
                plan.logical_row_end - plan.logical_row_start,
                plan.logical_column_end - plan.logical_column_start,
            )
            for plan in plans
        ]
        self.assertEqual(shapes, [(8, 12), (8, 12)])

    def test_oversized_physical_cell_returns_no_plan_instead_of_resizing(self) -> None:
        """单个物理格超过上限说明网格错误，规划器必须交给上层报错。"""

        plans = plan_grid_tiles(
            Box(0, 0, 5904, 200),
            0,
            (0, 100, 200),
            (0, 4604, 5904),
            max_side=3900,
            repeat_header_rows=0,
            repeat_stub_columns=0,
        )

        self.assertEqual(plans, [])

    def test_irregular_wide_columns_still_force_minimum_split_count(self) -> None:
        """即使某个逻辑列特别宽，整体细长表也不应退回单块。"""

        rows = (0, 10, 70, 85)
        columns = (
            0,
            10,
            60,
            100,
            145,
            190,
            235,
            275,
            320,
            365,
            410,
            465,
            495,
            570,
            600,
            694,
            739,
            784,
            1529,
            2268,
            2308,
            2353,
            2398,
            2528,
            2573,
            2618,
            2663,
            2708,
            2758,
            2768,
        )
        plans = plan_grid_tiles(
            Box(0, 0, 2768, 85),
            0,
            rows,
            columns,
            max_side=3900,
            repeat_header_rows=0,
            repeat_stub_columns=0,
        )
        self.assertGreaterEqual(len(plans), 5)
        self.assertTrue(all(plan.column_count > 1 for plan in plans))

    def test_borderless_table_uses_whitespace_only_after_line_detection_fails(
        self,
    ) -> None:
        image = Image.new("RGB", (600, 420), "white")
        draw = ImageDraw.Draw(image)
        for y in (60, 145, 230, 315):
            for x in (45, 245, 445):
                draw.rectangle((x, y, x + 105, y + 24), fill="black")
        grid = detect_grid_structure(
            image,
            Box(0, 0, 600, 420),
            TableConfig(
                grid_line_min_ratio=0.8,
                whitespace_blank_ratio=0.002,
                whitespace_min_band=8,
            ),
        )
        self.assertTrue(grid.available)
        self.assertEqual(grid.source, "whitespace")
        self.assertGreaterEqual(grid.row_count, 4)
        self.assertGreaterEqual(grid.column_count, 3)


if __name__ == "__main__":
    unittest.main()
