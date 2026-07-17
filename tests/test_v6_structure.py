import unittest

import numpy as np
from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.v6_structure import (
    adaptive_line_segments,
    detect_v6_grid,
)


class V6StructureTest(unittest.TestCase):
    def test_official_defaults_are_fixed_scale_v6_values(self) -> None:
        config = TableConfig()
        self.assertEqual(config.table_analysis_scale, 0.20)
        self.assertEqual(
            config.table_analysis_scale * config.table_density_scale,
            0.05,
        )
        self.assertEqual(config.grid_white_threshold, 225)
        self.assertEqual(config.grid_black_line_ratio, 0.90)
        self.assertEqual(config.grid_reliable_line_count, 5)
        self.assertEqual(config.whitespace_blank_ratio, 0.01)

    def test_black_line_requires_ninety_percent_of_independent_span(self) -> None:
        ink = np.zeros((20, 100), dtype=bool)
        envelope = np.ones_like(ink)
        ink[5, :90] = True
        ink[10, :89] = True

        lines = adaptive_line_segments(ink, envelope, axis=0, minimum_ratio=0.90)

        self.assertEqual([line.position for line in lines], [5])

    def test_each_direction_can_choose_black_line_or_whitespace_separately(self) -> None:
        image = Image.new("RGB", (600, 420), "white")
        draw = ImageDraw.Draw(image)
        # 横向有 6 条完整黑线，应走 90% 黑线；列方向只有文字块，应走白带。
        for y in (20, 95, 170, 245, 320, 395):
            draw.line((20, y, 579, y), fill="black", width=2)
        for y in (45, 120, 195, 270, 345):
            for x in (50, 250, 450):
                draw.rectangle((x, y, x + 80, y + 20), fill="black")

        grid, diagnostics = detect_v6_grid(
            image,
            Box(1000, 2000, 7000, 6200),
            TableConfig(),
        )

        self.assertTrue(grid.available)
        self.assertTrue(diagnostics.row_source.startswith("black-line"))
        self.assertEqual(diagnostics.column_source, "white-band")
        self.assertEqual(grid.row_boundaries[0], 2200)
        self.assertEqual(grid.row_boundaries[-1], 5960)
        self.assertEqual(grid.column_boundaries[0], 1000)
        self.assertEqual(grid.column_boundaries[-1], 7000)


if __name__ == "__main__":
    unittest.main()
