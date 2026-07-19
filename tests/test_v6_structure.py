import unittest

import numpy as np
from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
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
        self.assertEqual(config.grid_black_column_line_ratio, 0.95)
        self.assertEqual(config.grid_black_column_endpoint_trim_ratio, 0.05)
        self.assertEqual(config.grid_black_column_min_contrast, 30.0)
        self.assertEqual(config.grid_reliable_line_count, 5)
        self.assertEqual(config.whitespace_blank_ratio, 0.01)

    def test_black_line_requires_ninety_percent_of_independent_span(self) -> None:
        ink = np.zeros((20, 100), dtype=bool)
        envelope = np.ones_like(ink)
        ink[5, :90] = True
        ink[10, :89] = True

        lines = adaptive_line_segments(ink, envelope, axis=0, minimum_ratio=0.90)

        self.assertEqual([line.position for line in lines], [5])

    def test_strict_column_rule_keeps_grid_line_and_rejects_aligned_ones(self) -> None:
        gray = np.full((100, 40), 255, dtype=np.uint8)
        # 真实表格线颜色深、左右为背景；上下各留 5 像素模拟包络误差。
        gray[5:95, 10] = 170
        # 同列数字“1”覆盖率也很高，但左右同属文字，局部对比明显更小。
        gray[5:95, 20] = 205
        gray[5:95, 17:20] = 218
        gray[5:95, 21:24] = 218
        ink = gray < 225
        envelope = np.ones_like(ink)

        loose = adaptive_line_segments(ink, envelope, axis=1, minimum_ratio=0.90)
        strict = adaptive_line_segments(
            ink,
            envelope,
            axis=1,
            minimum_ratio=0.95,
            grayscale=gray,
            endpoint_trim_ratio=0.05,
            minimum_contrast=30.0,
        )

        self.assertIn(20, [line.position for line in loose])
        self.assertEqual([line.position for line in strict], [10])

    def test_each_direction_can_choose_black_line_or_whitespace_separately(
        self,
    ) -> None:
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

        self.assertTrue(grid.available, diagnostics.to_dict())
        self.assertTrue(diagnostics.row_source.startswith("black-line"))
        self.assertEqual(diagnostics.column_source, "white-band")
        self.assertEqual(grid.row_boundaries[0], 2200)
        self.assertEqual(grid.row_boundaries[-1], 5960)
        self.assertEqual(grid.column_boundaries[0], 1000)
        self.assertEqual(grid.column_boundaries[-1], 7000)

    def test_edge_only_black_lines_are_not_accepted_as_black_grid(self) -> None:
        """两侧边缘线再多，也不能伪装成中间存在物理列的正常表格。"""

        image = Image.new("RGB", (600, 420), "white")
        draw = ImageDraw.Draw(image)
        for y in (20, 95, 170, 245, 320, 395):
            draw.line((0, y, 599, y), fill="black", width=2)
        for x in (3, 7, 11, 588, 592, 596):
            draw.line((x, 0, x, 419), fill="black", width=1)
        grid, diagnostics = detect_v6_grid(
            image,
            Box(0, 0, 6000, 4200),
            TableConfig(grid_black_column_min_contrast=0),
        )
        self.assertEqual(diagnostics.column_source, "white-band")
        self.assertIn("中部没有物理边界", diagnostics.column_reliability)


if __name__ == "__main__":
    unittest.main()
