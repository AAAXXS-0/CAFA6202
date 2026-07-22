import unittest

import numpy as np
from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
    LineSegment,
    _merge_shallow_title_regions,
    _sparse_black_grid_centers,
    adaptive_line_segments,
    detect_v6_grid,
)
from afac_pipeline.table.步骤002_低密度分表 import find_density_bands


class V6StructureTest(unittest.TestCase):
    def test_single_pixel_razor_valleys_can_split_three_real_tables(self) -> None:
        """5%图只剩1像素时，极深且两侧有内容的表间低谷仍应落刀。"""

        density = np.full((200, 300), 0.16, dtype=np.float32)
        density[:10] = 0
        density[190:] = 0
        density[55] = 0
        density[125] = 0.001

        horizontal, vertical = find_density_bands(density)

        self.assertEqual(vertical, [])
        self.assertEqual([band.center for band in horizontal], [55, 125])

    def test_shallow_title_strip_is_merged_into_following_table(self) -> None:
        split, analysis = _merge_shallow_title_regions(
            [Box(0, 0, 600, 60), Box(0, 60, 600, 800)],
            [Box(80, 20, 520, 35), Box(50, 80, 550, 700)],
            (600, 800),
            TableConfig(),
        )

        self.assertEqual(split, [Box(0, 0, 600, 800)])
        self.assertEqual(analysis, [Box(50, 20, 550, 700)])

    def test_real_small_table_is_not_mistaken_for_title_strip(self) -> None:
        split, _ = _merge_shallow_title_regions(
            [Box(0, 0, 600, 130), Box(0, 130, 600, 800)],
            [Box(50, 20, 550, 110), Box(50, 150, 550, 700)],
            (600, 800),
            TableConfig(),
        )

        self.assertEqual(len(split), 2)

    def test_official_defaults_are_fixed_scale_v6_values(self) -> None:
        config = TableConfig()
        self.assertEqual(config.table_analysis_scale, 0.20)
        self.assertEqual(config.table_black_line_scale, 0.50)
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
        self.assertEqual(diagnostics.column_source, "white-band-50%-sliding-body")
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

    def test_black_lines_can_use_fifty_percent_image_without_changing_white_image(
        self,
    ) -> None:
        """黑线来自高分辨率图，白带分析图仍可保持原来的低分辨率。"""

        white_analysis = Image.new("RGB", (200, 120), "white")
        black_analysis = Image.new("RGB", (1000, 600), "white")
        draw = ImageDraw.Draw(black_analysis)
        for y in (0, 120, 240, 360, 480, 599):
            draw.line((0, y, 999, y), fill="black", width=3)
        for x in (0, 200, 400, 600, 800, 999):
            draw.line((x, 0, x, 599), fill="black", width=3)

        grid, diagnostics = detect_v6_grid(
            white_analysis,
            Box(0, 0, 2000, 1200),
            TableConfig(grid_black_column_min_contrast=10),
            black_analysis_image=black_analysis,
        )

        self.assertTrue(grid.available, diagnostics.to_dict())
        self.assertTrue(diagnostics.row_source.startswith("black-line"))
        self.assertTrue(diagnostics.column_source.startswith("black-line"))
        self.assertGreaterEqual(len(diagnostics.black_rows), 5)
        self.assertGreaterEqual(len(diagnostics.black_columns), 5)
        self.assertEqual(diagnostics.black_rows_at_whitespace_scale, ())
        self.assertEqual(diagnostics.black_columns_at_whitespace_scale, ())

    def test_partial_header_lines_use_rows_and_high_resolution_column_gaps(
        self,
    ) -> None:
        """局部表头横线不能吞掉密集表的真实行列结构。"""

        def draw_dense_table(
            size: tuple[int, int],
            *,
            horizontal_lines: tuple[int, ...],
            line_width: int,
        ) -> Image.Image:
            image = Image.new("RGB", size, "white")
            draw = ImageDraw.Draw(image)
            width, height = size
            row_step = height / 32
            column_step = width / 13
            for row in range(30):
                y1 = round((row + 1) * row_step)
                y2 = max(y1, round((row + 1.45) * row_step))
                for column in range(12):
                    x1 = round((column + 0.25) * column_step)
                    x2 = max(x1, round((column + 0.72) * column_step))
                    draw.rectangle((x1, y1, x2, y2), fill="black")
            for y in horizontal_lines:
                draw.line((0, y, width - 1, y), fill="black", width=line_width)
            return image

        # 低分辨率图上的3根粗分隔带会堵住纵向白带；高分辨率图保留了
        # 6根可擦除的细横线和真实列缝。这模拟185a超宽密集表。
        white_analysis = draw_dense_table(
            (240, 180),
            horizontal_lines=(28, 82, 136),
            line_width=4,
        )
        black_analysis = draw_dense_table(
            (1200, 900),
            horizontal_lines=(120, 126, 410, 416, 700, 706),
            line_width=3,
        )

        grid, diagnostics = detect_v6_grid(
            white_analysis,
            Box(0, 0, 2400, 1800),
            TableConfig(),
            black_analysis_image=black_analysis,
        )

        self.assertTrue(grid.available, diagnostics.to_dict())
        self.assertEqual(diagnostics.row_source, "white-band")
        self.assertEqual(diagnostics.column_source, "white-band-50%-sliding-body")
        self.assertTrue(diagnostics.white_column_uses_black_scale)
        # 样本实际画了12列；滑窗表体不能再被表头干扰多切成旧结果14列。
        self.assertEqual((grid.row_count, grid.column_count), (31, 12))
        self.assertIn("局部表头/分段线", diagnostics.row_reliability)
        self.assertIn("采用50%统一分析图", diagnostics.white_column_cleanup)

    def test_long_borderless_table_uses_body_rows_for_columns(self) -> None:
        """长表表头会切碎列缝，主体区复核应恢复11列而不是几十列。"""

        image = Image.new("RGB", (560, 860), "white")
        draw = ImageDraw.Draw(image)
        # 表头横跨所有列缝，模拟公司名、单位、险种名称等多行文字。
        for x in range(5, 550, 17):
            draw.rectangle((x, 4, x + 10, 34), fill="black")
        # 70行、11列的主体。单元格文字不碰列缝，行距和列距都很稳定。
        for row in range(70):
            y = 45 + row * 11
            for column in range(11):
                x = 8 + column * 50
                width = 24 + (row + column) % 8
                draw.rectangle((x, y, x + width, y + 5), fill="black")

        grid, diagnostics = detect_v6_grid(
            image,
            Box(0, 0, 5600, 8600),
            TableConfig(),
        )

        self.assertTrue(grid.available, diagnostics.to_dict())
        self.assertEqual(grid.column_count, 11)
        self.assertIn("表体窗口", diagnostics.white_column_cleanup)

    def test_two_column_table_combines_sparse_lines_and_row_gaps(self) -> None:
        """少于5根的真实竖线也应支持两列表，且不能把数据区吞成一行。"""

        image = Image.new("RGB", (100, 326), "white")
        draw = ImageDraw.Draw(image)
        # 左外框和中间分列线；右外框由分析框边界补齐。
        for x in (4, 58):
            draw.line((x, 0, x, 325), fill="black", width=2)
        # 顶线、表头底线和底线，只能定义表头与整段数据，不能代表全部行。
        for y in (4, 36, 321):
            draw.line((0, y, 99, y), fill="black", width=2)
        # 两行表头文字和18行数据，模拟8a150窄表。
        for y in (10, 23):
            draw.rectangle((65, y, 88, y + 5), fill="black")
        for index in range(18):
            y = 43 + index * 16
            draw.rectangle((70, y, 82, y + 6), fill="black")

        grid, diagnostics = detect_v6_grid(
            image,
            Box(0, 0, 500, 1630),
            TableConfig(),
        )

        self.assertTrue(grid.available, diagnostics.to_dict())
        self.assertEqual((grid.row_count, grid.column_count), (19, 2))
        self.assertEqual(
            diagnostics.row_source,
            "hybrid-sparse-lines-white-bands",
        )
        self.assertEqual(
            diagnostics.column_source,
            "sparse-black-lines-0.95-contrast",
        )
        self.assertIn("擦除2根严格竖线", diagnostics.row_reliability)
        self.assertIn("小列数表复核通过", diagnostics.column_reliability)

    def test_sparse_lines_cannot_create_an_uncuttable_giant_column(self) -> None:
        """少量局部竖线不能让超宽表退化成几个无法送模的巨列。"""

        _, accepted, reason = _sparse_black_grid_centers(
            [
                LineSegment(10, 0, 600),
                LineSegment(100, 0, 600),
                LineSegment(850, 0, 600),
            ],
            length=1000,
            minimum_cell_size=18,
            maximum_span_ratio=0.95,
            source_length=12000,
            maximum_source_cell_size=3900,
        )

        self.assertFalse(accepted)
        self.assertIn("超过切片上限3900px", reason)


if __name__ == "__main__":
    unittest.main()
