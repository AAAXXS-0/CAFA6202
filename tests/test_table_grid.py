import unittest

from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.grid import detect_grid_structure
from afac_pipeline.table.grid_tiling import plan_grid_tiles


class TableGridTest(unittest.TestCase):
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
            Box(0, 0, 6000, 6000), 0, boundaries, boundaries,
            max_side=3000, single_tile_min_scale=0.65,
            repeat_header_rows=1, repeat_stub_columns=1,
        )
        self.assertGreater(len(plans), 1)
        self.assertTrue(any(plan.header_context_rows == 1 for plan in plans))
        self.assertTrue(any(plan.stub_context_columns == 1 for plan in plans))
        self.assertTrue(all(plan.output_width <= 3000 for plan in plans))
        self.assertTrue(all(plan.output_height <= 3000 for plan in plans))


if __name__ == "__main__":
    unittest.main()
