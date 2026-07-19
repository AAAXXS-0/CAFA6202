import unittest

from afac_pipeline.common.models import Box
from afac_pipeline.table.步骤007_像素重叠切块 import plan_region_tiles


class TilingTest(unittest.TestCase):
    def test_medium_table_is_cut_at_original_resolution(self) -> None:
        plans = plan_region_tiles(Box(0, 0, 4678, 3308), 0, 3900, 160)
        self.assertGreater(len(plans), 1)
        self.assertTrue(all(plan.scale == 1.0 for plan in plans))

    def test_huge_table_is_split_and_all_tiles_fit(self) -> None:
        plans = plan_region_tiles(Box(100, 200, 10100, 8200), 0, 3900, 160)
        self.assertGreater(len(plans), 1)
        self.assertEqual({plan.row_count for plan in plans}, {3})
        self.assertEqual({plan.column_count for plan in plans}, {3})
        for plan in plans:
            self.assertLessEqual(plan.output_width, 3900)
            self.assertLessEqual(plan.output_height, 3900)
            self.assertGreater(plan.source_box.width, 0)
            self.assertGreater(plan.source_box.height, 0)

    def test_pixel_fallback_does_not_create_nearly_duplicate_tail_tiles(self) -> None:
        plans = plan_region_tiles(Box(0, 0, 10984, 7744), 0, 3900, 160)
        self.assertEqual({plan.row_count for plan in plans}, {2})
        self.assertEqual({plan.column_count for plan in plans}, {3})
        first_column = sorted(
            (plan for plan in plans if plan.column_index == 0),
            key=lambda plan: plan.row_index,
        )
        vertical_overlap = first_column[0].source_box.y2 - first_column[1].source_box.y1
        self.assertLessEqual(vertical_overlap, 160)


if __name__ == "__main__":
    unittest.main()
