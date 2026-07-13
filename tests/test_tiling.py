import unittest

from afac_pipeline.common.models import Box
from afac_pipeline.table.tiling import plan_region_tiles


class TilingTest(unittest.TestCase):
    def test_medium_table_prefers_single_resized_tile(self) -> None:
        plans = plan_region_tiles(Box(0, 0, 4678, 3308), 0, 3900, 160, 0.65)
        self.assertEqual(len(plans), 1)
        self.assertLessEqual(plans[0].output_width, 3900)
        self.assertLessEqual(plans[0].output_height, 3900)
        self.assertLess(plans[0].scale, 1.0)

    def test_huge_table_is_split_and_all_tiles_fit(self) -> None:
        plans = plan_region_tiles(Box(100, 200, 10100, 8200), 0, 3900, 160, 0.65)
        self.assertGreater(len(plans), 1)
        self.assertEqual({plan.row_count for plan in plans}, {3})
        self.assertEqual({plan.column_count for plan in plans}, {3})
        for plan in plans:
            self.assertLessEqual(plan.output_width, 3900)
            self.assertLessEqual(plan.output_height, 3900)
            self.assertGreater(plan.source_box.width, 0)
            self.assertGreater(plan.source_box.height, 0)


if __name__ == "__main__":
    unittest.main()
