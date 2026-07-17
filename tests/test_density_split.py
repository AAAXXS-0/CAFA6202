import unittest

import numpy as np

from afac_pipeline.table.density_split import boxes_from_bands, find_density_bands


class DensitySplitTest(unittest.TestCase):
    def test_wide_low_density_gap_with_small_title_splits_two_tables(self) -> None:
        density = np.zeros((200, 300), dtype=np.float32)
        density[20:80, 20:280] = 0.25
        density[120:185, 20:280] = 0.30
        # 表间标题只占很短的高度和宽度，不应抹掉整条低密度分隔带。
        density[98:100, 120:180] = 0.20
        horizontal, vertical = find_density_bands(density)
        boxes = boxes_from_bands(300, 200, horizontal, vertical, density)
        self.assertTrue(horizontal)
        self.assertLess(horizontal[0].center, 98)
        self.assertFalse(vertical)
        self.assertEqual(len(boxes), 2)

    def test_short_but_dense_interrupt_does_not_create_split_band(self) -> None:
        """低密度点较多，但整条区域并不空时，不应把正文误切成多张表。"""

        density = np.full((200, 300), 0.20, dtype=np.float32)
        density[20:185, 20:280] = 0.25
        # 伪造一条由低密度片段和较密正文交错组成的区域。高密度片段都很短，
        # 会被“允许标题穿过”的逻辑合并，但合并后平均密度远高于 2%。
        density[90:110, :] = 0.0
        density[96:98, :] = 0.20
        density[104:106, :] = 0.20
        horizontal, _ = find_density_bands(density)
        self.assertFalse(horizontal)

    def test_three_pixel_periodic_bands_split_seven_stacked_tables(self) -> None:
        """低清图上只有 3 像素宽的真实表间隔也必须保留。"""

        density = np.full((384, 272), 0.25, dtype=np.float32)
        for start in (54, 112, 167, 222, 273, 324):
            density[start : start + 3, :] = 0.012
        horizontal, vertical = find_density_bands(density)
        boxes = boxes_from_bands(272, 384, horizontal, vertical, density)
        self.assertEqual(len(horizontal), 6)
        self.assertFalse(vertical)
        self.assertEqual(len(boxes), 7)

    def test_dense_periodic_cell_gaps_are_not_used_to_split_tables(self) -> None:
        """间距很密的重复空隙属于表内列，不应产生大量窄表块。"""

        density = np.full((384, 272), 0.25, dtype=np.float32)
        for start in range(30, 240, 11):
            density[:, start : start + 3] = 0.0
        horizontal, vertical = find_density_bands(density)
        self.assertFalse(horizontal)
        self.assertFalse(vertical)

    def test_two_real_gaps_keep_three_tables_even_when_widths_differ(self) -> None:
        """两条真实分隔带宽度略有不同，也不能为了匹配先验而强行合并。"""

        density = np.full((384, 272), 0.25, dtype=np.float32)
        density[71:78, :] = 0.0
        density[307:315, :] = 0.0
        horizontal, vertical = find_density_bands(density)
        boxes = boxes_from_bands(272, 384, horizontal, vertical, density)
        self.assertEqual(len(horizontal), 2)
        self.assertFalse(vertical)
        self.assertEqual(len(boxes), 3)


if __name__ == "__main__":
    unittest.main()
