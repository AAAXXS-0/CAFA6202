import unittest

import numpy as np

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
    select_adaptive_white_column_bands,
)


def ink_with_white_bands(
    width: int,
    items: list[tuple[int, int]],
) -> np.ndarray:
    """生成只有指定纵向白带的墨迹图，便于精确测试宽度选择。"""

    ink = np.ones((100, width), dtype=bool)
    for center, band_width in items:
        start = center - (band_width - 1) // 2
        ink[:, start : start + band_width] = False
    return ink


class WhiteColumnBandCleanupTest(unittest.TestCase):
    def test_0f_header_and_digit_gaps_select_seven_pixels(self) -> None:
        """0f第二个表的细白线森林应被清掉，10像素真白带应保留。"""

        false_bands = [
            (2, 6), (51, 1), (55, 5), (61, 1), (70, 2), (72, 2),
            (81, 1), (83, 1), (86, 2), (90, 1), (96, 6), (100, 1),
            (104, 1), (110, 5), (116, 1), (118, 2), (121, 1),
            (127, 5), (133, 1), (141, 3), (146, 1), (149, 1),
        ]
        real_bands = [
            (170, 10), (196, 10), (222, 10), (248, 10), (272, 10),
            (298, 10), (324, 10), (350, 10), (381, 21),
        ]
        ink = ink_with_white_bands(403, [*false_bands, *real_bands, (400, 6)])

        raw, used, rejected, minimum, before, after, message = (
            select_adaptive_white_column_bands(ink, TableConfig())
        )

        self.assertEqual(minimum, 7)
        self.assertGreater(len(raw), len(used))
        self.assertEqual(
            [band.position for band in used[-len(real_bands) :]],
            [x for x, _ in real_bands],
        )
        self.assertEqual(len(rejected), len(raw) - len(used))
        self.assertGreater(after, before)
        self.assertIn("1→7px", message)

    def test_regular_narrow_columns_keep_one_pixel(self) -> None:
        """3bfd式密集真列即使只有1～2像素宽，也不能被统一加粗删除。"""

        items = [
            (2 + index * 8, 1 if index % 7 == 0 else 2)
            for index in range(100)
        ]
        ink = ink_with_white_bands(810, items)

        raw, used, rejected, minimum, before, after, _ = (
            select_adaptive_white_column_bands(ink, TableConfig())
        )

        self.assertEqual(minimum, 1)
        self.assertEqual(used, raw)
        self.assertEqual(rejected, [])
        self.assertGreaterEqual(before, 0.90)
        self.assertEqual(after, before)

    def test_rows_still_use_one_pixel_base_width(self) -> None:
        config = TableConfig()
        self.assertEqual(config.whitespace_min_band, 1)
        self.assertEqual(config.whitespace_column_max_min_band, 7)


if __name__ == "__main__":
    unittest.main()
