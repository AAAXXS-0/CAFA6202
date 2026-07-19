import unittest

import numpy as np

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import adaptive_line_segments


class BlackLineContrastBypassTest(unittest.TestCase):
    def test_default_bypass_ratio_is_absolute_ninety_eight_percent(self) -> None:
        self.assertEqual(
            TableConfig().grid_black_column_contrast_bypass_ratio,
            0.98,
        )

    def test_ninety_eight_percent_column_bypasses_contrast_only_when_requested(
        self,
    ) -> None:
        """几乎连续的竖线不能因邻域灰度差略低于30而被误杀。"""

        ink = np.zeros((100, 100), dtype=bool)
        envelope = np.ones((100, 100), dtype=bool)
        grayscale = np.full((100, 100), 255, dtype=np.uint8)
        ink[:, 50] = True
        grayscale[:, 50] = 200
        # 左右取样为229，和线芯只差29，模拟 aecf 的临界灰底。
        for x in (47, 48, 52, 53):
            grayscale[:, x] = 229

        without_bypass = adaptive_line_segments(
            ink,
            envelope,
            1,
            0.95,
            grayscale=grayscale,
            minimum_contrast=30,
        )
        with_bypass = adaptive_line_segments(
            ink,
            envelope,
            1,
            0.95,
            grayscale=grayscale,
            minimum_contrast=30,
            contrast_bypass_ratio=0.98,
        )

        self.assertEqual(without_bypass, [])
        self.assertEqual([line.position for line in with_bypass], [50])


if __name__ == "__main__":
    unittest.main()
