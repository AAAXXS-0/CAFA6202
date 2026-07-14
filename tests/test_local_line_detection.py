import unittest

import numpy as np

from afac_pipeline.table.tools.experiment_density_split_and_boundaries import (
    adaptive_line_segments,
)


class LocalLineDetectionTest(unittest.TestCase):
    def test_trapezoid_lines_use_their_own_active_span(self) -> None:
        """梯形短线相对自身是完整黑线，不应被外接矩形的白底稀释。"""

        ink = np.zeros((120, 220), dtype=bool)
        expected = 0
        for y in range(12, 108, 10):
            # 越靠下线越短，模拟梯形表格的斜边。
            end = 205 - y
            ink[y, 15:end] = True
            expected += 1
        segments = adaptive_line_segments(
            ink,
            axis=0,
            minimum_ratio=0.90,
            minimum_span_ratio=0.20,
        )
        self.assertEqual(len(segments), expected)
        self.assertTrue(all(segment.start == 15 for segment in segments))


if __name__ == "__main__":
    unittest.main()
