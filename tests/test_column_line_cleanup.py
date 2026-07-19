import unittest

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
    LineSegment,
    clean_suspicious_column_lines,
)


def make_lines(positions: list[int]) -> list[LineSegment]:
    return [LineSegment(position, 0, 100) for position in positions]


class ColumnLineCleanupTest(unittest.TestCase):
    def test_aecf_aligned_chinese_strokes_are_removed(self) -> None:
        """局部压字伪线不能把左侧真实大格拆成三个极窄格。"""

        positions = [23, 33, 37, 46, 64, 84, *range(111, 2812, 27)]
        kept, rejected, message = clean_suspicious_column_lines(
            make_lines(positions), TableConfig()
        )

        self.assertEqual([item.line.position for item in rejected], [33, 37])
        self.assertEqual(
            [line.position for line in kept[:8]],
            [23, 46, 64, 84, 111, 138, 165, 192],
        )
        self.assertIn("删除2根", message)
        payload = rejected[0].to_dict()
        self.assertEqual(payload["left_gap"], 10)
        self.assertEqual(payload["right_gap"], 4)
        self.assertEqual(payload["minimum_gap"], 16)

    def test_one_narrow_gap_alone_is_not_enough_to_delete_a_line(self) -> None:
        """普通列宽变化只形成一个窄间距时应保持原网格。"""

        positions = [20, 30, *range(57, 409, 27)]
        kept, rejected, _ = clean_suspicious_column_lines(
            make_lines(positions), TableConfig()
        )

        self.assertEqual([line.position for line in kept], positions)
        self.assertEqual(rejected, [])

    def test_repeated_narrow_spacing_is_treated_as_real_dense_columns(self) -> None:
        """窄列若在全表反复出现，就不是孤立的中文笔画误检。"""

        positions = [10]
        for gap in [5, 20] * 12:
            positions.append(positions[-1] + gap)
        kept, rejected, message = clean_suspicious_column_lines(
            make_lines(positions), TableConfig()
        )

        self.assertEqual([line.position for line in kept], positions)
        self.assertEqual(rejected, [])
        self.assertIn("真实密集列", message)


if __name__ == "__main__":
    unittest.main()
