import unittest

from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤003_滑窗与YOLO检测 import deduplicate_layout_blocks, plan_detection_windows
from afac_pipeline.long.步骤001_数据定义 import LayoutBlock
from afac_pipeline.common.models import Box


class LongDetectionTest(unittest.TestCase):
    def test_window_plan_keeps_requested_overlap_and_covers_tail(self) -> None:
        config = LongConfig(window_height=2048, window_step=1792)
        windows = plan_detection_windows(5000, config)
        self.assertEqual([window.start_y for window in windows], [0, 1792, 2952])
        self.assertEqual(windows[-1].end_y, 5000)
        self.assertEqual(windows[0].ownership_end_y, 1920)
        for first, second in zip(windows, windows[1:]):
            self.assertLess(second.start_y, first.end_y)
            self.assertEqual(first.ownership_end_y, second.ownership_start_y)

    def test_global_duplicate_boxes_keep_higher_confidence(self) -> None:
        blocks = [
            LayoutBlock("a", "Title", Box(10, 100, 500, 160), 0.70, 0),
            LayoutBlock("b", "Title", Box(12, 102, 502, 162), 0.92, 1),
            LayoutBlock("c", "Text", Box(10, 100, 500, 160), 0.95, 1),
        ]
        result = deduplicate_layout_blocks(blocks)
        self.assertEqual({block.id for block in result}, {"b", "c"})


if __name__ == "__main__":
    unittest.main()
