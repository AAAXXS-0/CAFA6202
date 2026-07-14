import unittest

from afac_pipeline.common.models import Box
from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤001_数据定义 import LayoutBlock
from afac_pipeline.long.步骤004_自适应安全切块 import build_adaptive_chunks
from afac_pipeline.long.步骤005_大模型请求打包 import (
    normalize_markdown_heading_levels,
)


class LongAdaptiveCutTest(unittest.TestCase):
    def test_blank_bands_create_non_overlapping_safe_chunks(self) -> None:
        height = 8000
        projection = [0.08] * height
        for start, end in ((3150, 3190), (5650, 5690)):
            projection[start:end] = [0.0] * (end - start)

        config = LongConfig(
            adaptive_target_height=3200,
            adaptive_min_height=2200,
            max_vlm_height=3900,
            safe_cut_search=600,
            minimum_blank_band=8,
            projection_blank_ratio=0.01,
        )
        blocks = [
            LayoutBlock("a", "Text", Box(40, 200, 560, 3000), 0.9, 0),
            LayoutBlock("b", "Text", Box(40, 3300, 560, 5500), 0.9, 1),
            LayoutBlock("c", "Text", Box(40, 5900, 560, 7800), 0.9, 2),
        ]
        chunks, debug = build_adaptive_chunks(
            600,
            height,
            projection,
            blocks,
            config,
        )

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.source_box.height <= 3900 for chunk in chunks))
        self.assertEqual(chunks[0].source_box.y2, chunks[1].source_box.y1)
        self.assertEqual(chunks[1].source_box.y2, chunks[2].source_box.y1)
        self.assertEqual(chunks[0].cut_method, "blank_band")
        self.assertEqual(chunks[1].cut_method, "blank_band")
        self.assertEqual(debug["fallback_overlap_count"], 0)

    def test_dense_area_uses_only_fallback_overlap(self) -> None:
        height = 7000
        config = LongConfig(
            adaptive_target_height=3200,
            adaptive_min_height=2200,
            max_vlm_height=3900,
            vlm_overlap=200,
        )
        projection = [0.12] * height
        blocks = [
            LayoutBlock("dense", "Text", Box(0, 0, 600, height), 0.9, 0)
        ]
        chunks, debug = build_adaptive_chunks(
            600,
            height,
            projection,
            blocks,
            config,
        )

        self.assertGreater(len(chunks), 1)
        self.assertGreater(debug["fallback_overlap_count"], 0)
        self.assertTrue(all(chunk.source_box.height <= 3900 for chunk in chunks))
        for first, second in zip(chunks, chunks[1:]):
            self.assertGreaterEqual(first.source_box.y2, second.source_box.y1)
        self.assertEqual(chunks[0].source_box.y1, 0)
        self.assertEqual(chunks[-1].source_box.y2, height)

    def test_numbered_vlm_headings_are_normalized_without_promoting_body(self) -> None:
        markdown = (
            "### 1 总则\n"
            "# 1.1 投保条件\n"
            "## 1.1.1 责任范围\n"
            "普通正文 2.1 不应成为标题"
        )
        self.assertEqual(
            normalize_markdown_heading_levels(markdown),
            "# 1 总则\n"
            "## 1.1 投保条件\n"
            "### 1.1.1 责任范围\n"
            "普通正文 2.1 不应成为标题",
        )


if __name__ == "__main__":
    unittest.main()
