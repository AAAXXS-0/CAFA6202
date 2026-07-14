import unittest

from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤001_数据定义 import LayoutBlock
from afac_pipeline.long.步骤004_标题层级与二次分块 import (
    attach_physical_parts,
    build_semantic_segments,
    infer_heading_hierarchy,
    merge_multiline_titles,
)
from afac_pipeline.common.models import Box


def block(identifier: str, label: str, y1: int, y2: int, x1: int = 80, x2: int = 520):
    return LayoutBlock(identifier, label, Box(x1, y1, x2, y2), 0.9, 0)


class LongStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LongConfig(center_tolerance_ratio=0.12)

    def test_multiline_centered_title_is_merged(self) -> None:
        blocks = [
            block("a", "Title", 100, 140, 150, 450),
            block("b", "Title", 146, 186, 160, 440),
        ]
        titles = merge_multiline_titles(blocks, 600, self.config)
        self.assertEqual(len(titles), 1)
        self.assertEqual(titles[0].member_ids, ("a", "b"))

    def test_user_title_rules_build_hierarchy_and_segments(self) -> None:
        blocks = [
            block("toc", "Title", 100, 145, 180, 420),
            block("toc_text", "Text", 200, 850),
            block("h1", "Title", 1000, 1050, 170, 430),
            block("intro", "Text", 1100, 1400),
            block("h2a", "Title", 1500, 1540, 40, 300),
            block("h3a", "Title", 1560, 1600, 60, 340),
            block("text_a", "Text", 1650, 1850),
            block("h3b", "Title", 1900, 1940, 60, 340),
            block("text_b", "Text", 1980, 2200),
            block("h2b", "Title", 2300, 2340, 40, 300),
            block("h3c", "Title", 2360, 2400, 60, 340),
            block("text_c", "Text", 2450, 2800),
        ]
        _, headings = infer_heading_hierarchy(blocks, 600, self.config)
        roles = [(heading.role, heading.level, heading.parent_id) for heading in headings]
        self.assertEqual(roles[0][:2], ("toc_title", 1))
        self.assertEqual(roles[1][:2], ("body_h1", 1))
        self.assertEqual([heading.level for heading in headings[2:]], [2, 3, 3, 2, 3])
        self.assertEqual(headings[3].parent_id, headings[2].id)
        self.assertEqual(headings[4].parent_id, headings[2].id)

        segments = build_semantic_segments(3000, blocks, headings)
        by_id = {segment.id: segment for segment in segments}
        self.assertEqual((by_id["toc"].start_y, by_id["toc"].end_y), (100, 1000))
        self.assertEqual(by_id["h3_0000_body"].start_y, 1500)
        self.assertEqual(by_id["h3_0000_body"].expected_heading_levels, (2, 3))
        self.assertEqual(by_id["h3_0001_body"].start_y, 1900)

    def test_oversized_semantic_segment_is_physically_split(self) -> None:
        blocks = [block("t1", "Text", 100, 1000), block("t2", "Text", 4200, 5000)]
        from afac_pipeline.long.步骤001_数据定义 import SemanticSegment

        segments = [SemanticSegment("body", "body", 0, 9000)]
        completed = attach_physical_parts(segments, blocks, 600, self.config)
        self.assertGreater(len(completed[0].parts), 1)
        self.assertTrue(
            all(part.source_box.height <= self.config.max_vlm_height for part in completed[0].parts)
        )
        for first, second in zip(completed[0].parts, completed[0].parts[1:]):
            self.assertGreater(first.source_box.y2, second.source_box.y1)


if __name__ == "__main__":
    unittest.main()
