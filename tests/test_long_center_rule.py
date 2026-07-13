import unittest

from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.models import LayoutBlock
from afac_pipeline.long.structure import infer_heading_hierarchy, is_centered
from afac_pipeline.common.models import Box


class LongCenterRuleTest(unittest.TestCase):
    def test_wide_body_line_is_not_centered_title(self) -> None:
        self.assertFalse(is_centered(Box(200, 100, 1300, 160), 1500, 0.10))
        self.assertTrue(is_centered(Box(450, 100, 1050, 180), 1500, 0.10))
        self.assertFalse(is_centered(Box(80, 100, 680, 180), 1500, 0.10))

    def test_continuation_document_does_not_invent_h1(self) -> None:
        blocks = [
            LayoutBlock("wide", "Title", Box(200, 100, 1300, 160), 0.9, 0),
            LayoutBlock("text", "Text", Box(100, 220, 1400, 700), 0.9, 0),
            LayoutBlock("next", "Title", Box(80, 800, 500, 860), 0.9, 0),
        ]
        _, headings = infer_heading_hierarchy(blocks, 1500, LongConfig())
        self.assertNotIn("body_h1", {heading.role for heading in headings})
        self.assertEqual(headings[0].level, 2)


if __name__ == "__main__":
    unittest.main()
