from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box
from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤001_数据定义 import DetectionWindow, Heading, LayoutBlock
from afac_pipeline.long.步骤004_语义标题分析 import analyze_semantic_headings
from afac_pipeline.long.步骤005_大模型请求打包 import (
    RecognitionPack,
    build_semantic_recognition_packs,
    strip_repeated_context_headings,
)


def block(identifier: str, label: str, box: Box, confidence: float = 0.9):
    return LayoutBlock(identifier, label, box, confidence, 0)


class LongSemanticTest(unittest.TestCase):
    def _analyze(self, title_specs, title_blocks):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        image = Image.new("RGB", (600, 2048), "white")
        draw = ImageDraw.Draw(image)
        # 独立墨迹正文基准为 8px，不使用 Text 框内部投影。
        for y in range(220, 500, 24):
            draw.rectangle((50, y, 550, y + 7), fill="black")
        for box, height in title_specs:
            draw.rectangle(
                (box.x1, box.y1, box.x2, box.y1 + height - 1),
                fill="black",
            )
        path = root / "window.png"
        image.save(path)
        blocks = [block("body", "Text", Box(50, 210, 550, 500)), *title_blocks]
        window = DetectionWindow(0, 0, 2048, 0, 2048, path.name)
        return analyze_semantic_headings(
            blocks,
            600,
            [path],
            [window],
            LongConfig(backend="pillow"),
        )

    def test_independent_ink_and_strict_model_build_style_levels(self) -> None:
        specs = [
            (Box(160, 100, 440, 140), 24),
            (Box(40, 600, 300, 630), 20),
            (Box(70, 650, 340, 675), 12),
            (Box(40, 1200, 300, 1230), 20),
            (Box(70, 1250, 340, 1275), 12),
        ]
        titles = [
            block("h1", "Title", Box(160, 100, 440, 140)),
            block("h2a", "Title", Box(40, 600, 300, 630)),
            block("h3a", "Title", Box(70, 650, 340, 675)),
            block("h2b", "Title", Box(40, 1200, 300, 1230)),
            block("h3b", "Title", Box(70, 1250, 340, 1275)),
        ]
        headings, evidence, debug = self._analyze(specs, titles)
        self.assertEqual([item.level for item in headings], [1, 2, 3, 2, 3])
        self.assertEqual(debug["method"], "strict-general6+independent-full-width-ink+style-clustering-v2")
        self.assertEqual(debug["h2_count"], 2)
        self.assertTrue(debug["selected_h2_style_ids"])
        self.assertTrue(
            all(
                item.model_confidence >= 0.60
                for item in evidence
                if item.source == "model"
            )
        )

    def test_title_below_point_six_does_not_enter_model_candidates(self) -> None:
        ordinary = Box(40, 700, 500, 720)
        _, evidence, debug = self._analyze(
            [(ordinary, 8)],
            [block("low", "Title", ordinary, confidence=0.55)],
        )
        self.assertEqual(debug["strict_model_title_count"], 0)
        self.assertFalse(any(item.source == "model" for item in evidence))
        self.assertEqual(debug["h2_count"], 0)

    def test_strong_ink_can_propose_h2_without_model_box(self) -> None:
        large = Box(40, 700, 300, 730)
        headings, evidence, debug = self._analyze([(large, 20)], [])
        self.assertEqual(debug["strict_model_title_count"], 0)
        self.assertEqual(debug["ink_only_candidate_count"], 1)
        self.assertEqual([item.level for item in headings], [2])
        self.assertEqual(evidence[0].source, "ink")

    def test_no_reliable_h2_requests_real_fallback(self) -> None:
        ordinary = Box(40, 700, 500, 720)
        headings, _, debug = self._analyze(
            [(ordinary, 8)],
            [block("ordinary", "Title", ordinary, confidence=0.90)],
        )
        self.assertFalse(any(item.level == 2 for item in headings))
        self.assertTrue(debug["fallback_required"])

    def test_long_h2_is_split_by_h3_with_context_chain(self) -> None:
        config = LongConfig(
            backend="pillow",
            adaptive_target_height=2500,
            adaptive_min_height=1200,
            max_vlm_height=3900,
        )
        h2 = Heading("h2", 2, "semantic_h2", Box(0, 1000, 600, 1040), None, 0.9)
        h3a = Heading("h3a", 3, "semantic_h3_candidate", Box(20, 2200, 500, 2240), "h2", 0.8)
        h3b = Heading("h3b", 3, "semantic_h3_candidate", Box(20, 6500, 500, 6540), "h2", 0.8)
        projection = [0.08] * 10000
        for start in (3500, 5600, 8200):
            projection[start : start + 30] = [0.0] * 30
        blocks = [
            LayoutBlock("body", "Text", Box(30, 1100, 570, 9900), 0.9, 0)
        ]
        packs, debug = build_semantic_recognition_packs(
            [h2, h3a, h3b], blocks, projection, 600, 10000, config
        )
        self.assertEqual(debug["mode"], "semantic-h2")
        self.assertTrue(any(pack.semantic_role.startswith("h3") for pack in packs))
        self.assertTrue(any(pack.context_heading_ids == ("h2",) for pack in packs))
        self.assertTrue(any(pack.context_heading_ids == ("h2", "h3a") for pack in packs))
        for pack in packs:
            total = pack.source_box.height + sum(box.height for box in pack.context_boxes)
            total += config.semantic_context_gap * len(pack.context_boxes)
            self.assertLessEqual(total, config.max_vlm_height)

    def test_repeated_context_heading_is_removed_by_stable_id(self) -> None:
        seen: dict[str, str] = {}
        first = RecognitionPack(
            "p0", Box(0, 0, 600, 1000), (), (), (), "p0.png",
            visible_heading_ids=("h2",), semantic_role="h2_whole",
        )
        self.assertIn(
            "## 保险责任",
            strip_repeated_context_headings("## 保险责任\n\n正文", first, seen),
        )
        second = RecognitionPack(
            "p1", Box(0, 1000, 600, 2000), (), (), (), "p1.png",
            context_heading_ids=("h2",),
            visible_heading_ids=("h3",),
            semantic_role="h3_whole",
        )
        cleaned = strip_repeated_context_headings(
            "## 保险责任\n\n### 基本责任\n\n内容", second, seen
        )
        self.assertNotIn("## 保险责任", cleaned)
        self.assertIn("### 基本责任", cleaned)


if __name__ == "__main__":
    unittest.main()
