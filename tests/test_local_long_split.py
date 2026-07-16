from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image, ImageChops, ImageDraw

from afac_pipeline.common.local_long_split import (
    create_local_long_parts,
    leading_header_height,
    merge_local_part_markdowns,
    needs_local_long_split,
)
from afac_pipeline.common.models import Box


class LocalLongSplitTest(unittest.TestCase):
    def test_manifest_heading_defines_repeated_toc_header(self) -> None:
        pack = SimpleNamespace(
            context_boxes=(),
            context_heading_ids=(),
            visible_heading_ids=("toc",),
            source_box=Box(0, 100, 600, 4700),
            body_scale=0.8,
        )
        manifest = {
            "config": {"semantic_title_padding": 10},
            "semantic_headings": [
                {"id": "toc", "box": {"x1": 0, "y1": 100, "x2": 600, "y2": 160}}
            ],
        }

        height, ids = leading_header_height(pack, manifest, 10, 3680)

        self.assertEqual(height, 56)
        self.assertEqual(ids, ("toc",))

    def test_existing_h2_h3_context_height_is_exact(self) -> None:
        pack = SimpleNamespace(
            context_boxes=(Box(0, 10, 600, 50), Box(0, 100, 600, 130)),
            context_heading_ids=("h2", "h3"),
            visible_heading_ids=(),
            source_box=Box(0, 500, 600, 3000),
            body_scale=1.0,
        )

        height, ids = leading_header_height(pack, {}, 10, 3000)

        self.assertEqual(height, 90)
        self.assertEqual(ids, ("h2", "h3"))

    def test_blank_band_split_copies_header_to_every_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "toc.png"
            image = Image.new("RGB", (600, 3900), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((180, 15, 420, 55), fill="black")
            for y in range(100, 3850, 35):
                draw.rectangle((40, y, 560, y + 8), fill="black")
            # 三处较宽空白给切割器选择，确保不会压到文字。
            image.save(image_path)

            parts = create_local_long_parts(
                image_path,
                root / "split",
                header_height=80,
                target_height=1500,
                maximum_height=1800,
                minimum_content_height=700,
                search_radius=260,
                fallback_overlap=128,
                sample_width=512,
                white_threshold=225,
                blank_ratio=0.002,
                minimum_blank_height=3,
            )

            self.assertGreaterEqual(len(parts), 3)
            with Image.open(image_path) as source:
                expected_header = source.crop((0, 0, source.width, 80)).convert("RGB")
            for part in parts:
                self.assertLessEqual(part.output_height, 1800)
                with Image.open(root / "split/parts" / part.file_name) as split_image:
                    actual_header = split_image.crop((0, 0, split_image.width, 80)).convert("RGB")
                self.assertIsNone(ImageChops.difference(expected_header, actual_header).getbbox())

            markdowns = [
                f"# 目录\n\n第 {index + 1} 段"
                for index in range(len(parts))
            ]
            merged = merge_local_part_markdowns(
                markdowns,
                parts,
                repeated_heading_count=1,
            )
            self.assertEqual(merged.count("# 目录"), 1)
            for index in range(len(parts)):
                self.assertIn(f"第 {index + 1} 段", merged)

    def test_split_trigger_uses_estimated_model_width(self) -> None:
        self.assertTrue(
            needs_local_long_split(
                1088,
                3900,
                300_000,
                trigger_height=3000,
                minimum_estimated_width=512,
            )
        )
        self.assertFalse(
            needs_local_long_split(
                1088,
                1800,
                300_000,
                trigger_height=3000,
                minimum_estimated_width=512,
            )
        )


if __name__ == "__main__":
    unittest.main()
