from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.long_config import LongConfig
from afac_pipeline.long_detection import LongLayoutDetector
from afac_pipeline.long_models import DetectionWindow, LayoutBlock
from afac_pipeline.long_pipeline import LongPipeline, merge_markdown_overlap
from afac_pipeline.models import Box


class FakeLongDetector(LongLayoutDetector):
    name = "fake-general6"

    def detect(
        self,
        window_paths: list[Path],
        windows: list[DetectionWindow],
        image_width: int,
        image_height: int,
    ) -> list[LayoutBlock]:
        self.window_count = len(windows)
        return [
            LayoutBlock("toc", "Title", Box(160, 100, 440, 150), 0.9, 0),
            LayoutBlock("toc_text", "Text", Box(50, 200, 550, 850), 0.9, 0),
            LayoutBlock("h1", "Title", Box(150, 1000, 450, 1050), 0.9, 0),
            LayoutBlock("intro", "Text", Box(50, 1100, 550, 1400), 0.9, 0),
            LayoutBlock("h2", "Title", Box(40, 1500, 300, 1540), 0.9, 0),
            LayoutBlock("h3", "Title", Box(60, 1560, 340, 1600), 0.9, 0),
            LayoutBlock("body", "Text", Box(50, 1650, 550, image_height - 100), 0.9, 0),
        ]


class LongPipelineTest(unittest.TestCase):
    def test_prepare_synthetic_long_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "images"
            input_dir.mkdir()
            image = Image.new("RGB", (600, 6000), "white")
            draw = ImageDraw.Draw(image)
            draw.text((100, 100), "toc", fill="black")
            draw.text((100, 1000), "body", fill="black")
            image.save(input_dir / "long.jpg", format="PNG")

            detector = FakeLongDetector()
            config = LongConfig(backend="pillow")
            pipeline = LongPipeline(config, root / "work", detector=detector)
            manifest_path = pipeline.prepare_directory(input_dir)
            dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
            image_manifest = json.loads(
                Path(dataset["items"][0]["image_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(dataset["image_count"], 1)
            self.assertGreater(detector.window_count, 1)
            self.assertEqual(image_manifest["detector"], "fake-general6")
            self.assertGreater(len(image_manifest["segments"]), 1)
            for segment in image_manifest["segments"]:
                for part in segment["parts"]:
                    path = Path(dataset["items"][0]["image_manifest"]).parent
                    crop = path / "semantic_crops" / part["file_name"]
                    self.assertTrue(crop.is_file())
                    self.assertLessEqual(part["source_box"]["y2"] - part["source_box"]["y1"], 3900)

    def test_markdown_overlap_is_removed_only_at_seam(self) -> None:
        left = "第一段\n共同接缝文字"
        right = "共同接缝文字\n第二段"
        self.assertEqual(merge_markdown_overlap(left, right), "第一段\n共同接缝文字\n第二段")


if __name__ == "__main__":
    unittest.main()
