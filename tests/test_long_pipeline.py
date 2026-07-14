from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤003_滑窗与YOLO检测 import LongLayoutDetector
from afac_pipeline.long.步骤001_数据定义 import DetectionWindow, LayoutBlock
from afac_pipeline.long.步骤006_全流程调度 import LongPipeline, merge_markdown_overlap
from afac_pipeline.common.models import Box


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
            self.assertEqual(image_manifest["schema_version"], 2)
            self.assertGreater(len(image_manifest["safe_chunks"]), 1)
            self.assertGreaterEqual(
                image_manifest["adaptive_cutting"]["fallback_overlap_count"],
                1,
            )
            manifest_dir = Path(dataset["items"][0]["image_manifest"]).parent
            for pack in image_manifest["request_packs"]:
                crop = manifest_dir / "vlm_requests" / pack["file_name"]
                self.assertTrue(crop.is_file())
                self.assertLessEqual(
                    pack["source_box"]["y2"] - pack["source_box"]["y1"],
                    3900,
                )

    def test_markdown_overlap_is_removed_only_at_seam(self) -> None:
        left = "第一段\n共同接缝文字"
        right = "共同接缝文字\n第二段"
        self.assertEqual(merge_markdown_overlap(left, right), "第一段\n共同接缝文字\n第二段")


if __name__ == "__main__":
    unittest.main()
