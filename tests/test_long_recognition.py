from pathlib import Path
import csv
import tempfile
import unittest

from PIL import Image

from afac_pipeline.long_config import LongConfig
from afac_pipeline.long_detection import LongLayoutDetector
from afac_pipeline.long_models import DetectionWindow, LayoutBlock
from afac_pipeline.long_pipeline import LongPipeline
from afac_pipeline.models import Box


class OneHeadingDetector(LongLayoutDetector):
    name = "fake-general6-one-heading"

    def detect(
        self,
        window_paths: list[Path],
        windows: list[DetectionWindow],
        image_width: int,
        image_height: int,
    ) -> list[LayoutBlock]:
        return [
            LayoutBlock("h1", "Title", Box(150, 100, 450, 160), 0.95, 0),
            LayoutBlock("body", "Text", Box(40, 220, 560, image_height - 50), 0.95, 0),
        ]


class FakeClient:
    model = "fake-finixdoc-long"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path, prompt: str) -> str:
        self.calls += 1
        if "严禁补全" not in prompt or not image_path.is_file():
            raise AssertionError("长图请求缺少约束或图片")
        if "front_matter" in prompt:
            return "前置信息\n\n# 正文标题\n\n正文内容"
        return "# 正文标题\n\n正文内容"


class LongRecognitionTest(unittest.TestCase):
    def test_long_result_cache_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "images"
            input_dir.mkdir()
            image = Image.new("RGB", (600, 2500), "white")
            image.save(input_dir / "long.jpg", format="PNG")

            pipeline = LongPipeline(
                LongConfig(backend="pillow"),
                root / "work",
                detector=OneHeadingDetector(),
            )
            manifest = pipeline.prepare_directory(input_dir)
            output_csv = root / "long_submission.csv"
            first = FakeClient()
            results = pipeline.recognize_dataset(manifest, first, output_csv)
            self.assertEqual(results["long.jpg"], "前置信息\n\n# 正文标题\n\n正文内容")
            self.assertEqual(first.calls, 1)

            second = FakeClient()
            pipeline.recognize_dataset(manifest, second, output_csv)
            self.assertEqual(second.calls, 0)
            with output_csv.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {"file_name", "ground_truth"})


if __name__ == "__main__":
    unittest.main()
