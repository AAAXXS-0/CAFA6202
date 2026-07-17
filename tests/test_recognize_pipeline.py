from pathlib import Path
import csv
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤011_全流程调度 import TablePipeline


class FakeClient:
    """不联网的 FinixDoc-VL 替身，用于验证缓存、聚合和 CSV。"""

    model = "fake-finixdoc"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path, prompt: str) -> str:
        self.calls += 1
        self.assertions(image_path, prompt)
        return "| 项目 | 数值 |\n| --- | --- |\n| A | 1 |"

    @staticmethod
    def assertions(image_path: Path, prompt: str) -> None:
        if not image_path.is_file() or "严禁补全" not in prompt:
            raise AssertionError("视觉请求参数不完整")


class RecognizePipelineTest(unittest.TestCase):
    def test_recognition_is_cached_and_duplicates_reuse_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "images"
            input_dir.mkdir()
            image = Image.new("RGB", (800, 600), "white")
            draw = ImageDraw.Draw(image)
            for y in (100, 200, 300, 400):
                draw.line((50, y, 750, y), fill="black", width=3)
            for x in (50, 300, 550, 750):
                draw.line((x, 100, x, 400), fill="black", width=3)
            image.save(input_dir / "a.jpg", format="PNG")
            (input_dir / "b.jpg").write_bytes((input_dir / "a.jpg").read_bytes())

            config = TableConfig(
                backend="pillow",
                detector="projection",
                preview_max_side=800,
                max_vlm_side=700,
                projection_min_line_ratio=0.5,
            )
            pipeline = TablePipeline(config, root / "work")
            manifest = pipeline.prepare_directory(input_dir)
            first_client = FakeClient()
            output_csv = root / "submission.csv"
            results = pipeline.recognize_dataset(manifest, first_client, output_csv)

            self.assertEqual(set(results), {"a.jpg", "b.jpg"})
            self.assertEqual(results["a.jpg"], results["b.jpg"])
            self.assertEqual(first_client.calls, 1)
            with output_csv.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(rows[0]), {"file_name", "ground_truth"})

            second_client = FakeClient()
            pipeline.recognize_dataset(manifest, second_client, output_csv)
            self.assertEqual(second_client.calls, 0)


if __name__ == "__main__":
    unittest.main()
