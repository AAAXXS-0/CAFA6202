from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.common.local_ocr import OCRBox
from afac_pipeline.common.models import Box, TilePlan
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤004_网格与白带检测 import GridStructure
from afac_pipeline.table.步骤010_本地OCR识别 import LocalTableRecognizer
from afac_pipeline.table.步骤011_全流程调度 import TablePipeline


class TopContextClient:
    model = "fake-top-context"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def recognize(self, image_path: Path, prompt: str) -> str:
        self.calls.append(image_path.name)
        if "top_context" in image_path.parts:
            if "严禁补全" not in prompt:
                raise AssertionError("顶部候选区没有使用独立提示词")
            return "# 表格标题"
        return "<table><tr><td>数据</td></tr></table>"


class TopContextOCR:
    def recognize_path(self, image_path: Path, image_key: str) -> list[OCRBox]:
        if "top_context" in image_path.parts:
            return [OCRBox("表格标题", 0.99, 10, 5, 80, 20)]
        return [OCRBox("数据", 0.99, 10, 5, 50, 20)]


class TableTopContextTest(unittest.TestCase):
    def test_preprocess_crops_analysis_top_before_first_horizontal_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            image = Image.new("RGB", (300, 200), "white")
            ImageDraw.Draw(image).rectangle((80, 30, 160, 45), fill="black")
            image.save(source_path)

            pipeline = TablePipeline(
                TableConfig(backend="pillow"),
                root / "work",
            )
            context = pipeline._save_top_context(
                source_path,
                Box(20, 10, 280, 180),
                GridStructure(
                    "v6:rows=black-line-0.90;columns=black-line-0.95-contrast",
                    (100, 150),
                    (30, 270),
                ),
                0,
                root / "prepared",
            )

            self.assertIsNotNone(context)
            self.assertEqual(context["box"], Box(30, 10, 270, 96).to_dict())
            self.assertEqual(context["image_size"], [240, 86])
            self.assertTrue(context["has_text"])
            self.assertTrue((root / "prepared" / context["file_name"]).is_file())
            recognition_path = root / "prepared" / context["recognition_file_name"]
            self.assertTrue(recognition_path.is_file())
            with Image.open(recognition_path) as recognition:
                self.assertLess(recognition.width, context["image_size"][0])

    @staticmethod
    def _manifest(root: Path) -> Path:
        prepared = root / "prepared"
        tiles = prepared / "tiles"
        top_dir = prepared / "top_context"
        tiles.mkdir(parents=True)
        top_dir.mkdir()

        source = Image.new("RGB", (120, 80), "white")
        ImageDraw.Draw(source).rectangle((10, 50, 50, 65), fill="black")
        source_path = prepared / "source.png"
        source.save(source_path)

        body = source.crop((0, 40, 120, 80))
        tile_path = tiles / "region_000_r000_c000.png"
        body.save(tile_path)
        title = Image.new("RGB", (120, 30), "white")
        ImageDraw.Draw(title).rectangle((10, 5, 80, 20), fill="black")
        title.save(top_dir / "region_000.png")

        tile = TilePlan(
            0,
            0,
            0,
            1,
            1,
            Box(0, 40, 120, 80),
            120,
            40,
            1.0,
            tile_path.name,
            logical_row_start=0,
            logical_row_end=1,
            logical_column_start=0,
            logical_column_end=1,
            tiling_mode="logical_grid",
        )
        manifest = {
            "image": {"path": str(source_path), "file_name": "source.png"},
            "regions": [
                {
                    "index": 0,
                    "box": Box(0, 40, 120, 80).to_dict(),
                    "row_boundaries": [40, 80],
                    "column_boundaries": [0, 120],
                    "top_context": {
                        "file_name": "top_context/region_000.png",
                        "has_text": True,
                    },
                    "tiles": [tile.to_dict()],
                }
            ],
        }
        manifest_path = prepared / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_path

    def test_model_title_is_cached_and_prefixed_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            pipeline = TablePipeline(
                TableConfig(backend="pillow"),
                root / "work",
            )
            first = TopContextClient()
            result = pipeline._recognize_manifest(manifest, first)

            self.assertTrue(result.startswith("# 表格标题\n\n<table>"))
            self.assertEqual(
                sorted(first.calls),
                ["region_000.png", "region_000_r000_c000.png"],
            )
            second = TopContextClient()
            self.assertEqual(
                pipeline._recognize_manifest(manifest, second),
                result,
            )
            self.assertEqual(second.calls, [])

    def test_local_ocr_title_is_prefixed_without_entering_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            recognizer = LocalTableRecognizer(TopContextOCR(), root / "local")
            result = recognizer.recognize_manifest(manifest, "sha256")

            self.assertTrue(result.startswith("表格标题\n\n<table>"))
            self.assertIn("<td>数据</td>", result)


if __name__ == "__main__":
    unittest.main()
