from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤011_全流程调度 import TablePipeline


class PreparePipelineTest(unittest.TestCase):
    def test_prepare_reuses_exact_duplicate(self) -> None:
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
                max_vlm_side=800,
                projection_min_line_ratio=0.5,
            )
            pipeline = TablePipeline(config, root / "work")
            manifest_path = pipeline.prepare_directory(input_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["image_count"], 2)
            self.assertEqual(manifest["unique_image_count"], 1)
            self.assertEqual(manifest["duplicate_reuse_count"], 1)
            image_manifest = Path(manifest["items"][0]["image_manifest"])
            self.assertTrue(image_manifest.is_file())
            prepared = json.loads(image_manifest.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(prepared["regions"]), 1)

    def test_prepare_large_grid_keeps_aspect_without_default_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "images"
            input_dir.mkdir()
            image = Image.new("RGB", (1000, 1000), "white")
            draw = ImageDraw.Draw(image)
            for value in range(0, 1001, 200):
                coordinate = min(value, 999)
                draw.line((0, coordinate, 999, coordinate), fill="black", width=3)
                draw.line((coordinate, 0, coordinate, 999), fill="black", width=3)
            image.save(input_dir / "grid.png")

            config = TableConfig(
                backend="pillow",
                detector="projection",
                preview_max_side=1000,
                max_vlm_side=512,
                projection_min_line_ratio=0.5,
                grid_analysis_max_side=1000,
                grid_line_min_ratio=0.8,
            )
            pipeline = TablePipeline(config, root / "work")
            dataset = json.loads(
                pipeline.prepare_directory(input_dir).read_text(encoding="utf-8")
            )
            prepared_path = Path(dataset["items"][0]["image_manifest"])
            prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
            region = prepared["regions"][0]
            self.assertEqual(region["grid_source"], "ruled-lines")
            self.assertTrue(
                all(not tile["header_context_rows"] for tile in region["tiles"])
            )
            self.assertTrue(
                all(not tile["stub_context_columns"] for tile in region["tiles"])
            )
            for tile in region["tiles"]:
                tile_path = prepared_path.parent / "tiles" / tile["file_name"]
                with Image.open(tile_path) as output:
                    expected = (tile["output_width"], tile["output_height"])
                    self.assertEqual(output.size, expected)
                    self.assertLessEqual(max(output.size), 512)
                    self.assertEqual(tile["scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
