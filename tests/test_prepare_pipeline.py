from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.pipeline import TablePipeline


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
                max_vlm_side=700,
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


if __name__ == "__main__":
    unittest.main()
