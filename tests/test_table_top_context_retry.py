from pathlib import Path
import tempfile
import unittest

from PIL import Image

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤011_全流程调度 import TablePipeline


class OptionalTitleClient:
    model = "fake-optional-title"

    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.calls = 0

    def recognize(self, image_path: Path, prompt: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("模拟标题识别失败")
        return "表格标题"


class TableTopContextRetryTest(unittest.TestCase):
    def test_failed_optional_title_is_not_cached_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            title_dir = prepared / "top_context"
            title_dir.mkdir(parents=True)
            Image.new("RGB", (100, 20), "black").save(
                title_dir / "region_000.png"
            )
            manifest_path = prepared / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            region = {
                "index": 0,
                "top_context": {
                    "file_name": "top_context/region_000.png",
                    "has_text": True,
                },
            }
            pipeline = TablePipeline(
                TableConfig(backend="pillow"),
                root / "work",
            )

            failed = OptionalTitleClient(fail=True)
            self.assertEqual(
                pipeline._recognize_top_context(
                    manifest_path,
                    region,
                    "source.png",
                    failed,
                    failed.model,
                ),
                "",
            )
            warning = (
                prepared / "quality" / "top_context_region_000_warning.json"
            )
            self.assertTrue(warning.is_file())

            successful = OptionalTitleClient(fail=False)
            self.assertEqual(
                pipeline._recognize_top_context(
                    manifest_path,
                    region,
                    "source.png",
                    successful,
                    successful.model,
                ),
                "表格标题",
            )
            self.assertFalse(warning.exists())

            cached = OptionalTitleClient(fail=True)
            self.assertEqual(
                pipeline._recognize_top_context(
                    manifest_path,
                    region,
                    "source.png",
                    cached,
                    cached.model,
                ),
                "表格标题",
            )
            self.assertEqual(cached.calls, 0)


if __name__ == "__main__":
    unittest.main()
