from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.common.models import Box, TilePlan
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤011_全流程调度 import TablePipeline


class EmptyMarkdownClient:
    """模拟模型在确实发起识别后返回空 Markdown。"""

    model = "fake-empty-markdown"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path, prompt: str) -> str:
        self.calls += 1
        raise RuntimeError("FireRed-OCR 返回了空 Markdown")


class TableEmptyFallbackTest(unittest.TestCase):
    @staticmethod
    def _prepare_case(root: Path, with_text: bool) -> tuple[TablePipeline, Path]:
        prepared = root / "prepared"
        tile_dir = prepared / "tiles"
        tile_dir.mkdir(parents=True)

        image = Image.new("RGB", (60, 40), "white")
        if with_text:
            # 黑块位于第一个单元格内部，确保预处理不会提前把它判成纯空表。
            ImageDraw.Draw(image).rectangle((5, 5, 10, 10), fill="black")
        source_path = prepared / "source.png"
        tile_path = tile_dir / "region_000_r000_c000.png"
        image.save(source_path)
        image.save(tile_path)

        tile = TilePlan(
            0,
            0,
            0,
            1,
            1,
            Box(0, 0, 60, 40),
            60,
            40,
            1.0,
            tile_path.name,
            logical_row_start=0,
            logical_row_end=2,
            logical_column_start=0,
            logical_column_end=3,
            tiling_mode="logical_grid",
        )
        manifest = {
            "image": {"path": str(source_path)},
            "regions": [
                {
                    "index": 0,
                    "row_boundaries": [0, 20, 40],
                    "column_boundaries": [0, 20, 40, 60],
                    "tiles": [tile.to_dict()],
                }
            ],
        }
        manifest_path = prepared / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        return TablePipeline(
            TableConfig(backend="pillow"), root / "work"
        ), manifest_path

    def test_preprocessed_empty_table_skips_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, manifest_path = self._prepare_case(Path(directory), False)
            client = EmptyMarkdownClient()

            html = pipeline._recognize_manifest(manifest_path, client)

            self.assertEqual(client.calls, 0)
            self.assertEqual(html.count("<tr>"), 2)
            self.assertEqual(html.count("<td></td>"), 6)

    def test_model_empty_markdown_uses_preprocessed_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, manifest_path = self._prepare_case(Path(directory), True)
            client = EmptyMarkdownClient()

            html = pipeline._recognize_manifest(manifest_path, client)

            # 首次请求后再复核 3 次，连续全空才采用预处理物理矩阵兜底。
            self.assertEqual(client.calls, 4)
            self.assertEqual(html.count("<tr>"), 2)
            self.assertEqual(html.count("<td></td>"), 6)

            # 已确认的全空兜底带有明确缓存状态，重跑不能再做四次复核。
            cached_html = pipeline._recognize_manifest(manifest_path, client)
            self.assertEqual(client.calls, 4)
            self.assertEqual(cached_html, html)


if __name__ == "__main__":
    unittest.main()
