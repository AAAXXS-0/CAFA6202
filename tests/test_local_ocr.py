import unittest

from afac_pipeline.common.local_ocr import OCRBox, OCRLine, plan_ocr_patches
from afac_pipeline.common.models import Box
from afac_pipeline.long.步骤001_数据定义 import Heading
from afac_pipeline.long.步骤005_大模型请求打包 import RecognitionPack
from afac_pipeline.long.步骤007_本地OCR识别 import _pack_markdown
from afac_pipeline.table.local_ocr import LocalTableRecognizer, _matrix_to_html


class LocalOCRTest(unittest.TestCase):
    def test_patch_ownership_covers_each_pixel_exactly_once(self) -> None:
        patches = plan_ocr_patches(3900, 1000, maximum_side=2000, overlap=160)
        self.assertEqual(len(patches), 3)
        for x in range(3900):
            owners = [
                patch
                for patch in patches
                if patch.ownership_x1 <= x < patch.ownership_x2
            ]
            self.assertEqual(len(owners), 1)

    def test_table_context_is_excluded_before_mapping_to_original(self) -> None:
        rows = [100, 200, 300]
        columns = [1000, 1200, 1400]
        tile = {
            "source_box": Box(1200, 200, 1400, 300).to_dict(),
            "header_context_rows": 1,
            "stub_context_columns": 1,
            "output_width": 400,
            "output_height": 200,
        }
        context = OCRBox("重复表头", 1.0, 80, 40, 120, 60)
        body = OCRBox("正文", 1.0, 230, 130, 270, 170)
        self.assertIsNone(
            LocalTableRecognizer._map_body_center(context, tile, rows, columns)
        )
        self.assertEqual(
            LocalTableRecognizer._map_body_center(body, tile, rows, columns),
            (1250.0, 250.0),
        )

    def test_pixel_overlap_tiles_split_ownership_at_seam_center(self) -> None:
        tiles = [
            {
                "row_index": 0,
                "column_index": 0,
                "source_box": Box(0, 0, 60, 100).to_dict(),
            },
            {
                "row_index": 0,
                "column_index": 1,
                "source_box": Box(40, 0, 100, 100).to_dict(),
            },
        ]
        left = LocalTableRecognizer._pixel_tile_ownership(
            tiles[0], tiles, Box(0, 0, 100, 100)
        )
        right = LocalTableRecognizer._pixel_tile_ownership(
            tiles[1], tiles, Box(0, 0, 100, 100)
        )
        self.assertEqual(left, Box(0, 0, 50, 100))
        self.assertEqual(right, Box(50, 0, 100, 100))

    def test_matrix_only_trims_empty_outer_margin(self) -> None:
        cells = [[[] for _ in range(3)] for _ in range(3)]
        cells[1][1].append(OCRBox("金额100元", 0.9, 0, 0, 30, 10))
        html, quality = _matrix_to_html(cells)
        self.assertIn("<td>金额100元</td>", html)
        self.assertEqual(quality["output_rows"], 1)
        self.assertEqual(quality["output_columns"], 1)

    def test_long_title_box_turns_multiline_ocr_into_one_heading(self) -> None:
        pack = RecognitionPack(
            id="request_00000",
            source_box=Box(0, 0, 1000, 500),
            segment_ids=(),
            part_ids=(),
            heading_hints=(),
            file_name="request.png",
            overlap_top=0,
            overlap_bottom=0,
        )
        heading = Heading(
            id="h2_0000",
            level=2,
            role="h2",
            box=Box(100, 100, 500, 180),
            parent_id=None,
            confidence=0.9,
        )
        lines = [
            OCRLine([OCRBox("保险责任", 0.9, 120, 105, 300, 130)]),
            OCRLine([OCRBox("说明", 0.9, 120, 140, 220, 165)]),
        ]
        self.assertEqual(_pack_markdown(lines, pack, [heading]), "## 保险责任说明")


if __name__ == "__main__":
    unittest.main()
