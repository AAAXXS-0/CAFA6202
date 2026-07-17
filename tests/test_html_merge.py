import unittest

from afac_pipeline.common.models import Box, TilePlan
from afac_pipeline.table.步骤009_HTML表格软对齐 import (
    merge_logical_tiles,
    normalize_table_response,
    normalize_table_response_soft,
)


def tile(column_index: int, start: int, end: int, stub: int) -> TilePlan:
    return TilePlan(
        0, 0, column_index, 1, 2, Box(start, 0, end, 200),
        end - start, 200, 1.0, f"tile_{column_index}.png",
        logical_row_start=0, logical_row_end=2,
        logical_column_start=start // 100, logical_column_end=end // 100,
        stub_context_columns=stub, tiling_mode="logical_grid",
    )


class HtmlMergeTest(unittest.TestCase):
    def test_horizontal_context_column_is_removed_by_logical_coordinate(self) -> None:
        plans = [tile(0, 0, 200, 0), tile(1, 200, 400, 1)]
        contents = {
            (0, 0): "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
            (0, 1): "<table><tr><th>A</th><th>C</th><th>D</th></tr><tr><td>1</td><td>3</td><td>4</td></tr></table>",
        }
        merged, report = merge_logical_tiles(contents, plans, 2, 4)
        self.assertEqual(merged.count("<th>A</th>"), 1)
        self.assertIn("<th>D</th>", merged)
        self.assertEqual(report["covered_cells"], 8)

    def test_normalize_preserves_merged_cell(self) -> None:
        html, shape = normalize_table_response(
            '<table><tr><th colspan="2">标题</th></tr><tr><td>A</td><td>1</td></tr></table>'
        )
        self.assertIn('colspan="2"', html)
        self.assertEqual(shape, {"rows": 2, "columns": 2})


    def test_soft_alignment_restores_internal_blank_columns(self) -> None:
        response = (
            "<table><tr><td>A</td><td>B</td><td>C</td></tr></table>"
        )
        html, report = normalize_table_response_soft(
            response,
            1,
            5,
            [[True, False, True, False, True]],
        )
        self.assertEqual(html.count("<td></td>"), 2)
        self.assertTrue(report["warnings"])

    def test_soft_alignment_restores_omitted_blank_row(self) -> None:
        response = (
            "<table>"
            "<tr><td>A</td></tr>"
            "<tr><td>C</td></tr>"
            "</table>"
        )
        html, report = normalize_table_response_soft(
            response,
            3,
            1,
            [[True], [False], [True]],
        )
        self.assertEqual(html.count("<tr>"), 3)

if __name__ == "__main__":
    unittest.main()
