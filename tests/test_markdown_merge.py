import unittest

from afac_pipeline.table.步骤008_Markdown表格合并 import merge_horizontal, merge_vertical, parse_first_table


class MarkdownMergeTest(unittest.TestCase):
    def test_vertical_overlap_row_is_removed(self) -> None:
        top = """# 表题

| 编号 | 数值 |
| --- | --- |
| 1 | 10 |
| 2 | 20 |"""
        bottom = """| 编号 | 数值 |
| --- | --- |
| 2 | 20 |
| 3 | 30 |"""
        merged = parse_first_table(merge_vertical(top, bottom))
        self.assertEqual(merged.rows, [["1", "10"], ["2", "20"], ["3", "30"]])

    def test_horizontal_overlap_column_is_removed(self) -> None:
        left = """| 编号 | 项目A |
| --- | --- |
| 1 | x |
| 2 | z |"""
        right = """| 项目A | 项目B |
| --- | --- |
| x | y |
| z | w |"""
        merged = parse_first_table(merge_horizontal(left, right))
        self.assertEqual(merged.header, ["编号", "项目A", "项目B"])
        self.assertEqual(merged.rows, [["1", "x", "y"], ["2", "z", "w"]])


if __name__ == "__main__":
    unittest.main()
