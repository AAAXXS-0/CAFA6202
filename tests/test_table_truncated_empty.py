import unittest

from afac_pipeline.table.步骤011_全流程调度 import _is_truncated_empty_html


class TruncatedEmptyHtmlTest(unittest.TestCase):
    def test_only_empty_cells_and_partial_final_tag_are_accepted(self) -> None:
        response = "<table><tr><td></td><td></td></tr><tr><td></td><td"
        self.assertTrue(_is_truncated_empty_html(response))

    def test_unclosed_markdown_fence_around_empty_html_is_accepted(self) -> None:
        response = "```markdown\n<table><tr><td></td></tr><tr><td"
        self.assertTrue(_is_truncated_empty_html(response))

    def test_any_visible_text_keeps_strict_failure(self) -> None:
        response = "<table><tr><td>文字</td></tr><tr><td"
        self.assertFalse(_is_truncated_empty_html(response))

    def test_complete_empty_table_uses_normal_empty_table_path(self) -> None:
        response = "<table><tr><td></td></tr></table>"
        self.assertFalse(_is_truncated_empty_html(response))


if __name__ == "__main__":
    unittest.main()
