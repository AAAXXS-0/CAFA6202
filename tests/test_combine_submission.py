import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from afac_pipeline.common.submission import (
    combine_submissions,
    combine_submissions_in_order,
    write_submission,
)


class CombineSubmissionTest(unittest.TestCase):
    def test_dataset_order_works_without_mock_template(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            long_csv = root / "long.csv"
            table_csv = root / "table.csv"
            write_submission({"long.jpg": "long"}, long_csv)
            write_submission({"table.jpg": "table"}, table_csv)
            output = combine_submissions_in_order(
                [long_csv, table_csv],
                ["table.jpg", "long.jpg"],
                root / "final.csv",
            )
            with output.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([row["file_name"] for row in rows], ["table.jpg", "long.jpg"])

    def test_two_branches_are_written_in_template_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.csv"
            write_submission({"b.jpg": "mock-b", "a.jpg": "mock-a"}, template, ["b.jpg", "a.jpg"])
            long_csv = root / "long.csv"
            table_csv = root / "table.csv"
            write_submission({"a.jpg": "long"}, long_csv)
            write_submission({"b.jpg": "table"}, table_csv)

            output = combine_submissions([long_csv, table_csv], template, root / "final.csv")
            with output.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual([row["file_name"] for row in rows], ["b.jpg", "a.jpg"])
        self.assertEqual([row["ground_truth"] for row in rows], ["table", "long"])

    def test_missing_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.csv"
            partial = root / "partial.csv"
            write_submission({"a.jpg": "", "b.jpg": ""}, template)
            write_submission({"a.jpg": "result"}, partial)

            with self.assertRaisesRegex(ValueError, "缺少"):
                combine_submissions([partial], template, root / "final.csv")

    def test_large_html_field_can_be_combined(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.csv"
            branch = root / "table.csv"
            large_html = "<table>" + "<td>1234567890</td>" * 10000 + "</table>"
            write_submission({"table.jpg": ""}, template)
            write_submission({"table.jpg": large_html}, branch)

            output = combine_submissions([branch], template, root / "final.csv")
            with output.open("r", encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))

        self.assertEqual(row["ground_truth"], large_html)


if __name__ == "__main__":
    unittest.main()
