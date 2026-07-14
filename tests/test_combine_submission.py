import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from afac_pipeline.common.submission import combine_submissions, write_submission


class CombineSubmissionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
