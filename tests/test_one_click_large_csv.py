from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


class OneClickLargeCsvTest(unittest.TestCase):
    def test_row_count_check_accepts_large_ground_truth(self) -> None:
        """一键脚本应能检查超过 csv 默认 128KiB 的完整图表 HTML。"""

        script_path = Path(__file__).resolve().parents[1] / "一键生成最终CSV.py"
        spec = importlib.util.spec_from_file_location("one_click_csv", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "large.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["file_name", "ground_truth"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "file_name": "large-table.jpg",
                        "ground_truth": "<table>" + "字" * 200_000 + "</table>",
                    }
                )

            module.检查输出行数(csv_path, 1)


if __name__ == "__main__":
    unittest.main()
