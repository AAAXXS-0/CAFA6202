import csv
import tempfile
import unittest
from pathlib import Path

from afac_pipeline.common.竞赛数据集 import 解析竞赛数据集


class CompetitionDatasetTest(unittest.TestCase):
    def test_b_is_preferred_and_zone_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_dir = root / "raw_data/finix_huge_long_rest_B/images"
            table_dir = root / "raw_data/finix_huge_table_rest_B/images"
            long_dir.mkdir(parents=True)
            table_dir.mkdir(parents=True)
            (long_dir / "long.jpg").write_bytes(b"x")
            (long_dir / "long.jpg:Zone.Identifier").write_bytes(b"x")
            (table_dir / "table.png").write_bytes(b"x")
            dataset = 解析竞赛数据集(root)
        self.assertEqual(dataset.榜单, "B")
        self.assertEqual(dataset.提交顺序, ("long.jpg", "table.png"))
        self.assertIsNone(dataset.模板路径)

    def test_existing_template_controls_order_but_not_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_dir = root / "raw_data/finix_huge_long_rest_B/images"
            table_dir = root / "raw_data/finix_huge_table_rest_B/images"
            long_dir.mkdir(parents=True)
            table_dir.mkdir(parents=True)
            (long_dir / "long.jpg").write_bytes(b"x")
            (table_dir / "table.png").write_bytes(b"x")
            with (root / "finix_ab_B_submit_mock.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
                writer.writeheader()
                writer.writerow({"file_name": "table.png", "ground_truth": ""})
                writer.writerow({"file_name": "long.jpg", "ground_truth": ""})
            dataset = 解析竞赛数据集(root, "B")
        self.assertEqual(dataset.提交顺序, ("table.png", "long.jpg"))

    def test_mojibake_name_maps_to_official_chinese_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_dir = root / "raw_data/finix_huge_long_rest_B/images"
            table_dir = root / "raw_data/finix_huge_table_rest_B/images"
            long_dir.mkdir(parents=True)
            table_dir.mkdir(parents=True)
            official = "21470_1.0005_招商仁和条款_FXTK2021100900000001504383_page5.png"
            broken = official.encode("utf-8").decode("gb18030")
            (long_dir / broken).write_bytes(b"x")
            (table_dir / "table.jpg").write_bytes(b"x")
            with (root / "finix_ab_B_submit_mock.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
                writer.writeheader()
                writer.writerow({"file_name": official, "ground_truth": ""})
                writer.writerow({"file_name": "table.jpg", "ground_truth": ""})
            dataset = 解析竞赛数据集(root, "B")
        self.assertEqual(dataset.文件名映射[broken], official)


if __name__ == "__main__":
    unittest.main()
