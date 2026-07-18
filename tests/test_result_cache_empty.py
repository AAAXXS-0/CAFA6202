from pathlib import Path
import tempfile
import unittest

from afac_pipeline.common.cache import ResultCache, merge_result_caches


class ResultCacheEmptyMarkdownTest(unittest.TestCase):
    def test_empty_tile_result_is_distinct_from_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(Path(directory) / "cache.sqlite3")
            self.assertIsNone(cache.get_tile("blank-tile"))

            cache.put_tile("blank-tile", "", {"skipped_blank": True})

            self.assertEqual(cache.get_tile("blank-tile"), "")

    def test_merge_old_cache_keeps_first_success_for_same_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ResultCache(root / "first.sqlite3")
            second = ResultCache(root / "second.sqlite3")
            first.put_tile("shared", "较新结果", {"source": "first"})
            second.put_tile("shared", "较旧结果", {"source": "second"})
            second.put_tile("only-second", "独有结果", {})

            inserted = merge_result_caches(
                root / "merged.sqlite3",
                [first.path, second.path],
            )
            merged = ResultCache(root / "merged.sqlite3")

            self.assertEqual(inserted["tile_results"], 2)
            self.assertEqual(merged.get_tile("shared"), "较新结果")
            self.assertEqual(merged.get_tile("only-second"), "独有结果")


if __name__ == "__main__":
    unittest.main()
