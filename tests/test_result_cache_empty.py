from pathlib import Path
import tempfile
import unittest

from afac_pipeline.common.cache import ResultCache


class ResultCacheEmptyMarkdownTest(unittest.TestCase):
    def test_empty_tile_result_is_distinct_from_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(Path(directory) / "cache.sqlite3")
            self.assertIsNone(cache.get_tile("blank-tile"))

            cache.put_tile("blank-tile", "", {"skipped_blank": True})

            self.assertEqual(cache.get_tile("blank-tile"), "")


if __name__ == "__main__":
    unittest.main()
