import tempfile
import unittest
from pathlib import Path

from afac_pipeline.table.步骤011_全流程调度 import (
    _raw_metadata_matches,
    _raw_response_candidates,
)


class RawResponseRecoveryTest(unittest.TestCase):
    def test_attempts_are_sorted_numerically_from_new_to_old(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in (1, 9, 10, 2):
                (root / f"tile_attempt_{number}.md").write_text("x", encoding="utf-8")
            names = [path.name for path in _raw_response_candidates(root, "tile")]
        self.assertEqual(
            names,
            ["tile_attempt_10.md", "tile_attempt_9.md", "tile_attempt_2.md", "tile_attempt_1.md"],
        )

    def test_metadata_requires_all_content_signatures(self) -> None:
        expected = {
            "tile_sha256": "tile",
            "prompt_sha256": "prompt",
            "cache_key": "key",
            "model": "model",
        }
        self.assertTrue(_raw_metadata_matches(dict(expected), expected))
        changed = dict(expected, tile_sha256="other")
        self.assertFalse(_raw_metadata_matches(changed, expected))
        missing = dict(expected)
        missing.pop("prompt_sha256")
        self.assertFalse(_raw_metadata_matches(missing, expected))


if __name__ == "__main__":
    unittest.main()
