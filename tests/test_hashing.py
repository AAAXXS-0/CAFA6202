from pathlib import Path
import tempfile
import unittest

from afac_pipeline.common.hashing import group_exact_duplicates, sha256_file


class HashingTest(unittest.TestCase):
    def test_exact_duplicate_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.jpg"
            second = root / "b.jpg"
            different = root / "c.jpg"
            first.write_bytes(b"same-image-bytes")
            second.write_bytes(b"same-image-bytes")
            different.write_bytes(b"different-image-bytes")

            groups = group_exact_duplicates([first, second, different])
            self.assertEqual(len(groups), 2)
            self.assertEqual({path.name for path in groups[sha256_file(first)]}, {"a.jpg", "b.jpg"})


if __name__ == "__main__":
    unittest.main()
