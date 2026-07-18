from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.步骤011_全流程调度 import TablePipeline
from afac_pipeline.common.models import Box, TilePlan


class TileFailureClient:
    model = "fake-tile-failure-client"

    def __init__(self, fail_first: bool) -> None:
        self.fail_first = fail_first
        self.calls: list[str] = []

    def recognize(self, image_path: Path, prompt: str) -> str:
        self.calls.append(image_path.name)
        if self.fail_first and image_path.name.endswith("c000.png"):
            raise RuntimeError("请求过载")
        return (
            "<table><tr><td>A</td><td>B</td><td>C</td></tr>"
            "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
        )


class TableTileFailureTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        prepared = root / "prepared"
        tiles_dir = prepared / "tiles"
        tiles_dir.mkdir(parents=True)
        source_path = prepared / "source.png"
        Image.new("RGB", (120, 40), "black").save(source_path)

        tiles = []
        for column_index, start in enumerate((0, 60)):
            path = tiles_dir / f"region_000_r000_c{column_index:03d}.png"
            Image.new("RGB", (60, 40), 240 + column_index).save(path)
            tiles.append(
                TilePlan(
                    0, 0, column_index, 1, 2,
                    Box(start, 0, start + 60, 40),
                    60, 40, 1.0, path.name,
                    logical_column_start=column_index * 3,
                    logical_column_end=(column_index + 1) * 3,
                    logical_row_start=0,
                    logical_row_end=2,
                    tiling_mode="logical_grid",
                ).to_dict()
            )
        manifest_path = prepared / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "image": {"path": str(source_path)},
                    "regions": [{
                        "index": 0,
                        "row_boundaries": [0, 20, 40],
                        "column_boundaries": [0, 20, 40, 60, 80, 100, 120],
                        "tiles": tiles,
                    }],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return manifest_path

    def test_failed_tile_does_not_stop_following_tile_and_retries_only_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = TablePipeline(
                TableConfig(backend="pillow"),
                root / "work",
            )
            manifest = self._manifest(root)
            first = TileFailureClient(fail_first=True)
            with self.assertRaisesRegex(RuntimeError, "1 个切块失败"):
                pipeline._recognize_manifest(manifest, first)
            self.assertEqual(
                first.calls,
                [
                    "region_000_r000_c000.png",
                    "region_000_r000_c001.png",
                ],
            )
            report = json.loads(
                (manifest.parent / "recognition_failures.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["failure_count"], 1)
            self.assertEqual(
                report["failed_parts"][0]["tile_file_name"],
                "region_000_r000_c000.png",
            )

            second = TileFailureClient(fail_first=False)
            result = pipeline._recognize_manifest(manifest, second)
            self.assertEqual(second.calls, ["region_000_r000_c000.png"])
            self.assertIn("<table>", result)


if __name__ == "__main__":
    unittest.main()
