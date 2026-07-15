from pathlib import Path
from threading import Lock
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤006_全流程调度 import LongPipeline
from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.pipeline import TablePipeline


class FakeClient:
    model = "fake-parallel-client"


class ActivityTracker:
    """记录同时进入识别函数的任务数，用于证明不是串行循环。"""

    def __init__(self) -> None:
        self.lock = Lock()
        self.calls = 0
        self.active = 0
        self.maximum_active = 0

    def __call__(self, manifest_path: Path, client: FakeClient) -> str:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return f"结果：{manifest_path.stem}"


class ParallelRecognitionTest(unittest.TestCase):
    def _manifest(self, root: Path, count: int = 8) -> Path:
        items = []
        for index in range(count):
            name = f"image_{index:02d}.png"
            items.append(
                {
                    "file_name": name,
                    "canonical_file_name": name,
                    "sha256": f"{index:064x}",
                    "image_manifest": str(root / f"manifest_{index:02d}.json"),
                }
            )
        path = root / "dataset_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "config_digest": "parallel-config",
                    "items": items,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_both_branches_run_six_images_in_parallel_and_reuse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            cases = [
                (
                    "long",
                    LongPipeline(
                        LongConfig(backend="pillow"),
                        root / "long_work",
                    ),
                ),
                (
                    "table",
                    TablePipeline(
                        TableConfig(backend="pillow", detector="projection"),
                        root / "table_work",
                    ),
                ),
            ]
            for name, pipeline in cases:
                with self.subTest(branch=name):
                    tracker = ActivityTracker()
                    output = root / f"{name}.csv"
                    with patch.object(
                        pipeline,
                        "_recognize_manifest",
                        side_effect=tracker,
                    ):
                        results = pipeline.recognize_dataset(
                            manifest,
                            FakeClient(),
                            output,
                            max_workers=6,
                        )

                    self.assertEqual(len(results), 8)
                    self.assertEqual(tracker.calls, 8)
                    self.assertGreaterEqual(tracker.maximum_active, 2)
                    self.assertTrue(output.is_file())

                    cached_tracker = ActivityTracker()
                    with patch.object(
                        pipeline,
                        "_recognize_manifest",
                        side_effect=cached_tracker,
                    ):
                        cached = pipeline.recognize_dataset(
                            manifest,
                            FakeClient(),
                            output,
                            max_workers=6,
                        )
                    self.assertEqual(cached, results)
                    self.assertEqual(cached_tracker.calls, 0)

    def test_worker_count_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, count=1)
            pipeline = LongPipeline(
                LongConfig(backend="pillow"),
                root / "work",
            )
            with self.assertRaisesRegex(ValueError, "max_workers"):
                pipeline.recognize_dataset(
                    manifest,
                    FakeClient(),
                    root / "result.csv",
                    max_workers=0,
                )


if __name__ == "__main__":
    unittest.main()
