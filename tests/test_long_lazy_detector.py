from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from afac_pipeline.long.config import LongConfig
from afac_pipeline.long.步骤006_全流程调度 import LongPipeline


class LongLazyDetectorTest(unittest.TestCase):
    def test_recognition_pipeline_constructor_does_not_load_yolo(self) -> None:
        with TemporaryDirectory() as directory:
            with patch(
                "afac_pipeline.long.步骤006_全流程调度.GeneralYoloDetector",
                side_effect=AssertionError("识别阶段不应加载 YOLO"),
            ):
                pipeline = LongPipeline(
                    LongConfig(backend="pillow"),
                    Path(directory),
                )

        self.assertIsNone(pipeline.detector)


if __name__ == "__main__":
    unittest.main()
