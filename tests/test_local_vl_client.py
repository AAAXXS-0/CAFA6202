from pathlib import Path
import json
import tempfile
import time
import unittest

from PIL import Image

from afac_pipeline.common.local_vl_client import PaddleOCRVLClient
from afac_pipeline.common.models import Box
from afac_pipeline.long.步骤005_大模型请求打包 import RecognitionPack


class FakeResult:
    markdown = {"markdown_texts": "## 本地标题"}
    json = {"res": {"parsing_res_list": []}}


class FakePipeline:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.predict_calls = []

    def predict(self, image_path: str, **kwargs):
        self.predict_calls.append((image_path, kwargs))
        return [FakeResult()]


class SlowPipeline(FakePipeline):
    def predict(self, image_path: str, **kwargs):
        self.predict_calls.append((image_path, kwargs))
        time.sleep(5)
        return [FakeResult()]


class EmptyResult:
    markdown = {"markdown_texts": ""}
    json = {"res": {"parsing_res_list": []}}


class EmptyPipeline(FakePipeline):
    def predict(self, image_path: str, **kwargs):
        self.predict_calls.append((image_path, kwargs))
        return [EmptyResult()]


class LocalVLClientTest(unittest.TestCase):
    @staticmethod
    def _image(path: Path, width: int, height: int) -> None:
        path.parent.mkdir(parents=True)
        image = Image.new("RGB", (width, height), "white")
        # 常规路由测试使用一个明确黑块，避免被新增的空白短路逻辑跳过。
        for x in range(min(12, width)):
            for y in range(min(12, height)):
                image.putpixel((x, y), (0, 0, 0))
        image.save(path)

    def test_short_long_block_keeps_layout_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/tile.png"
            self._image(image, 100, 100)
            client = PaddleOCRVLClient(
                max_pixels=123456,
                max_new_tokens=789,
                heartbeat_seconds=60,
                pipeline_factory=FakePipeline,
            )
            result = client.recognize(image, "这段提示词必须被忽略")

            self.assertEqual(result, "## 本地标题")
            self.assertEqual(client._pipeline.init_kwargs["device"], "gpu:0")
            self.assertFalse(client._pipeline.init_kwargs["use_queues"])
            self.assertEqual(
                client._pipeline.predict_calls[0][1],
                {"max_pixels": 123456, "max_new_tokens": 789},
            )
            raw = image.parent.parent / "local_vl_raw/tile.json"
            self.assertTrue(raw.is_file())
            self.assertIn("long-short-layout", raw.read_text(encoding="utf-8"))

    def test_tall_long_block_uses_single_ocr_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/tall.png"
            self._image(image, 100, 2200)
            client = PaddleOCRVLClient(
                max_pixels=300000,
                max_new_tokens=1024,
                pipeline_factory=FakePipeline,
            )
            client.recognize(image)
            self.assertEqual(
                client._pipeline.predict_calls[0][1],
                {
                    "max_pixels": 300000,
                    "max_new_tokens": 1024,
                    "use_layout_detection": False,
                    "prompt_label": "ocr",
                },
            )

    def test_extreme_toc_is_split_and_each_part_reuses_toc_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "prepared/sample/vlm_requests/toc.png"
            image.parent.mkdir(parents=True)
            canvas = Image.new("RGB", (600, 3900), "white")
            for x in range(180, 420):
                for y in range(15, 55):
                    canvas.putpixel((x, y), (0, 0, 0))
            for y in range(100, 3850, 35):
                for x in range(40, 560):
                    for offset in range(8):
                        canvas.putpixel((x, y + offset), (0, 0, 0))
            canvas.save(image)

            pack = RecognitionPack(
                "request_00001",
                Box(0, 100, 600, 4000),
                ("table_of_contents",),
                (),
                (),
                image.name,
                visible_heading_ids=("toc",),
                semantic_role="table_of_contents",
            )
            manifest = {
                "config": {"semantic_title_padding": 10},
                "semantic_headings": [
                    {
                        "id": "toc",
                        "box": {
                            "x1": 0,
                            "y1": 100,
                            "x2": 600,
                            "y2": 160,
                        },
                    }
                ],
            }
            client = PaddleOCRVLClient(pipeline_factory=FakePipeline)

            markdown = client.recognize_long_pack(
                image,
                "",
                pack,
                manifest,
                10,
            )

            self.assertGreaterEqual(len(client._pipeline.predict_calls), 3)
            self.assertEqual(markdown.count("## 本地标题"), 1)
            split_root = image.parent.parent / "local_vl_parts/toc"
            plan = json.loads((split_root / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["header_heading_ids"], ["toc"])
            self.assertGreater(plan["header_height"], 0)
            first_call_count = len(client._pipeline.predict_calls)

            cached = client.recognize_long_pack(image, "", pack, manifest, 10)
            self.assertEqual(cached, markdown)
            self.assertEqual(len(client._pipeline.predict_calls), first_call_count)

    def test_single_prediction_has_hard_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/slow.png"
            self._image(image, 100, 100)
            client = PaddleOCRVLClient(
                inference_timeout_seconds=0.05,
                pipeline_factory=SlowPipeline,
            )

            with self.assertRaisesRegex(RuntimeError, "超过 0.05s"):
                client.recognize(image)

    def test_table_uses_independent_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/tiles/table.png"
            self._image(image, 100, 100)
            client = PaddleOCRVLClient(
                table_max_pixels=654321,
                table_max_new_tokens=2048,
                pipeline_factory=FakePipeline,
            )
            client.recognize(image)
            self.assertEqual(
                client._pipeline.predict_calls[0][1],
                {"max_pixels": 654321, "max_new_tokens": 2048},
            )

    def test_8gb_safe_defaults(self) -> None:
        client = PaddleOCRVLClient(pipeline_factory=FakePipeline)
        self.assertEqual(client.max_pixels, 300_000)
        self.assertEqual(client.max_new_tokens, 1024)
        self.assertEqual(client.table_max_pixels, 1_000_000)
        self.assertEqual(client.table_max_new_tokens, 4096)

    def test_limits_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PaddleOCRVLClient(max_pixels=0, pipeline_factory=FakePipeline)

    def test_blank_image_skips_pipeline_and_writes_debug_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/blank.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (600, 1200), "white").save(image)
            client = PaddleOCRVLClient(pipeline_factory=FakePipeline)

            self.assertEqual(client.recognize(image), "")
            self.assertEqual(client._pipeline.predict_calls, [])
            raw = image.parent.parent / "local_vl_raw/blank.json"
            payload = raw.read_text(encoding="utf-8")
            self.assertIn('"skipped_blank": true', payload)
            self.assertIn('"ink_pixels": 0', payload)

    def test_uniform_gray_image_is_also_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/gray.png"
            image.parent.mkdir(parents=True)
            # 正式数据里出现过整张灰度恒为 244 的空白切块。
            Image.new("L", (1500, 107), 244).save(image)
            client = PaddleOCRVLClient(pipeline_factory=FakePipeline)

            self.assertEqual(client.recognize(image), "")
            self.assertEqual(client._pipeline.predict_calls, [])

    def test_faint_content_is_not_misclassified_as_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/faint.png"
            image.parent.mkdir(parents=True)
            faint = Image.new("L", (512, 512), 255)
            # 240 比近白阈值只深一点，模拟浅灰文字或细线。
            for x in range(80, 430):
                faint.putpixel((x, 250), 240)
            faint.save(image)
            client = PaddleOCRVLClient(pipeline_factory=FakePipeline)

            self.assertEqual(client.recognize(image), "## 本地标题")
            self.assertEqual(len(client._pipeline.predict_calls), 1)

    def test_nonblank_image_with_empty_model_result_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/vlm_requests/ink.png"
            self._image(image, 100, 100)
            client = PaddleOCRVLClient(pipeline_factory=EmptyPipeline)

            with self.assertRaisesRegex(RuntimeError, "检测到了明显墨迹"):
                client.recognize(image)
            raw = image.parent.parent / "local_vl_raw/ink.json"
            self.assertTrue(raw.is_file())


if __name__ == "__main__":
    unittest.main()
