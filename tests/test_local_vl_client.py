from pathlib import Path
import tempfile
import unittest

from afac_pipeline.common.local_vl_client import PaddleOCRVLClient


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


class LocalVLClientTest(unittest.TestCase):
    def test_local_client_returns_markdown_and_writes_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "prepared" / "sample" / "vlm_requests" / "tile.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fake-image")

            client = PaddleOCRVLClient(
                max_pixels=123456,
                max_new_tokens=789,
                pipeline_factory=FakePipeline,
            )
            result = client.recognize(image, "这段提示词必须被忽略")

            self.assertEqual(result, "## 本地标题")
            self.assertEqual(client._pipeline.init_kwargs["device"], "gpu:0")
            self.assertEqual(
                client._pipeline.predict_calls[0][1],
                {"max_pixels": 123456, "max_new_tokens": 789},
            )
            self.assertIn("max_pixels=123456", client.model)
            self.assertIn("max_new_tokens=789", client.model)
            raw = image.parent.parent / "local_vl_raw" / "tile.json"
            self.assertTrue(raw.is_file())
            self.assertIn("local_runtime", raw.read_text(encoding="utf-8"))

    def test_8gb_safe_defaults(self) -> None:
        client = PaddleOCRVLClient(pipeline_factory=FakePipeline)
        self.assertEqual(client.max_pixels, 1_000_000)
        self.assertEqual(client.max_new_tokens, 4096)

    def test_limits_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PaddleOCRVLClient(max_pixels=0, pipeline_factory=FakePipeline)


if __name__ == "__main__":
    unittest.main()
