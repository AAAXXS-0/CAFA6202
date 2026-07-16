from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from afac_pipeline.common.firered_vl_client import (
    FIRERED_OFFICIAL_PROMPT,
    FireRedOCRClient,
    _strip_outer_markdown_fence,
    align_headings_to_manifest,
)


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def memory_allocated() -> int:
        return 2 * 1024**3

    @staticmethod
    def reset_peak_memory_stats() -> None:
        return None

    @staticmethod
    def max_memory_allocated() -> int:
        return 3 * 1024**3


class FakeTorch:
    bfloat16 = "bf16"
    cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeInputs(dict):
    input_ids = [[10, 11]]

    def to(self, device: str):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeInputs(input_ids=self.input_ids)

    input_ids = [[10, 11]]

    def batch_decode(self, values, **kwargs):
        return ["```markdown\n# 主标题\n\n正文\n```"]


class FakeModel:
    def __init__(self) -> None:
        self.generate_calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        return [[10, 11, 20, 21]]


class FireRedClientTest(unittest.TestCase):
    def test_single_loaded_model_is_reused_and_official_prompt_is_used(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()
        load_counts = {"processor": 0, "model": 0}

        def load_processor(*args, **kwargs):
            load_counts["processor"] += 1
            return processor

        def load_model(*args, **kwargs):
            load_counts["model"] += 1
            return model

        client = FireRedOCRClient(
            processor_loader=load_processor,
            model_loader=load_model,
            torch_module=FakeTorch(),
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "prepared/sample/tiles/one.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (100, 200), "white").save(image)

            first = client.recognize(image, "这段外部提示词不应进入模型")
            second = client.recognize(image)

            self.assertEqual(first, "# 主标题\n\n正文")
            self.assertEqual(second, first)
            self.assertEqual(load_counts, {"processor": 1, "model": 1})
            self.assertEqual(model.generate_calls, 2)
            sent_prompt = processor.messages[0]["content"][1]["text"]
            self.assertEqual(sent_prompt, FIRERED_OFFICIAL_PROMPT)

    def test_limits_are_checked_before_model_loading(self) -> None:
        with self.assertRaises(ValueError):
            FireRedOCRClient(max_pixels=0, torch_module=FakeTorch())

    def test_outer_fence_is_removed(self) -> None:
        self.assertEqual(
            _strip_outer_markdown_fence("```markdown\n## 标题\n```"),
            "## 标题",
        )

    def test_manifest_shifts_relative_headings_without_reading_title_text(self) -> None:
        pack = SimpleNamespace(
            heading_hints=(
                {"heading_id": "h2", "level": 2},
                {"heading_id": "h3", "level": 3},
            )
        )
        markdown = "# 任意一级文字\n\n## 任意二级文字\n\n正文"

        self.assertEqual(
            align_headings_to_manifest(markdown, pack),
            "## 任意一级文字\n\n### 任意二级文字\n\n正文",
        )


if __name__ == "__main__":
    unittest.main()
