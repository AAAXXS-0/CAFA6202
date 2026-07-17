from contextlib import nullcontext
import json
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
    promote_numbered_bold_definitions,
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


class FakeImageProcessor:
    max_pixels = 0


class FakeProcessor:
    def __init__(self) -> None:
        self.messages = None
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeInputs(input_ids=self.input_ids)

    input_ids = [[10, 11]]

    def batch_decode(self, values, **kwargs):
        return ["```markdown\n# 主标题\n\n正文\n```"]


class FakeModel:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.max_new_tokens = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.max_new_tokens.append(kwargs["max_new_tokens"])
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
            max_pixels=300000,
            table_max_pixels=654321,
            max_new_tokens=1234,
            table_max_new_tokens=4321,
            processor_loader=load_processor,
            model_loader=load_model,
            torch_module=FakeTorch(),
        )
        with tempfile.TemporaryDirectory() as directory:
            table_image = Path(directory) / "prepared/sample/tiles/one.png"
            long_image = Path(directory) / "prepared/sample/vlm_requests/two.png"
            table_image.parent.mkdir(parents=True)
            long_image.parent.mkdir(parents=True)
            for image_path in (table_image, long_image):
                image = Image.new("RGB", (100, 200), "white")
                image.paste("black", (0, 0, 12, 12))
                image.save(image_path)

            first = client.recognize(
                table_image,
                "这段外部提示词不应进入模型",
            )
            second = client.recognize(long_image)

            self.assertEqual(first, "# 主标题\n\n正文")
            self.assertEqual(second, first)
            self.assertEqual(load_counts, {"processor": 1, "model": 1})
            self.assertEqual(model.generate_calls, 2)
            self.assertEqual(model.max_new_tokens, [4321, 1234])
            self.assertIn("table=654321px-4096tok", client.model)
            self.assertTrue(client.table_cache_model().endswith("table-output=4321tok"))
            self.assertEqual(processor.image_processor.max_pixels, 300000)
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

    def test_manifest_restores_matching_bold_headings(self) -> None:
        pack = SimpleNamespace(
            heading_hints=(
                {"heading_id": "h2", "level": 2},
                {"heading_id": "h3", "level": 3},
            )
        )
        markdown = "**章节标题**\n\n正文\n\n**条目标题**\n\n条目正文"
        self.assertEqual(
            align_headings_to_manifest(markdown, pack),
            "## 章节标题\n\n正文\n\n### 条目标题\n\n条目正文",
        )

    def test_legacy_consecutive_numbered_definitions_become_h4(self) -> None:
        pack = SimpleNamespace(heading_hints=())
        markdown = (
            "**96. 第一项：**\n\n正文\n\n"
            "**97. 第二项：**\n\n正文\n\n"
            "**98. 第三项：**\n\n正文"
        )
        repaired = promote_numbered_bold_definitions(markdown, pack)
        self.assertIn("#### 96. 第一项：", repaired)
        self.assertIn("#### 98. 第三项：", repaired)

    def test_wrapped_h1_is_joined(self) -> None:
        pack = SimpleNamespace(heading_hints=())
        self.assertEqual(
            align_headings_to_manifest(
                "# 产品名称\n# （2026版）条款\n\n正文", pack
            ),
            "# 产品名称 （2026版）条款\n\n正文",
        )

    def test_blank_image_skips_the_only_model(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()
        client = FireRedOCRClient(
            processor_loader=lambda *args, **kwargs: processor,
            model_loader=lambda *args, **kwargs: model,
            torch_module=FakeTorch(),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "prepared/sample/vlm_requests/blank.png"
            )
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (600, 1200), "white").save(image_path)

            markdown = client.recognize(image_path)

            self.assertEqual(markdown, "")
            self.assertEqual(model.generate_calls, 0)
            debug_path = image_path.parent.parent / "firered_raw/blank.json"
            self.assertIn('"skipped_blank": true', debug_path.read_text())

    def test_extreme_toc_is_split_but_reuses_the_same_model(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "prepared/sample/vlm_requests/toc.png"
            )
            image_path.parent.mkdir(parents=True)
            image = Image.new("RGB", (600, 3900), "white")
            for y in range(80, 3850, 40):
                for x in range(30, 260):
                    image.putpixel((x, y), (0, 0, 0))
                for x in range(340, 570):
                    image.putpixel((x, y), (0, 0, 0))
            image.save(image_path)

            client = FireRedOCRClient(
                max_pixels=300000,
                processor_loader=lambda *args, **kwargs: processor,
                model_loader=lambda *args, **kwargs: model,
                torch_module=FakeTorch(),
            )
            pack = SimpleNamespace(
                context_boxes=(),
                context_heading_ids=(),
                visible_heading_ids=(),
                source_box=None,
                body_scale=1.0,
                semantic_role="table_of_contents",
                heading_hints=(),
            )
            markdown = client.recognize_long_pack(
                image_path,
                "",
                pack,
                {"config": {}, "semantic_headings": []},
                10,
            )

            self.assertTrue(markdown)
            self.assertGreaterEqual(model.generate_calls, 4)
            plan_path = (
                image_path.parent.parent
                / "firered_parts/toc/plan.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["column_count"], 2)
            self.assertTrue(plan["single_model_instance"])


if __name__ == "__main__":
    unittest.main()
