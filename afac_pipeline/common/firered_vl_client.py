"""FireRed-OCR-2B 本地单模型客户端。

这个客户端只实例化一份 FireRed 模型，并用互斥锁串行处理所有图片。它不
导入、不初始化 PaddleOCR-VL 或其他视觉模型，避免 8GB 显存同时驻留两套
权重。外部仍使用项目统一的 ``recognize(image_path, prompt)`` 接口。
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Callable

from PIL import Image

from .local_long_split import (
    SPLIT_VERSION,
    create_local_long_parts,
    estimated_model_width,
    leading_header_height,
    merge_local_part_markdowns,
    needs_local_long_split,
)
from .vlm_client import LOCAL_FIRERED_PROTOCOL


FIRERED_OFFICIAL_PROMPT = """You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \\( and \\).
- Enclose block formulas with \\[ and \\].

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines. Output only the converted Markdown without explanations or comments."""


def _strip_outer_markdown_fence(text: str) -> str:
    """仅移除包住整个答案的 Markdown 围栏，不碰正文内部代码块。"""

    value = text.strip()
    match = re.fullmatch(
        r"```(?:markdown|md)?\s*\n?(.*?)\n?```",
        value,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else value


def _join_wrapped_h1(lines: list[str]) -> str:
    """把原图换行导致的连续 H1 合回一个文档标题。"""

    output: list[str] = []
    for line in lines:
        if output and output[-1].startswith("# ") and line.startswith("# "):
            output[-1] = output[-1].rstrip() + " " + line[2:].strip()
        else:
            output.append(line)
    return "\n".join(output).strip()


def promote_numbered_bold_definitions(markdown: str, pack: Any) -> str:
    """无语义提示时，把连续编号的整行粗体定义项保守恢复为 H4。"""

    if getattr(pack, "heading_hints", ()):
        return markdown.strip()
    lines = markdown.strip().splitlines()
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        bold = re.fullmatch(r"\s*\*\*(.+)\*\*\s*", line)
        if not bold:
            continue
        text = bold.group(1).strip()
        numbered = re.match(r"^(\d{1,4})[.．、]\s*(.+)$", text)
        if numbered and text.rstrip().endswith(("：", ":")):
            candidates.append((index, int(numbered.group(1)), text))
    if len(candidates) < 3:
        return markdown.strip()
    numbers = [number for _, number, _ in candidates]
    consecutive = sum(
        right == left + 1 for left, right in zip(numbers, numbers[1:])
    )
    if consecutive < len(numbers) - 2:
        return markdown.strip()
    replacements = {index: f"#### {text}" for index, _, text in candidates}
    return "\n".join(
        replacements.get(index, line) for index, line in enumerate(lines)
    ).strip()


def align_headings_to_manifest(markdown: str, pack: Any) -> str:
    """用 manifest 校准标题，并修复 FireRed 输出的整行粗体标题。"""

    expected_levels = [
        int(item.get("level", 0))
        for item in getattr(pack, "heading_hints", ())
        if 1 <= int(item.get("level", 0)) <= 6
    ]
    lines = markdown.strip().splitlines()
    heading_rows = [
        index for index, line in enumerate(lines)
        if re.match(r"^(#{1,6})\s+", line)
    ]
    bold_rows = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := re.fullmatch(r"\s*\*\*(.+)\*\*\s*", line))
    ]

    # 只有候选数量完全一致才提升粗体，避免把普通强调文字误当标题。
    if expected_levels and not heading_rows and len(bold_rows) == len(expected_levels):
        for level, (index, text) in zip(expected_levels, bold_rows):
            lines[index] = f"{'#' * level} {text}"
        return _join_wrapped_h1(lines)

    if not expected_levels or not heading_rows:
        return _join_wrapped_h1(lines)

    if len(expected_levels) == len(heading_rows):
        for level, index in zip(expected_levels, heading_rows):
            text = re.sub(r"^#{1,6}\s+", "", lines[index])
            lines[index] = f"{'#' * level} {text}"
        return _join_wrapped_h1(lines)

    observed_levels = [
        len(re.match(r"^(#{1,6})", lines[index]).group(1))
        for index in heading_rows
    ]
    shift = min(expected_levels) - min(observed_levels)
    if shift:
        for index in heading_rows:
            match = re.match(r"^(#{1,6})(\s+.*)$", lines[index])
            level = min(6, max(1, len(match.group(1)) + shift))
            lines[index] = f"{'#' * level}{match.group(2)}"
    return _join_wrapped_h1(lines)


class FireRedOCRClient:
    """一张 GPU 上常驻一份 FireRed-OCR-2B，所有推理严格串行。"""

    protocol = LOCAL_FIRERED_PROTOCOL

    def __init__(
        self,
        *,
        model_name: str = "FireRedTeam/FireRed-OCR",
        device: str = "cuda:0",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        table_max_pixels: int = 2048 * 28 * 28,
        max_new_tokens: int = 4096,
        table_max_new_tokens: int = 8192,
        processor_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if any(
            value <= 0
            for value in (
                min_pixels,
                max_pixels,
                table_max_pixels,
                max_new_tokens,
                table_max_new_tokens,
            )
        ):
            raise ValueError("FireRed 像素和输出 token 上限必须大于 0")
        if min_pixels > max_pixels:
            raise ValueError("FireRed 最小像素数不能超过最大像素数")

        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as error:
                raise RuntimeError(
                    "没有找到 PyTorch；请使用独立 AFAC_FIRERED 环境"
                ) from error
        self._torch = torch_module
        if not self._torch.cuda.is_available():
            raise RuntimeError("FireRed 没有检测到 CUDA GPU，已停止以避免 CPU 慢跑")

        if processor_loader is None or model_loader is None:
            try:
                from transformers import (
                    AutoProcessor,
                    Qwen3VLForConditionalGeneration,
                )
            except ImportError as error:
                raise RuntimeError(
                    "没有找到 FireRed 所需 Transformers 依赖"
                ) from error
            processor_loader = processor_loader or AutoProcessor.from_pretrained
            model_loader = (
                model_loader
                or Qwen3VLForConditionalGeneration.from_pretrained
            )

        self.model_name = model_name
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.table_max_pixels = table_max_pixels
        self.max_new_tokens = max_new_tokens
        self.table_max_new_tokens = table_max_new_tokens
        # self.model 保持旧长图缓存身份；图表另用独立缓存身份，
        # 避免只调整图表输出上限，却让已完成的长图缓存全部失效。
        self.model = (
            f"{model_name}@torch-bf16-single;"
            f"long={min_pixels}-{max_pixels}px-{max_new_tokens}tok;"
            f"table={table_max_pixels}px-4096tok"
        )
        self.table_model = (
            f"{self.model};table-output={table_max_new_tokens}tok"
        )
        self._lock = Lock()

        print(f"[FireRed] 加载处理器：{model_name}", flush=True)
        self._processor = processor_loader(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max(max_pixels, table_max_pixels),
        )
        print("[FireRed] 加载唯一模型实例到 cuda:0", flush=True)
        self._model = model_loader(
            model_name,
            dtype=self._torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self._model.eval()
        print(
            f"[FireRed] 模型就绪，显存 {self._allocated_gib():.2f} GiB",
            flush=True,
        )

    def _allocated_gib(self) -> float:
        return float(self._torch.cuda.memory_allocated()) / 1024**3

    @staticmethod
    def _messages(image_path: Path) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": FIRERED_OFFICIAL_PROMPT},
                ],
            }
        ]

    @staticmethod
    def _debug_path(image_path: Path) -> Path:
        output_dir = image_path.parent.parent / "firered_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{image_path.stem}.json"

    @staticmethod
    def _blank_metrics(image_path: Path) -> dict[str, Any]:
        """极保守地跳过纯白或完全均匀块，避免空块白白调用大模型。"""

        with Image.open(image_path) as image:
            gray = image.convert("L")
            gray.thumbnail((512, 512), Image.Resampling.BOX)
        histogram = gray.histogram()
        pixel_count = gray.width * gray.height
        ink_pixels = sum(histogram[:245])
        dark_pixels = sum(histogram[:225])
        allowed_ink = max(2, int(pixel_count * 0.00001))
        gray_min, gray_max = gray.getextrema()
        uniform_span = gray_max - gray_min
        skipped = (
            ink_pixels <= allowed_ink and dark_pixels == 0
        ) or uniform_span <= 2
        return {
            "skipped_blank": skipped,
            "preview_width": gray.width,
            "preview_height": gray.height,
            "ink_pixels": ink_pixels,
            "dark_pixels": dark_pixels,
            "allowed_ink_pixels": allowed_ink,
            "gray_min": gray_min,
            "gray_max": gray_max,
            "uniform_span": uniform_span,
        }

    def recognize(self, image_path: str | Path, prompt: str = "") -> str:
        """使用官方 FireRed 提示词识别一张图片；传入 prompt 故意不使用。"""

        del prompt
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"FireRed 输入图片不存在：{image_path}")
        with Image.open(image_path) as image:
            source_size = image.size

        blank_metrics = self._blank_metrics(image_path)
        if blank_metrics["skipped_blank"]:
            debug = {
                "model": self.model,
                "source_image": str(image_path),
                "source_width": source_size[0],
                "source_height": source_size[1],
                "elapsed_seconds": 0.0,
                "peak_gpu_memory_gib": round(self._allocated_gib(), 3),
                "markdown_characters": 0,
                "single_model_instance": True,
                "blank_detection": blank_metrics,
            }
            self._debug_path(image_path).write_text(
                json.dumps(debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[FireRed] {image_path.name}：空白块，跳过模型", flush=True)
            return ""

        is_table = "tiles" in image_path.parts
        actual_max_pixels = (
            self.table_max_pixels if is_table else self.max_pixels
        )
        actual_max_new_tokens = (
            self.table_max_new_tokens if is_table else self.max_new_tokens
        )
        messages = self._messages(image_path)
        started = time.perf_counter()
        with self._lock:
            self._torch.cuda.reset_peak_memory_stats()
            image_processor = getattr(self._processor, "image_processor", None)
            if image_processor is not None:
                image_processor.max_pixels = actual_max_pixels
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            inference_context = getattr(
                self._torch,
                "inference_mode",
                nullcontext,
            )
            with inference_context():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=actual_max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        elapsed = time.perf_counter() - started
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        markdown = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        markdown = _strip_outer_markdown_fence(markdown)
        if not markdown:
            raise RuntimeError("FireRed-OCR 返回了空 Markdown")

        peak_gib = float(self._torch.cuda.max_memory_allocated()) / 1024**3
        debug = {
            "model": self.table_model if is_table else self.model,
            "source_image": str(image_path),
            "source_width": source_size[0],
            "source_height": source_size[1],
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_memory_gib": round(peak_gib, 3),
            "markdown_characters": len(markdown),
            "branch": "table" if is_table else "long",
            "max_pixels": actual_max_pixels,
            "max_new_tokens": actual_max_new_tokens,
            "single_model_instance": True,
            "blank_detection": blank_metrics,
        }
        self._debug_path(image_path).write_text(
            json.dumps(debug, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[FireRed] {image_path.name}：{elapsed:.2f}s，"
            f"峰值显存 {peak_gib:.2f} GiB，Markdown {len(markdown)} 字符",
            flush=True,
        )
        return markdown

    @staticmethod
    def postprocess_long_pack(markdown: str, pack: Any) -> str:
        """让块内相对标题从 manifest 指定的 H1/H2/H3 起点开始。"""

        repaired = promote_numbered_bold_definitions(markdown, pack)
        return align_headings_to_manifest(repaired, pack)

    def table_cache_model(self) -> str:
        """图表缓存单独记录输出 token 上限，不影响长图缓存。"""

        return self.table_model


    def table_legacy_cache_models(self) -> tuple[str, ...]:
        """旧图表响应只有完整可解析时才由流水线迁移复用。"""

        return (self.model,)

    def long_pack_cache_model(self, image_path: str | Path) -> str:
        """极端长图把临时切割版本写入外层 SQLite 缓存键。"""

        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if needs_local_long_split(
            width,
            height,
            self.max_pixels,
            trigger_height=2048,
            minimum_estimated_width=512,
        ):
            return f"{self.model};{SPLIT_VERSION}-firered-v1"
        return self.model

    def recognize_long_pack(
        self,
        image_path: str | Path,
        prompt: str,
        pack: Any,
        image_manifest: dict[str, Any],
        context_gap: int,
    ) -> str:
        """只对会被缩得过窄的长图临时切割，所有子块仍由同一模型顺序识别。"""

        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if not needs_local_long_split(
            width,
            height,
            self.max_pixels,
            trigger_height=2048,
            minimum_estimated_width=512,
        ):
            return self.recognize(image_path, prompt)

        header_height, header_ids = leading_header_height(
            pack,
            image_manifest,
            context_gap,
            height,
        )
        split_root = image_path.parent.parent / "firered_parts" / image_path.stem
        parts = create_local_long_parts(
            image_path,
            split_root,
            header_height=header_height,
            target_height=1500,
            maximum_height=1800,
            minimum_content_height=700,
            search_radius=260,
            fallback_overlap=128,
            sample_width=512,
            white_threshold=225,
            blank_ratio=0.002,
            minimum_blank_height=3,
            split_columns=str(
                getattr(pack, "semantic_role", "")
            ).startswith("table_of_contents"),
        )
        if not parts:
            return self.recognize(image_path, prompt)

        split_version = f"{SPLIT_VERSION}-firered-v1"
        estimated_width = estimated_model_width(
            width,
            height,
            self.max_pixels,
        )
        plan = {
            "version": split_version,
            "source_image": str(image_path.resolve()),
            "source_width": width,
            "source_height": height,
            "estimated_model_width_without_split": round(estimated_width, 2),
            "header_height": header_height,
            "header_heading_ids": list(header_ids),
            "semantic_role": str(getattr(pack, "semantic_role", "unknown")),
            "part_count": len(parts),
            "column_count": max(item.column_count for item in parts),
            "single_model_instance": True,
            "parts": [item.to_dict() for item in parts],
        }
        plan_path = split_root / "plan.json"
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[FireRed/长图临时切割] {image_path.name}："
            f"{width}x{height}，预计模型宽度 {estimated_width:.0f}px，"
            f"顺序识别 {len(parts)} 个子块",
            flush=True,
        )

        responses_dir = split_root / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        markdowns: list[str] = []
        for index, part in enumerate(parts, start=1):
            part_path = split_root / "parts" / part.file_name
            response_path = responses_dir / f"part_{part.index:03d}.md"
            metadata_path = responses_dir / f"part_{part.index:03d}.json"
            image_sha256 = hashlib.sha256(part_path.read_bytes()).hexdigest()
            reusable = False
            if response_path.is_file() and metadata_path.is_file():
                try:
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    reusable = (
                        metadata.get("image_sha256") == image_sha256
                        and metadata.get("model") == self.model
                        and metadata.get("split_version") == split_version
                    )
                except (OSError, json.JSONDecodeError):
                    reusable = False

            if reusable:
                markdown = response_path.read_text(encoding="utf-8")
                source = "缓存"
            else:
                print(
                    f"[FireRed/长图子块 {index:02d}/{len(parts):02d}] "
                    f"{part.file_name}",
                    flush=True,
                )
                markdown = self.recognize(part_path, prompt)
                response_path.write_text(markdown, encoding="utf-8")
                metadata_path.write_text(
                    json.dumps(
                        {
                            "image_sha256": image_sha256,
                            "model": self.model,
                            "split_version": split_version,
                            "part": part.to_dict(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                source = "模型"
            markdowns.append(markdown)
            print(
                f"[FireRed/长图子块 {index:02d}/{len(parts):02d}] "
                f"{source}完成，Markdown {len(markdown)} 字符",
                flush=True,
            )

        repeated_heading_count = len(header_ids)
        if header_height > 0 and repeated_heading_count == 0:
            repeated_heading_count = 1
        merged = merge_local_part_markdowns(
            markdowns,
            parts,
            repeated_heading_count=repeated_heading_count,
        )
        (split_root / "merged.md").write_text(merged, encoding="utf-8")
        plan["markdown_characters"] = len(merged)
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return merged
