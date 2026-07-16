"""FireRed-OCR-2B 本地单模型客户端。

这个客户端只实例化一份 FireRed 模型，并用互斥锁串行处理所有图片。它不
导入、不初始化 PaddleOCR-VL 或其他视觉模型，避免 8GB 显存同时驻留两套
权重。外部仍使用项目统一的 ``recognize(image_path, prompt)`` 接口。
"""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Callable

from PIL import Image

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


def align_headings_to_manifest(markdown: str, pack: Any) -> str:
    """按 manifest 已知的最高层级整体平移标题，不根据标题文字猜层级。"""

    expected_levels = [
        int(item.get("level", 0))
        for item in getattr(pack, "heading_hints", ())
        if 1 <= int(item.get("level", 0)) <= 6
    ]
    observed_levels = [
        len(match.group(1))
        for line in markdown.splitlines()
        if (match := re.match(r"^(#{1,6})\s+", line))
    ]
    if not expected_levels or not observed_levels:
        return markdown.strip()

    shift = min(expected_levels) - min(observed_levels)
    if shift == 0:
        return markdown.strip()

    output: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})(\s+.*)$", line)
        if not match:
            output.append(line)
            continue
        level = min(6, max(1, len(match.group(1)) + shift))
        output.append(f"{'#' * level}{match.group(2)}")
    return "\n".join(output).strip()


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
        max_new_tokens: int = 4096,
        processor_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if min_pixels <= 0 or max_pixels <= 0 or max_new_tokens <= 0:
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
        self.max_new_tokens = max_new_tokens
        self.model = (
            f"{model_name}@torch-bf16-single;"
            f"pixels={min_pixels}-{max_pixels};tokens={max_new_tokens}"
        )
        self._lock = Lock()

        print(f"[FireRed] 加载处理器：{model_name}", flush=True)
        self._processor = processor_loader(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
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

    def recognize(self, image_path: str | Path, prompt: str = "") -> str:
        """使用官方 FireRed 提示词识别一张图片；传入 prompt 故意不使用。"""

        del prompt
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"FireRed 输入图片不存在：{image_path}")
        with Image.open(image_path) as image:
            source_size = image.size

        messages = self._messages(image_path)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        started = time.perf_counter()
        with self._lock:
            self._torch.cuda.reset_peak_memory_stats()
            inference_context = getattr(
                self._torch,
                "inference_mode",
                nullcontext,
            )
            with inference_context():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
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
            "model": self.model,
            "source_image": str(image_path),
            "source_width": source_size[0],
            "source_height": source_size[1],
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_memory_gib": round(peak_gib, 3),
            "markdown_characters": len(markdown),
            "single_model_instance": True,
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

        return align_headings_to_manifest(markdown, pack)
