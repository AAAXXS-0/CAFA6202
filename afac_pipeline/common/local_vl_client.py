"""PaddleOCR-VL-1.6 本地 GPU 客户端。

对外保持与 FinixDocClient 相同的 ``recognize(image_path, prompt)`` 接口，
这样长图、图表现有的缓存和聚合代码无需复制。PaddleOCR-VL 使用自身按版面
标签选择的内部提示，因此这里不会接收上层自定义提示词。
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
import time
from typing import Any, Callable

from .vlm_client import LOCAL_PADDLEOCR_PROTOCOL


class PaddleOCRVLClient:
    """单卡常驻 PaddleOCR-VL；所有推理串行进入同一个模型实例。"""

    protocol = LOCAL_PADDLEOCR_PROTOCOL

    def __init__(
        self,
        *,
        pipeline_version: str = "v1.6",
        device: str = "gpu:0",
        max_pixels: int = 1_000_000,
        max_new_tokens: int = 4096,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        if max_pixels <= 0 or max_new_tokens <= 0:
            raise ValueError("max_pixels 和 max_new_tokens 必须大于 0")
        if pipeline_factory is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as error:
                raise RuntimeError(
                    "没有找到 PaddleOCR-VL。本项目应由 AFAC_LOCAL_VL 环境运行"
                ) from error
            pipeline_factory = PaddleOCRVL

        self.pipeline_version = pipeline_version
        self.device = device
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        # ResultCache 会把 model 字符串纳入缓存键。像素上限和输出长度会改变
        # 识别结果，因此也写进签名；以后调参不会误拿旧参数的结果。
        self.model = (
            f"PaddleOCR-VL-{pipeline_version}@paddle-gpu"
            f";max_pixels={max_pixels};max_new_tokens={max_new_tokens}"
        )
        self._lock = Lock()
        self._pipeline = pipeline_factory(
            pipeline_version=pipeline_version,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            # 外层已经由锁保证 GPU 单并行。关闭内部 worker 队列可避免在
            # WSL/Paddle 动态图组合下出现“显存占满但 CPU 空转”的卡住现象。
            use_queues=False,
        )

    @staticmethod
    def _markdown_text(result: Any) -> str:
        markdown = result.markdown
        value = markdown.get("markdown_texts", "")
        if isinstance(value, list):
            value = "\n\n".join(str(item) for item in value)
        text = str(value).strip()
        if not text:
            raise RuntimeError("PaddleOCR-VL 返回了空 Markdown")
        return text

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    def _save_debug(
        self,
        image_path: Path,
        result: Any,
        elapsed_seconds: float,
    ) -> None:
        """把模型原始结构结果放回该图片的 prepared 目录，便于逐块检查。"""

        output_dir = image_path.parent.parent / "local_vl_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = result.json
        payload["local_runtime"] = {
            "model": self.model,
            "device": self.device,
            "max_pixels": self.max_pixels,
            "max_new_tokens": self.max_new_tokens,
            "elapsed_seconds": round(elapsed_seconds, 4),
        }
        (output_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            ),
            encoding="utf-8",
        )

    def recognize(self, image_path: str | Path, prompt: str = "") -> str:
        """本地解析单张切块；prompt 参数仅为兼容现有客户端接口。"""

        del prompt
        image_path = Path(image_path)
        start = time.perf_counter()
        # Paddle 动态图模型和单张 8GB 显卡都不适合被多个 Python 线程同时进入。
        with self._lock:
            outputs = list(
                self._pipeline.predict(
                    str(image_path),
                    max_pixels=self.max_pixels,
                    max_new_tokens=self.max_new_tokens,
                )
            )
        elapsed = time.perf_counter() - start
        if len(outputs) != 1:
            raise RuntimeError(
                f"PaddleOCR-VL 单图应返回 1 个结果，实际为 {len(outputs)} 个"
            )
        result = outputs[0]
        markdown = self._markdown_text(result)
        self._save_debug(image_path, result, elapsed)
        print(
            f"[本地 VL] {image_path.name}：{elapsed:.2f}s，"
            f"Markdown {len(markdown)} 字符",
            flush=True,
        )
        return markdown
