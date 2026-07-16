"""PaddleOCR-VL-1.6 本地 GPU 客户端。

对外保持与 FinixDocClient 相同的 ``recognize(image_path, prompt)`` 接口。
长图根据切块高度选择是否再次做版面检测；图表暂时保留版面检测。两类图片
共用一个常驻模型，但拥有独立的像素、token 和缓存签名参数。
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any, Callable

from PIL import Image

from .vlm_client import LOCAL_PADDLEOCR_PROTOCOL


class PaddleOCRVLClient:
    """单卡常驻 PaddleOCR-VL；所有推理串行进入同一个模型实例。"""

    protocol = LOCAL_PADDLEOCR_PROTOCOL

    def __init__(
        self,
        *,
        pipeline_version: str = "v1.6",
        device: str = "gpu:0",
        max_pixels: int = 300_000,
        table_max_pixels: int = 1_000_000,
        max_new_tokens: int = 1024,
        table_max_new_tokens: int = 4096,
        long_layout_height: int = 2048,
        heartbeat_seconds: float = 30.0,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        limits = (
            max_pixels,
            table_max_pixels,
            max_new_tokens,
            table_max_new_tokens,
            long_layout_height,
            heartbeat_seconds,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("像素、token、高度和心跳间隔必须大于 0")
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
        self.table_max_pixels = table_max_pixels
        self.max_new_tokens = max_new_tokens
        self.table_max_new_tokens = table_max_new_tokens
        self.long_layout_height = long_layout_height
        self.heartbeat_seconds = heartbeat_seconds
        # ResultCache 会把 model 字符串纳入缓存键。路由方式和各分支上限都会
        # 改变识别结果，必须写进签名，不能误拿旧策略生成的缓存。
        self.model = (
            f"PaddleOCR-VL-{pipeline_version}@paddle-gpu"
            f";long=adaptive{long_layout_height}-{max_pixels}px-{max_new_tokens}tok"
            f";table=layout-{table_max_pixels}px-{table_max_new_tokens}tok"
        )
        self._lock = Lock()
        self._pipeline = pipeline_factory(
            pipeline_version=pipeline_version,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            # 短长图块和图表仍需要 PP-DocLayoutV3；高长图块在 predict 时
            # 关闭它。内部 worker 队列在当前 WSL 环境会出现挂起。
            use_layout_detection=True,
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

    @staticmethod
    def _image_height(image_path: Path) -> int:
        with Image.open(image_path) as image:
            return image.height

    def _save_debug(
        self,
        image_path: Path,
        result: Any,
        elapsed_seconds: float,
        runtime: dict[str, Any],
    ) -> None:
        """把模型原始结构结果放回该图片的 prepared 目录，便于逐块检查。"""

        output_dir = image_path.parent.parent / "local_vl_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = result.json
        payload["local_runtime"] = {
            "model": self.model,
            "device": self.device,
            "elapsed_seconds": round(elapsed_seconds, 4),
            **runtime,
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

    def _heartbeat(
        self,
        stop: Event,
        image_name: str,
        branch: str,
        started: float,
    ) -> None:
        """模型没有 token 级日志时，定时报告仍在工作和累计耗时。"""

        while not stop.wait(self.heartbeat_seconds):
            elapsed = time.perf_counter() - started
            print(
                f"[本地 VL/{branch}] {image_name}：仍在识别，已用 {elapsed:.0f}s",
                flush=True,
            )

    def recognize(self, image_path: str | Path, prompt: str = "") -> str:
        """本地解析单张切块；prompt 参数仅为兼容现有客户端接口。"""

        del prompt
        image_path = Path(image_path)
        is_table = image_path.parent.name == "tiles"
        actual_max_pixels = self.table_max_pixels if is_table else self.max_pixels
        actual_max_new_tokens = (
            self.table_max_new_tokens if is_table else self.max_new_tokens
        )
        predict_kwargs: dict[str, Any] = {
            "max_pixels": actual_max_pixels,
            "max_new_tokens": actual_max_new_tokens,
        }

        if is_table:
            branch = "table-layout"
            image_height = self._image_height(image_path)
            use_layout_detection = True
            prompt_label = None
        else:
            image_height = self._image_height(image_path)
            use_layout_detection = image_height <= self.long_layout_height
            if use_layout_detection:
                # 短块通常只有标题、目录或少量正文；版面模型帮助 VLM 快速
                # 结束，避免整块 OCR 偶尔不吐结束符而跑满 token 上限。
                branch = "long-short-layout"
                prompt_label = None
            else:
                # 高块若再次版面检测，可能被拆成十几个内部 VLM 调用。
                branch = "long-tall-ocr"
                prompt_label = "ocr"
                predict_kwargs.update(
                    use_layout_detection=False,
                    prompt_label=prompt_label,
                )

        runtime = {
            "branch": branch,
            "image_height": image_height,
            "use_layout_detection": use_layout_detection,
            "prompt_label": prompt_label,
            "max_pixels": actual_max_pixels,
            "max_new_tokens": actual_max_new_tokens,
        }

        start = time.perf_counter()
        stop_heartbeat = Event()
        heartbeat = Thread(
            target=self._heartbeat,
            args=(stop_heartbeat, image_path.name, branch, start),
            daemon=True,
        )
        heartbeat.start()
        try:
            with self._lock:
                outputs = list(
                    self._pipeline.predict(str(image_path), **predict_kwargs)
                )
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=1)

        elapsed = time.perf_counter() - start
        if len(outputs) != 1:
            raise RuntimeError(
                f"PaddleOCR-VL 单图应返回 1 个结果，实际为 {len(outputs)} 个"
            )
        result = outputs[0]
        markdown = self._markdown_text(result)
        self._save_debug(image_path, result, elapsed, runtime)
        print(
            f"[本地 VL/{branch}] {image_path.name}：{elapsed:.2f}s，"
            f"Markdown {len(markdown)} 字符",
            flush=True,
        )
        return markdown
