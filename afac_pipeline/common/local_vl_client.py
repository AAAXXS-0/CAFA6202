"""PaddleOCR-VL-1.6 本地 GPU 客户端。

对外保持与 FinixDocClient 相同的 ``recognize(image_path, prompt)`` 接口。
长图根据切块高度选择是否再次做版面检测；极端长宽比请求会在本地适配层
沿空白带临时拆开并复制标题头。图表暂时保留版面检测。两类图片共用一个
常驻模型，但拥有独立的像素、token 和缓存签名参数。
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
from threading import Event, Lock, Thread, current_thread, main_thread
import time
from typing import Any, Callable, Iterator

from PIL import Image

from .local_long_split import (
    SPLIT_VERSION,
    create_local_long_parts,
    estimated_model_width,
    leading_header_height,
    merge_local_part_markdowns,
    needs_local_long_split,
    restore_local_markdown_headings,
)
from .vlm_client import LOCAL_PADDLEOCR_PROTOCOL


class PaddleOCRVLClient:
    """单卡常驻 PaddleOCR-VL；所有推理串行进入同一个模型实例。"""

    # 空白判定故意很保守：缩略图中只要存在少量明显墨迹，就仍然交给模型。
    # 版本单独写进调试记录，但不改 model 缓存签名，避免本次容错修复使已经
    # 跑完的非空切块全部失去缓存。
    BLANK_DETECTION_VERSION = "blank-v1"

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
        blank_preview_long_edge: int = 512,
        blank_gray_threshold: int = 245,
        blank_dark_threshold: int = 225,
        blank_max_ink_ratio: float = 0.00001,
        blank_max_dark_pixels: int = 0,
        blank_uniform_span: int = 2,
        long_split_trigger_height: int = 2048,
        long_split_minimum_model_width: int = 512,
        long_subblock_target_height: int = 1500,
        long_subblock_max_height: int = 1800,
        long_subblock_min_content_height: int = 700,
        long_subblock_search_radius: int = 260,
        long_subblock_fallback_overlap: int = 128,
        long_subblock_sample_width: int = 512,
        long_subblock_white_threshold: int = 225,
        long_subblock_blank_ratio: float = 0.002,
        long_subblock_minimum_blank_height: int = 3,
        inference_timeout_seconds: int = 120,
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
        if blank_preview_long_edge <= 0:
            raise ValueError("空白检测缩略图长边必须大于 0")
        if not 0 <= blank_dark_threshold <= blank_gray_threshold <= 255:
            raise ValueError(
                "空白检测灰度阈值必须满足 0 <= 深色阈值 <= 墨迹阈值 <= 255"
            )
        if not 0 <= blank_max_ink_ratio < 1:
            raise ValueError("空白检测最大墨迹比例必须位于 [0, 1) 内")
        if blank_max_dark_pixels < 0:
            raise ValueError("空白检测最大深色像素数不能小于 0")
        if not 0 <= blank_uniform_span <= 255:
            raise ValueError("空白检测最大均匀灰度跨度必须位于 [0, 255] 内")
        local_split_values = (
            long_split_trigger_height,
            long_split_minimum_model_width,
            long_subblock_target_height,
            long_subblock_max_height,
            long_subblock_min_content_height,
            long_subblock_search_radius,
            long_subblock_sample_width,
            long_subblock_minimum_blank_height,
            inference_timeout_seconds,
        )
        if any(value <= 0 for value in local_split_values):
            raise ValueError("本地长图切割尺寸、搜索范围和超时时间必须大于 0")
        if long_subblock_target_height > long_subblock_max_height:
            raise ValueError("本地长图目标高度不能超过最大高度")
        if not 0 <= long_subblock_fallback_overlap < long_subblock_max_height:
            raise ValueError("本地长图兜底重叠必须位于 [0, 最大高度) 内")
        if not 0 <= long_subblock_white_threshold <= 255:
            raise ValueError("本地长图白色阈值必须位于 [0, 255] 内")
        if not 0 <= long_subblock_blank_ratio < 1:
            raise ValueError("本地长图空白比例必须位于 [0, 1) 内")
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
        self.blank_preview_long_edge = blank_preview_long_edge
        self.blank_gray_threshold = blank_gray_threshold
        self.blank_dark_threshold = blank_dark_threshold
        self.blank_max_ink_ratio = blank_max_ink_ratio
        self.blank_max_dark_pixels = blank_max_dark_pixels
        self.blank_uniform_span = blank_uniform_span
        self.long_split_trigger_height = long_split_trigger_height
        self.long_split_minimum_model_width = long_split_minimum_model_width
        self.long_subblock_target_height = long_subblock_target_height
        self.long_subblock_max_height = long_subblock_max_height
        self.long_subblock_min_content_height = long_subblock_min_content_height
        self.long_subblock_search_radius = long_subblock_search_radius
        self.long_subblock_fallback_overlap = long_subblock_fallback_overlap
        self.long_subblock_sample_width = long_subblock_sample_width
        self.long_subblock_white_threshold = long_subblock_white_threshold
        self.long_subblock_blank_ratio = long_subblock_blank_ratio
        self.long_subblock_minimum_blank_height = long_subblock_minimum_blank_height
        self.inference_timeout_seconds = inference_timeout_seconds
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
        return str(value).strip()

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    @staticmethod
    def _image_height(image_path: Path) -> int:
        with Image.open(image_path) as image:
            return image.height

    def _blank_metrics(self, image_path: Path) -> dict[str, Any]:
        """在小缩略图上判断是否近乎纯白，并返回可核查的统计数据。

        不按“白色占 99%”这类宽松条件判断，因为页边的一行小字也可能只占
        很少像素。只有近白阈值以下的像素几乎为零，并且完全没有更深像素
        时，才把切块视为空白。
        """

        with Image.open(image_path) as image:
            # 透明图片先铺到白底，否则透明黑色会被误当成墨迹。
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                gray = Image.alpha_composite(white, rgba).convert("L")
            else:
                gray = image.convert("L")
            gray.thumbnail(
                (self.blank_preview_long_edge, self.blank_preview_long_edge),
                Image.Resampling.BOX,
            )

        histogram = gray.histogram()
        pixel_count = gray.width * gray.height
        ink_pixels = sum(histogram[: self.blank_gray_threshold])
        dark_pixels = sum(histogram[: self.blank_dark_threshold])
        # 至少容忍两个孤立噪点；大图再按极低比例增加容忍量。
        allowed_ink_pixels = max(2, int(pixel_count * self.blank_max_ink_ratio))
        gray_min, gray_max = gray.getextrema()
        near_white_blank = (
            ink_pixels <= allowed_ink_pixels
            and dark_pixels <= self.blank_max_dark_pixels
        )
        # 有些预处理空块不是白色，而是整块同色的浅灰背景。颜色完全均匀
        # 同样不可能承载文字；只容忍 2 级灰度浮动，避免吞掉浅色细字。
        uniform_span = gray_max - gray_min
        uniform_blank = uniform_span <= self.blank_uniform_span
        is_blank = near_white_blank or uniform_blank
        return {
            "version": self.BLANK_DETECTION_VERSION,
            "skipped_blank": is_blank,
            "preview_width": gray.width,
            "preview_height": gray.height,
            "pixel_count": pixel_count,
            "gray_threshold": self.blank_gray_threshold,
            "dark_threshold": self.blank_dark_threshold,
            "ink_pixels": ink_pixels,
            "ink_ratio": ink_pixels / pixel_count if pixel_count else 0.0,
            "allowed_ink_pixels": allowed_ink_pixels,
            "dark_pixels": dark_pixels,
            "allowed_dark_pixels": self.blank_max_dark_pixels,
            "gray_min": gray_min,
            "gray_max": gray_max,
            "uniform_span": uniform_span,
            "allowed_uniform_span": self.blank_uniform_span,
            "near_white_blank": near_white_blank,
            "uniform_blank": uniform_blank,
        }

    @staticmethod
    def _debug_path(image_path: Path) -> Path:
        output_dir = image_path.parent.parent / "local_vl_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{image_path.stem}.json"

    def _save_blank_debug(
        self,
        image_path: Path,
        elapsed_seconds: float,
        runtime: dict[str, Any],
        blank_metrics: dict[str, Any],
    ) -> None:
        """记录未进入模型的空白块，便于确认它为何被跳过。"""

        payload = {
            "res": {"parsing_res_list": []},
            "local_runtime": {
                "model": self.model,
                "device": self.device,
                "elapsed_seconds": round(elapsed_seconds, 4),
                **runtime,
            },
            "blank_detection": blank_metrics,
        }
        self._debug_path(image_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_debug(
        self,
        image_path: Path,
        result: Any,
        elapsed_seconds: float,
        runtime: dict[str, Any],
    ) -> None:
        """把模型原始结构结果放回该图片的 prepared 目录，便于逐块检查。"""

        payload = dict(result.json)
        payload["local_runtime"] = {
            "model": self.model,
            "device": self.device,
            "elapsed_seconds": round(elapsed_seconds, 4),
            **runtime,
        }
        self._debug_path(image_path).write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            ),
            encoding="utf-8",
        )

    @contextmanager
    def _prediction_timeout(self, image_name: str) -> Iterator[None]:
        """主线程推理超过上限时主动失败，避免 Paddle 永久占满单个 CPU 核心。"""

        can_interrupt = (
            current_thread() is main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
        )
        if not can_interrupt:
            yield
            return

        def timeout_handler(_signum: int, _frame: Any) -> None:
            raise RuntimeError(
                f"PaddleOCR-VL 识别 {image_name} 超过 "
                f"{self.inference_timeout_seconds}s，已中断本次运行；"
                "已完成的子块缓存仍可复用"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.inference_timeout_seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

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
        blank_metrics = self._blank_metrics(image_path)
        if blank_metrics["skipped_blank"]:
            elapsed = time.perf_counter() - start
            self._save_blank_debug(image_path, elapsed, runtime, blank_metrics)
            print(
                f"[本地 VL/{branch}] {image_path.name}：检测为空白切块，"
                "跳过模型并继续",
                flush=True,
            )
            return ""

        stop_heartbeat = Event()
        heartbeat = Thread(
            target=self._heartbeat,
            args=(stop_heartbeat, image_path.name, branch, start),
            daemon=True,
        )
        heartbeat.start()
        try:
            with self._lock:
                with self._prediction_timeout(image_path.name):
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
        self._save_debug(
            image_path,
            result,
            elapsed,
            {**runtime, "blank_detection": blank_metrics},
        )
        if not markdown:
            raise RuntimeError(
                "PaddleOCR-VL 返回了空 Markdown，但图片中检测到了明显墨迹；"
                "原始结果已保存，未把它当作正常空白吞掉"
            )
        print(
            f"[本地 VL/{branch}] {image_path.name}：{elapsed:.2f}s，"
            f"Markdown {len(markdown)} 字符",
            flush=True,
        )
        return markdown

    def long_pack_cache_model(self, image_path: str | Path) -> str:
        """只有会临时切割的请求把切割版本加入 tile 缓存键。"""

        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if needs_local_long_split(
            width,
            height,
            self.max_pixels,
            trigger_height=self.long_split_trigger_height,
            minimum_estimated_width=self.long_split_minimum_model_width,
        ):
            return f"{self.model};{SPLIT_VERSION}"
        return self.model

    @staticmethod
    def postprocess_long_pack(markdown: str, pack: Any) -> str:
        """缓存命中和新识别统一恢复 H1～H5；目录项本身不升级。"""

        return restore_local_markdown_headings(markdown, pack)

    def recognize_long_pack(
        self,
        image_path: str | Path,
        prompt: str,
        pack: Any,
        image_manifest: dict[str, Any],
        context_gap: int,
    ) -> str:
        """对极端长宽比请求做本地临时切割；正式请求包和原图坐标保持不变。"""

        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if not needs_local_long_split(
            width,
            height,
            self.max_pixels,
            trigger_height=self.long_split_trigger_height,
            minimum_estimated_width=self.long_split_minimum_model_width,
        ):
            return self.recognize(image_path, prompt)

        header_height, header_ids = leading_header_height(
            pack,
            image_manifest,
            context_gap,
            height,
        )
        split_root = (
            image_path.parent.parent
            / "local_vl_parts"
            / image_path.stem
        )
        parts = create_local_long_parts(
            image_path,
            split_root,
            header_height=header_height,
            target_height=self.long_subblock_target_height,
            maximum_height=self.long_subblock_max_height,
            minimum_content_height=self.long_subblock_min_content_height,
            search_radius=self.long_subblock_search_radius,
            fallback_overlap=self.long_subblock_fallback_overlap,
            sample_width=self.long_subblock_sample_width,
            white_threshold=self.long_subblock_white_threshold,
            blank_ratio=self.long_subblock_blank_ratio,
            minimum_blank_height=self.long_subblock_minimum_blank_height,
            split_columns=str(getattr(pack, "semantic_role", "")).startswith(
                "table_of_contents"
            ),
        )
        if not parts:
            return self.recognize(image_path, prompt)

        estimated_width = estimated_model_width(width, height, self.max_pixels)
        plan = {
            "version": SPLIT_VERSION,
            "source_image": str(image_path.resolve()),
            "source_width": width,
            "source_height": height,
            "estimated_model_width_without_split": round(estimated_width, 2),
            "header_height": header_height,
            "header_heading_ids": list(header_ids),
            "semantic_role": str(getattr(pack, "semantic_role", "unknown")),
            "part_count": len(parts),
            "column_count": max(item.column_count for item in parts),
            "parts": [item.to_dict() for item in parts],
        }
        plan_path = split_root / "plan.json"
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[本地 VL/长图临时切割] {image_path.name}："
            f"{width}x{height}，预计模型宽度仅 {estimated_width:.0f}px，"
            f"拆为 {len(parts)} 块；每块复制 {header_height}px 标题头",
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
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    reusable = (
                        metadata.get("image_sha256") == image_sha256
                        and metadata.get("model") == self.model
                        and metadata.get("split_version") == SPLIT_VERSION
                    )
                except (OSError, json.JSONDecodeError):
                    reusable = False

            if reusable:
                markdown = response_path.read_text(encoding="utf-8")
                source = "子块缓存"
            else:
                source = "本地模型"
                print(
                    f"[本地 VL/长图子块 {index:02d}/{len(parts):02d}] "
                    f"{part.file_name}：开始识别",
                    flush=True,
                )
                markdown = self.recognize(part_path, prompt)
                response_path.write_text(markdown, encoding="utf-8")
                metadata_path.write_text(
                    json.dumps(
                        {
                            "image_sha256": image_sha256,
                            "model": self.model,
                            "split_version": SPLIT_VERSION,
                            "part": part.to_dict(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            markdowns.append(markdown)
            print(
                f"[本地 VL/长图子块 {index:02d}/{len(parts):02d}] "
                f"{part.file_name}：{source}完成，Markdown {len(markdown)} 字符",
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
