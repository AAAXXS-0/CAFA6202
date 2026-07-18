"""长图分支端到端编排。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ..common.cache import ResultCache
from ..common.hashing import discover_images, group_exact_duplicates
from ..common.image_backend import ImageBackend, create_backend
from .config import LongConfig
from .步骤003_滑窗与YOLO检测 import (
    GeneralYoloDetector,
    LongLayoutDetector,
    plan_detection_windows,
)
from .步骤002_图片读写与裁切 import (
    save_many_crops,
    save_recognition_pack_images,
)
from .步骤004_语义标题分析 import (
    analyze_semantic_headings,
    save_semantic_audit_windows,
)
from .步骤005_大模型请求打包 import (
    RecognitionPack,
    build_adaptive_recognition_packs,
    build_pack_prompt,
    build_semantic_recognition_packs,
    normalize_markdown_heading_levels,
    strip_repeated_context_headings,
)
from .步骤004_自适应安全切块 import (
    build_adaptive_chunks,
    build_row_ink_projection,
)
from ..common.models import Box
from ..common.submission import write_submission
from ..common.vlm_client import (
    CHAT_PROTOCOL,
    FinixDocClient,
    select_request_prompt,
)


LONG_PROMPT_VERSION = "long-markdown-v7-firered-heading-repair"


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def merge_markdown_overlap(left: str, right: str, max_chars: int = 1600) -> str:
    """只在相邻物理块接缝处做后缀/前缀去重。"""

    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    max_overlap = min(len(left), len(right), max_chars)
    for length in range(max_overlap, 19, -1):
        if left[-length:] == right[:length]:
            return left + right[length:]

    left_lines = left.splitlines()
    right_lines = right.splitlines()
    max_lines = min(len(left_lines), len(right_lines), 20)
    normalize = lambda text: re.sub(r"\s+", "", text)
    for count in range(max_lines, 0, -1):
        if [normalize(line) for line in left_lines[-count:]] == [
            normalize(line) for line in right_lines[:count]
        ]:
            return "\n".join(left_lines + right_lines[count:]).strip()
    return left + "\n" + right


class LongPipeline:
    """显式处理赛题长图目录，不包含任何自动路由逻辑。"""

    def __init__(
        self,
        config: LongConfig,
        work_dir: str | Path,
        detector: LongLayoutDetector | None = None,
    ) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.backend: ImageBackend = create_backend(config.backend)
        self.detector = detector
        self.cache = ResultCache(self.work_dir / "cache.sqlite3")

    def _prepare_detector(self) -> LongLayoutDetector:
        if self.detector is None:
            self.detector = GeneralYoloDetector(self.config)
        return self.detector

    def _find_reusable_manifest(
        self, manifest_path: Path, image_sha256: str
    ) -> Path | None:
        """确认单图预处理产物完整，可安全用于断点续跑。"""

        if not manifest_path.is_file():
            return None
        try:
            manifest = _load_json(manifest_path)
            if manifest.get("schema_version") != 3:
                return None
            if manifest.get("config") != self.config.to_dict():
                return None
            if manifest.get("image", {}).get("sha256") != image_sha256:
                return None
            request_packs = manifest.get("request_packs")
            if not isinstance(request_packs, list) or not request_packs:
                return None
            request_dir = manifest_path.parent / "vlm_requests"
            for pack in request_packs:
                file_name = pack.get("file_name") if isinstance(pack, dict) else None
                if not file_name or not (request_dir / file_name).is_file():
                    return None
        except (OSError, ValueError, TypeError, KeyError):
            # manifest 损坏或上次只生成了一半时，直接重新预处理。
            return None
        return manifest_path

    def _prepare_one(self, image_path: Path, image_sha256: str) -> Path:
        image_dir = self.work_dir / "prepared" / f"{image_path.stem}_{image_sha256[:12]}"
        manifest_path = image_dir / "manifest.json"
        reusable = self._find_reusable_manifest(manifest_path, image_sha256)
        if reusable is not None:
            print(f"[长图准备复用] {image_path.name}：请求切块已完整生成", flush=True)
            return reusable

        window_dir = image_dir / "detection_windows"
        request_dir = image_dir / "vlm_requests"
        audit_dir = image_dir / "semantic_audit"
        image_dir.mkdir(parents=True, exist_ok=True)

        meta = self.backend.read_meta(image_path, known_sha256=image_sha256)
        windows = plan_detection_windows(meta.height, self.config)
        window_requests = [
            (
                Box(0, window.start_y, meta.width, window.end_y),
                window_dir / window.file_name,
                1.0,
            )
            for window in windows
        ]
        save_many_crops(image_path, window_requests, self.backend)
        window_paths = [window_dir / window.file_name for window in windows]

        detector = self._prepare_detector()
        blocks = detector.detect(window_paths, windows, meta.width, meta.height)
        projection = build_row_ink_projection(
            window_paths,
            windows,
            meta.height,
            sample_width=self.config.projection_sample_width,
            white_threshold=self.config.projection_white_threshold,
        )
        # legacy 结果始终计算并写入清单：既是显式回退，也是新旧流程对照基线。
        legacy_chunks, legacy_debug = build_adaptive_chunks(
            meta.width,
            meta.height,
            projection,
            blocks,
            self.config,
        )
        semantic_headings: list[Any] = []
        heading_debug: dict[str, Any] | None = None
        semantic_debug: dict[str, Any] | None = None
        if self.config.strategy == "semantic":
            semantic_headings, evidence, heading_debug = analyze_semantic_headings(
                blocks,
                meta.width,
                window_paths,
                windows,
                self.config,
            )
            request_packs, semantic_debug = build_semantic_recognition_packs(
                semantic_headings,
                blocks,
                projection,
                meta.width,
                meta.height,
                self.config,
            )
            _dump_json(audit_dir / "006_标题样式聚类.json", heading_debug)
            _dump_json(audit_dir / "008_请求切块清单.json", semantic_debug)
            if self.config.semantic_audit_windows:
                save_semantic_audit_windows(
                    window_paths,
                    windows,
                    evidence,
                    semantic_headings,
                    heading_debug,
                    audit_dir,
                )
        else:
            request_packs = build_adaptive_recognition_packs(
                legacy_chunks,
                meta.width,
            )

        save_recognition_pack_images(
            image_path,
            request_packs,
            request_dir,
            self.backend,
            context_gap=self.config.semantic_context_gap,
            maximum_height=self.config.max_vlm_height,
        )

        manifest = {
            "schema_version": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": meta.to_dict(),
            "backend": self.backend.name,
            "detector": detector.name,
            "strategy": self.config.strategy,
            "config": self.config.to_dict(),
            "yolo_raw": (
                {
                    "directory": "yolo_raw",
                    "predictions": "yolo_raw/predictions.json",
                }
                if self.config.save_yolo_debug
                else None
            ),
            "windows": [window.to_dict() for window in windows],
            "layout_blocks": [block.to_dict() for block in blocks],
            "adaptive_cutting": legacy_debug,
            "safe_chunks": [chunk.to_dict() for chunk in legacy_chunks],
            "semantic_headings": [item.to_dict() for item in semantic_headings],
            "semantic_analysis": heading_debug,
            "semantic_cutting": semantic_debug,
            "request_packs": [pack.to_dict() for pack in request_packs],
        }
        _dump_json(manifest_path, manifest)
        return manifest_path

    def prepare_directory(self, input_dir: str | Path) -> Path:
        paths = discover_images(input_dir)
        if not paths:
            raise RuntimeError(f"长图目录中没有图片：{input_dir}")
        groups = group_exact_duplicates(paths)
        items: list[dict[str, Any]] = []
        sorted_groups = sorted(groups.items(), key=lambda pair: pair[1][0].name)
        for index, (image_sha256, group) in enumerate(sorted_groups, start=1):
            canonical = sorted(group, key=lambda path: path.name)[0]
            print(
                f"[长图准备 {index:02d}/{len(sorted_groups):02d}] "
                f"{canonical.name}（同内容文件 {len(group)} 张）",
                flush=True,
            )
            manifest_path = self._prepare_one(canonical, image_sha256)
            for path in sorted(group, key=lambda item: item.name):
                items.append(
                    {
                        "file_name": path.name,
                        "path": str(path.resolve()),
                        "sha256": image_sha256,
                        "canonical_file_name": canonical.name,
                        "duplicate_of": None if path == canonical else canonical.name,
                        "image_manifest": str(manifest_path.resolve()),
                    }
                )

        dataset_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(Path(input_dir).resolve()),
            "config": self.config.to_dict(),
            "config_digest": self.config.digest(),
            "image_count": len(paths),
            "unique_image_count": len(groups),
            "duplicate_reuse_count": len(paths) - len(groups),
            "items": sorted(items, key=lambda item: item["file_name"]),
        }
        output_path = self.work_dir / "dataset_manifest.json"
        _dump_json(output_path, dataset_manifest)
        return output_path

    def _recognize_manifest(self, manifest_path: Path, client: FinixDocClient) -> str:
        image_manifest = _load_json(manifest_path)
        image_info = image_manifest["image"]
        image_name = str(
            image_info.get("file_name") or Path(image_info["path"]).name
        )
        current = ""
        seen_heading_text: dict[str, str] = {}
        response_dir = manifest_path.parent / "responses"
        response_dir.mkdir(parents=True, exist_ok=True)
        raw_packs = image_manifest["request_packs"]
        for index, raw_pack in enumerate(raw_packs, start=1):
            pack = RecognitionPack.from_dict(raw_pack)
            crop_path = manifest_path.parent / "vlm_requests" / pack.file_name
            request_label = (
                f"原图 {image_name} / 切块 {pack.file_name}"
            )
            # 官方 multipart 文档没有 prompt 字段：官方模式连提示词都不生成。
            # 只有本地 Chat Completions 兼容服务才使用自定义提示词。
            prompt = select_request_prompt(
                getattr(client, "protocol", CHAT_PROTOCOL),
                lambda: build_pack_prompt(pack),
            )
            image_bytes = crop_path.read_bytes()
            cache_model = client.model
            long_pack_cache_model = getattr(client, "long_pack_cache_model", None)
            if callable(long_pack_cache_model):
                cache_model = long_pack_cache_model(crop_path)
            cache_key = self.cache.tile_key(image_bytes, prompt, cache_model)
            markdown = self.cache.get_tile(cache_key)
            source = "缓存"
            if markdown is None:
                source = "API"
                print(
                    f"[长图识别 {index:02d}/{len(raw_packs):02d}] "
                    f"{request_label}：请求 API",
                    flush=True,
                )
                recognize_long_pack = getattr(client, "recognize_long_pack", None)
                if callable(recognize_long_pack):
                    markdown = recognize_long_pack(
                        crop_path,
                        prompt,
                        pack,
                        image_manifest,
                        self.config.semantic_context_gap,
                    )
                else:
                    if isinstance(client, FinixDocClient):
                        markdown = client.recognize(
                            crop_path,
                            prompt,
                            request_label=request_label,
                        )
                    else:
                        markdown = client.recognize(crop_path, prompt)
                self.cache.put_tile(
                    cache_key,
                    markdown,
                    {
                        "source_image": image_name,
                        "pack": pack.to_dict(),
                        "model": cache_model,
                        "prompt_version": LONG_PROMPT_VERSION,
                    },
                )
            postprocess_long_pack = getattr(client, "postprocess_long_pack", None)
            if callable(postprocess_long_pack):
                markdown = postprocess_long_pack(markdown, pack)
            (response_dir / f"{pack.id}.md").write_text(markdown, encoding="utf-8")
            cleaned = strip_repeated_context_headings(
                markdown,
                pack,
                seen_heading_text,
            )
            (response_dir / f"{pack.id}_聚合输入.md").write_text(
                cleaned,
                encoding="utf-8",
            )
            print(
                f"[长图识别 {index:02d}/{len(raw_packs):02d}] "
                f"{request_label}：{source}完成，Markdown {len(markdown)} 字符",
                flush=True,
            )
            if not current:
                current = cleaned.strip()
            elif pack.overlap_top != 0:
                current = merge_markdown_overlap(current, cleaned)
            else:
                current = current.rstrip() + "\n\n" + cleaned.lstrip()
        return normalize_markdown_heading_levels(current.strip())

    def recognize_dataset(
        self,
        dataset_manifest_path: str | Path,
        client: FinixDocClient,
        output_csv: str | Path,
        max_workers: int = 1,
    ) -> dict[str, str]:
        dataset_manifest = _load_json(Path(dataset_manifest_path))
        recognition_digest = hashlib.sha256(
            (
                dataset_manifest["config_digest"]
                + "\0"
                + client.model
                + "\0"
                + LONG_PROMPT_VERSION
            ).encode("utf-8")
        ).hexdigest()
        if max_workers <= 0:
            raise ValueError("max_workers 必须大于 0")

        unique_items: dict[str, dict[str, Any]] = {}
        aliases: dict[str, list[str]] = {}
        for item in dataset_manifest["items"]:
            unique_items.setdefault(item["canonical_file_name"], item)
            aliases.setdefault(item["canonical_file_name"], []).append(
                item["file_name"]
            )

        def recognize_one(item: dict[str, Any]) -> tuple[str, str]:
            canonical_name = item["canonical_file_name"]
            markdown = self.cache.get_image(item["sha256"], recognition_digest)
            if markdown is None:
                markdown = self._recognize_manifest(
                    Path(item["image_manifest"]), client
                )
                self.cache.put_image(
                    item["sha256"],
                    recognition_digest,
                    markdown,
                    {
                        "canonical_file_name": canonical_name,
                        "model": client.model,
                        "prompt_version": LONG_PROMPT_VERSION,
                    },
                )
            return canonical_name, markdown

        def recognize_safely(
            item: dict[str, Any],
        ) -> tuple[str, str | None, dict[str, Any] | None]:
            """单张图失败只隔离该图，其他图片继续识别并积累缓存。"""

            canonical_name = item["canonical_file_name"]
            try:
                name, markdown = recognize_one(item)
                return name, markdown, None
            except Exception as error:
                failure = {
                    "canonical_file_name": canonical_name,
                    "file_names": aliases[canonical_name],
                    "image_manifest": item["image_manifest"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print(
                    f"[长图失败] 原图 {canonical_name}：{error}；"
                    "未写整图缓存，继续处理其他图片。",
                    flush=True,
                )
                return canonical_name, None, failure

        canonical_items = list(unique_items.values())
        if max_workers == 1 or len(canonical_items) <= 1:
            outcomes = [recognize_safely(item) for item in canonical_items]
        else:
            worker_count = min(max_workers, len(canonical_items))
            print(
                f"[长图并行] {worker_count} 个任务并发识别 "
                f"{len(canonical_items)} 张唯一图片",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                outcomes = list(executor.map(recognize_safely, canonical_items))
        canonical_results = {
            name: markdown
            for name, markdown, _ in outcomes
            if markdown is not None
        }
        failures = [
            failure for _, _, failure in outcomes if failure is not None
        ]

        results = {
            item["file_name"]: canonical_results[item["canonical_file_name"]]
            for item in dataset_manifest["items"]
            if item["canonical_file_name"] in canonical_results
        }
        result_dir = self.work_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        for file_name, markdown in results.items():
            result_path = result_dir / f"{Path(file_name).stem}.md"
            result_path.write_text(markdown, encoding="utf-8")

        failure_report_path = self.work_dir / "recognition_failures.json"
        _dump_json(
            failure_report_path,
            {
                "status": "incomplete" if failures else "ok",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "failure_count": len(failures),
                "successful_unique_images": len(canonical_results),
                "failures": failures,
            },
        )
        if failures:
            partial_csv = self.work_dir / "partial_results.csv"
            write_submission(results, partial_csv)
            failed_names = ", ".join(
                failure["canonical_file_name"]
                for failure in failures
            )
            raise RuntimeError(
                f"长图识别跑完其他图片后仍有 {len(failures)} 张失败；"
                f"失败原图：{failed_names}；详情：{failure_report_path}"
            )
        write_submission(results, output_csv)
        return results
