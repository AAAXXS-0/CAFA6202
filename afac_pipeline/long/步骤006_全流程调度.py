"""长图分支端到端编排。"""

from __future__ import annotations

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
from .步骤003_滑窗与YOLO检测 import GeneralYoloDetector, LongLayoutDetector, plan_detection_windows
from .步骤002_图片读写与裁切 import save_many_crops
from .步骤005_大模型请求打包 import RecognitionPack, build_pack_prompt, build_recognition_packs
from .步骤001_数据定义 import SemanticPart
from .步骤004_标题层级与二次分块 import (
    attach_physical_parts,
    build_semantic_segments,
    infer_heading_hierarchy,
)
from ..common.models import Box
from ..common.submission import write_submission
from ..common.vlm_client import FinixDocClient


LONG_PROMPT_VERSION = "long-markdown-v2-packed"


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_long_prompt(part: SemanticPart) -> str:
    role_instructions = {
        "front_matter": "这是文档前置信息，保留公司名、产品名、版本号和所有可见文字。",
        "toc": "这是目录区域，严格保持目录项的先后顺序、编号和页码，不要补全不可见目录项。",
        "body_intro": "这是正文开头，保留正文一级标题及其后的全部可见内容。",
        "h2_intro": "这是二级标题及其直属正文，二级标题使用 ##。",
        "h2_body": "这是一个二级标题完整章节，二级标题使用 ##。",
        "h3_body": "这是三级标题章节；若图中同时出现父级标题，父级使用 ##，子级使用 ###。",
        "body": "这是正文内容，按图片中的标题和段落结构输出。",
    }
    levels = "、".join(f"H{level}" for level in part.expected_heading_levels) or "无强制标题"
    return f"""请把这张金融文档长图切片转换为与原图严格对应的 Markdown。

区域类型：{part.role}
逻辑段：{part.segment_id}
物理分块：{part.part_index + 1}/{part.part_count}
当前切片预期首次出现的标题层级：{levels}
{role_instructions.get(part.role, role_instructions['body'])}

要求：
1. 只输出图片中真实可见的内容，严禁补全、总结、解释或改写。
2. 保持文字、数字、编号、标点、特殊符号、列表和表格的原始顺序。
3. 标题层级遵循提示；普通加粗文字不要擅自提升为标题。
4. 如果切片从段落中间开始或结束，只抄录可见部分，不猜测缺失内容。
5. 不要输出 Markdown 代码围栏、识别说明、页码统计或置信度。
6. 相邻物理块包含少量重叠内容时照实输出，程序会在接缝处去重。

直接输出 Markdown："""


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

    # Markdown 模型有时只改变接缝空白；按行去除空白后再比较最近若干行。
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
        # run-long 只读取已经准备好的裁片，不应强制安装或加载 YOLO。
        # 首次进入 prepare-long 时再创建检测器，并在后续图片间复用模型实例。
        self.detector = detector
        self.cache = ResultCache(self.work_dir / "cache.sqlite3")

    def _prepare_detector(self) -> LongLayoutDetector:
        if self.detector is None:
            self.detector = GeneralYoloDetector(self.config)
        return self.detector

    def _prepare_one(self, image_path: Path, image_sha256: str) -> Path:
        image_dir = self.work_dir / "prepared" / f"{image_path.stem}_{image_sha256[:12]}"
        window_dir = image_dir / "detection_windows"
        semantic_dir = image_dir / "semantic_crops"
        request_dir = image_dir / "vlm_requests"
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
        logical_titles, headings = infer_heading_hierarchy(blocks, meta.width, self.config)
        segments = build_semantic_segments(meta.height, blocks, headings)
        segments = attach_physical_parts(segments, blocks, meta.width, self.config)
        request_packs = build_recognition_packs(
            segments, meta.width, self.config.max_vlm_height
        )

        crop_requests = [
            (part.source_box, semantic_dir / part.file_name, 1.0)
            for segment in segments
            for part in segment.parts
        ]
        save_many_crops(image_path, crop_requests, self.backend)
        request_crops = [
            (pack.source_box, request_dir / pack.file_name, 1.0)
            for pack in request_packs
        ]
        save_many_crops(image_path, request_crops, self.backend)

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": meta.to_dict(),
            "backend": self.backend.name,
            "detector": detector.name,
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
            "logical_titles": [title.to_dict() for title in logical_titles],
            "headings": [heading.to_dict() for heading in headings],
            "segments": [segment.to_dict() for segment in segments],
            "request_packs": [pack.to_dict() for pack in request_packs],
        }
        manifest_path = image_dir / "manifest.json"
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

    @staticmethod
    def _part_from_dict(raw: dict[str, Any]) -> SemanticPart:
        return SemanticPart(
            id=str(raw["id"]),
            segment_id=str(raw["segment_id"]),
            role=str(raw["role"]),
            source_box=Box.from_dict(raw["source_box"]),
            part_index=int(raw["part_index"]),
            part_count=int(raw["part_count"]),
            h1_id=raw.get("h1_id"),
            h2_id=raw.get("h2_id"),
            h3_id=raw.get("h3_id"),
            expected_heading_levels=tuple(int(value) for value in raw["expected_heading_levels"]),
            file_name=str(raw["file_name"]),
        )

    def _recognize_manifest(self, manifest_path: Path, client: FinixDocClient) -> str:
        image_manifest = _load_json(manifest_path)
        current = ""
        response_dir = manifest_path.parent / "responses"
        response_dir.mkdir(parents=True, exist_ok=True)
        raw_packs = image_manifest["request_packs"]
        for index, raw_pack in enumerate(raw_packs, start=1):
            pack = RecognitionPack.from_dict(raw_pack)
            crop_path = manifest_path.parent / "vlm_requests" / pack.file_name
            prompt = build_pack_prompt(pack)
            image_bytes = crop_path.read_bytes()
            cache_key = self.cache.tile_key(image_bytes, prompt, client.model)
            markdown = self.cache.get_tile(cache_key)
            source = "缓存"
            if markdown is None:
                source = "API"
                print(
                    f"[长图识别 {index:02d}/{len(raw_packs):02d}] "
                    f"{pack.file_name}：请求 API",
                    flush=True,
                )
                markdown = client.recognize(crop_path, prompt)
                self.cache.put_tile(
                    cache_key,
                    markdown,
                    {
                        "pack": pack.to_dict(),
                        "model": client.model,
                        "prompt_version": LONG_PROMPT_VERSION,
                    },
                )
            (response_dir / f"{pack.id}.md").write_text(markdown, encoding="utf-8")
            print(
                f"[长图识别 {index:02d}/{len(raw_packs):02d}] "
                f"{pack.file_name}：{source}完成，Markdown {len(markdown)} 字符",
                flush=True,
            )
            current = markdown.strip() if not current else merge_markdown_overlap(current, markdown)
        return current.strip()

    def recognize_dataset(
        self,
        dataset_manifest_path: str | Path,
        client: FinixDocClient,
        output_csv: str | Path,
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
        canonical_results: dict[str, str] = {}
        for item in dataset_manifest["items"]:
            canonical_name = item["canonical_file_name"]
            if canonical_name in canonical_results:
                continue
            markdown = self.cache.get_image(item["sha256"], recognition_digest)
            if markdown is None:
                markdown = self._recognize_manifest(Path(item["image_manifest"]), client)
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
            canonical_results[canonical_name] = markdown

        results = {
            item["file_name"]: canonical_results[item["canonical_file_name"]]
            for item in dataset_manifest["items"]
        }
        result_dir = self.work_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        for file_name, markdown in results.items():
            (result_dir / f"{Path(file_name).stem}.md").write_text(markdown, encoding="utf-8")
        write_submission(results, output_csv)
        return results
