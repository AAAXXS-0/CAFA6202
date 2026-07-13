"""图表分支端到端编排。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import ImageDraw

from .cache import ResultCache
from .config import TableConfig
from .detectors import (
    ProjectionTableDetector,
    create_detector,
    suppress_duplicate_boxes,
)
from .hashing import discover_images, group_exact_duplicates
from .image_backend import ImageBackend, create_backend
from .markdown_merge import MarkdownMergeError, merge_markdown_grid
from .models import Box, DetectedBox, ImageMeta, PreparedRegion, TilePlan
from .submission import write_submission
from .tiling import plan_region_tiles
from .vlm_client import FinixDocClient


PROMPT_VERSION = "table-markdown-v1"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_table_prompt(tile: TilePlan) -> str:
    """让每个切片的输出尽量适合确定性矩阵合并。"""

    return f"""你正在解析金融文档中的表格切片。

切片位置：表格区域 {tile.region_index + 1}，纵向第 {tile.row_index + 1}/{tile.row_count} 块，横向第 {tile.column_index + 1}/{tile.column_count} 块。

要求：
1. 只识别图片中真实可见的内容，严禁补全被裁掉的行、列或文字。
2. 表格必须输出为标准 Markdown 表格；表题、单位、注释放在表格前后。
3. 保持原始行列顺序、数字、小数点、百分号、括号和特殊符号。
4. 遇到空单元格保留为空，不要自行填写“无”“-”或其他内容。
5. 不要输出解释、分析、置信度、HTML 或 Markdown 代码围栏。
6. 若边缘出现与相邻切片重复的行列，仍按图片如实输出，程序会根据重叠内容去重。

直接输出 Markdown："""


class TablePipeline:
    """只处理已经由赛题分好的图表目录，不承担自动路由。"""

    def __init__(self, config: TableConfig, work_dir: str | Path):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.backend: ImageBackend = create_backend(config.backend)
        self.detector = create_detector(config)
        self.cache = ResultCache(self.work_dir / "cache.sqlite3")

    @staticmethod
    def _map_preview_box(box: Box, preview_width: int, preview_height: int, meta: ImageMeta) -> Box:
        """把预览图坐标精确映射回原图坐标。"""

        scale_x = meta.width / preview_width
        scale_y = meta.height / preview_height
        return Box(
            round(box.x1 * scale_x),
            round(box.y1 * scale_y),
            round(box.x2 * scale_x),
            round(box.y2 * scale_y),
        ).clamp(meta.width, meta.height)

    def _detect_regions(self, preview, meta: ImageMeta) -> list[DetectedBox]:
        preview_boxes = self.detector.detect(preview)
        # YOLO 在极端表格上可能无框；此时用投影检测兜底，而不是直接丢图。
        if not preview_boxes and self.detector.name != "projection":
            preview_boxes = ProjectionTableDetector(self.config).detect(preview)

        mapped: list[DetectedBox] = []
        padding_x = max(4, round(meta.width * self.config.table_box_padding_ratio))
        padding_y = max(4, round(meta.height * self.config.table_box_padding_ratio))
        for item in preview_boxes:
            source_box = self._map_preview_box(
                item.box, preview.width, preview.height, meta
            ).expand(padding_x, padding_y, meta.width, meta.height)
            if source_box.width <= 0 or source_box.height <= 0:
                continue
            mapped.append(
                DetectedBox(
                    source_box,
                    label=item.label,
                    confidence=item.confidence,
                    source=item.source,
                )
            )
        return suppress_duplicate_boxes(mapped)

    @staticmethod
    def _draw_preview_boxes(preview, boxes: list[DetectedBox], meta: ImageMeta):
        overlay = preview.copy()
        draw = ImageDraw.Draw(overlay)
        scale_x = preview.width / meta.width
        scale_y = preview.height / meta.height
        for index, item in enumerate(boxes):
            box = item.box
            coords = (
                round(box.x1 * scale_x),
                round(box.y1 * scale_y),
                round(box.x2 * scale_x),
                round(box.y2 * scale_y),
            )
            draw.rectangle(coords, outline=(255, 0, 0), width=3)
            draw.text((coords[0] + 4, coords[1] + 4), str(index + 1), fill=(255, 0, 0))
        return overlay

    def _prepare_one(self, image_path: Path, image_sha256: str) -> Path:
        image_dir = self.work_dir / "prepared" / f"{image_path.stem}_{image_sha256[:12]}"
        tiles_dir = image_dir / "tiles"
        image_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)

        meta = self.backend.read_meta(image_path, known_sha256=image_sha256)
        preview = self.backend.make_preview(image_path, self.config.preview_max_side)
        preview.save(image_dir / "preview.png", format="PNG", optimize=True)

        detected = self._detect_regions(preview, meta)
        if not detected:
            # 两套检测都失败时，保守地把整图作为一个区域，保证不漏文件。
            detected = [
                DetectedBox(
                    Box(0, 0, meta.width, meta.height),
                    confidence=0.0,
                    source="whole-image-fallback",
                )
            ]
        self._draw_preview_boxes(preview, detected, meta).save(
            image_dir / "preview_detected.png", format="PNG", optimize=True
        )

        regions: list[PreparedRegion] = []
        for region_index, item in enumerate(detected):
            plans = plan_region_tiles(
                item.box,
                region_index=region_index,
                max_side=self.config.max_vlm_side,
                overlap=self.config.tile_overlap,
                single_tile_min_scale=self.config.single_tile_min_scale,
            )
            for plan in plans:
                self.backend.save_crop(
                    image_path,
                    plan.source_box,
                    tiles_dir / plan.file_name,
                    scale=plan.scale,
                )
            regions.append(
                PreparedRegion(
                    index=region_index,
                    box=item.box,
                    detector_source=item.source,
                    tiles=plans,
                )
            )

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": meta.to_dict(),
            "backend": self.backend.name,
            "detector": self.detector.name,
            "config": self.config.to_dict(),
            "regions": [region.to_dict() for region in regions],
        }
        manifest_path = image_dir / "manifest.json"
        _json_dump(manifest_path, manifest)
        return manifest_path

    def prepare_directory(self, input_dir: str | Path) -> Path:
        """切分图表目录并生成可审计的数据集清单。"""

        paths = discover_images(input_dir)
        if not paths:
            raise RuntimeError(f"图表目录中没有图片：{input_dir}")
        groups = group_exact_duplicates(paths)
        items: list[dict[str, Any]] = []
        sorted_groups = sorted(groups.items(), key=lambda pair: pair[1][0].name)
        for group_index, (image_sha256, group) in enumerate(sorted_groups, start=1):
            canonical = sorted(group, key=lambda path: path.name)[0]
            print(
                f"[准备 {group_index:02d}/{len(sorted_groups):02d}] "
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
        _json_dump(output_path, dataset_manifest)
        return output_path

    @staticmethod
    def _tile_from_dict(raw: dict[str, Any]) -> TilePlan:
        return TilePlan(
            region_index=int(raw["region_index"]),
            row_index=int(raw["row_index"]),
            column_index=int(raw["column_index"]),
            row_count=int(raw["row_count"]),
            column_count=int(raw["column_count"]),
            source_box=Box.from_dict(raw["source_box"]),
            output_width=int(raw["output_width"]),
            output_height=int(raw["output_height"]),
            scale=float(raw["scale"]),
            file_name=str(raw["file_name"]),
        )

    def _recognize_manifest(self, manifest_path: Path, client: FinixDocClient) -> str:
        image_manifest = _load_json(manifest_path)
        region_markdowns: list[str] = []
        for region in image_manifest["regions"]:
            contents: dict[tuple[int, int], str] = {}
            response_dir = manifest_path.parent / "responses"
            response_dir.mkdir(parents=True, exist_ok=True)
            for raw_tile in region["tiles"]:
                tile = self._tile_from_dict(raw_tile)
                tile_path = manifest_path.parent / "tiles" / tile.file_name
                prompt = build_table_prompt(tile)
                image_bytes = tile_path.read_bytes()
                cache_key = self.cache.tile_key(image_bytes, prompt, client.model)
                markdown = self.cache.get_tile(cache_key)
                if markdown is None:
                    markdown = client.recognize(tile_path, prompt)
                    self.cache.put_tile(
                        cache_key,
                        markdown,
                        {
                            "tile": tile.to_dict(),
                            "model": client.model,
                            "prompt_version": PROMPT_VERSION,
                        },
                    )
                (response_dir / f"{Path(tile.file_name).stem}.md").write_text(
                    markdown, encoding="utf-8"
                )
                contents[(tile.row_index, tile.column_index)] = markdown

            try:
                region_markdown = (
                    next(iter(contents.values()))
                    if len(contents) == 1
                    else merge_markdown_grid(contents)
                )
            except MarkdownMergeError as error:
                _json_dump(
                    manifest_path.parent / "merge_error.json",
                    {
                        "region_index": region["index"],
                        "error": str(error),
                        "responses": {
                            f"r{row}_c{column}": text
                            for (row, column), text in contents.items()
                        },
                    },
                )
                raise RuntimeError(
                    f"{manifest_path.parent.name} 的表格切片无法可靠合并；原始响应已保留"
                ) from error
            region_markdowns.append(region_markdown.strip())
        return "\n\n".join(text for text in region_markdowns if text).strip()

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
                + PROMPT_VERSION
            ).encode("utf-8")
        ).hexdigest()

        canonical_results: dict[str, str] = {}
        for item in dataset_manifest["items"]:
            canonical_name = item["canonical_file_name"]
            if canonical_name in canonical_results:
                continue
            cached = self.cache.get_image(item["sha256"], recognition_digest)
            if cached is None:
                cached = self._recognize_manifest(Path(item["image_manifest"]), client)
                self.cache.put_image(
                    item["sha256"],
                    recognition_digest,
                    cached,
                    {
                        "canonical_file_name": canonical_name,
                        "model": client.model,
                        "prompt_version": PROMPT_VERSION,
                    },
                )
            canonical_results[canonical_name] = cached

        results = {
            item["file_name"]: canonical_results[item["canonical_file_name"]]
            for item in dataset_manifest["items"]
        }
        results_dir = self.work_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for file_name, markdown in results.items():
            (results_dir / f"{Path(file_name).stem}.md").write_text(markdown, encoding="utf-8")
        write_submission(results, output_csv)
        return results
