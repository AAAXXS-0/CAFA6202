"""图表分支端到端编排。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory

from ..common.cache import ResultCache
from .config import TableConfig
from .步骤003_区域检测器入口 import (
    InkTableDetector,
    ProjectionTableDetector,
    create_detector,
    suppress_duplicate_boxes,
)
from ..common.hashing import discover_images, group_exact_duplicates
from ..common.image_backend import ImageBackend, create_backend
from .步骤004_网格与白带检测 import GridStructure, detect_grid_structure
from .步骤006_逻辑网格切块 import plan_grid_tiles
from .步骤009_HTML表格软对齐 import (
    HtmlTableMergeError,
    merge_logical_tiles,
    normalize_table_response,
    normalize_table_response_soft,
    render_empty_table,
)
from .步骤001_墨水密度定位 import density_visualization
from .步骤008_Markdown表格合并 import MarkdownMergeError, merge_markdown_grid
from .步骤005_黑线白带结构检测 import (
    V6RegionResult,
    detect_v6_grid,
    detect_v6_regions,
    detected_boxes,
)
from ..common.models import Box, DetectedBox, ImageMeta, PreparedRegion, TilePlan
from ..common.submission import write_submission
from .步骤007_像素重叠切块 import plan_region_tiles
from ..common.vlm_client import (
    CHAT_PROTOCOL,
    FinixDocClient,
    select_request_prompt,
)


PROMPT_VERSION = "table-structured-html-v4-empty-fallback"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _table_cache_model(client: FinixDocClient) -> str:
    """允许本地模型只让图表参数进入图表缓存，不牵连长图缓存。"""

    resolver = getattr(client, "table_cache_model", None)
    return str(resolver()) if callable(resolver) else str(client.model)


def _table_legacy_cache_models(client: FinixDocClient) -> tuple[str, ...]:
    resolver = getattr(client, "table_legacy_cache_models", None)
    if not callable(resolver):
        return ()
    return tuple(str(value) for value in resolver())


def build_table_prompt(tile: TilePlan) -> str:
    """让每个切片的输出尽量适合确定性矩阵合并。"""

    if tile.tiling_mode == "logical_grid":
        visible_rows = tile.header_context_rows + tile.logical_row_end - tile.logical_row_start
        visible_columns = tile.stub_context_columns + tile.logical_column_end - tile.logical_column_start
        return f"""你正在解析金融文档中的表格切片。

该图片应包含 {visible_rows} 个逻辑行、{visible_columns} 个逻辑列。顶部可能重复表头，左侧可能重复行名列，这些是有意保留的上下文。

要求：
1. 只识别图片中真实可见的内容，严禁补全被裁掉的行、列或文字。
2. 只输出一个 HTML <table>；用 <tr>、<th>、<td> 表示原始结构。
3. 合并单元格必须使用 rowspan/colspan，不能拆成虚构单元格。
4. 保持行列顺序、数字、小数点、百分号、括号、空单元格和特殊符号。
5. 不要输出分析、置信度、Markdown 表格或代码围栏。

直接输出 HTML："""

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


def _logical_tile_shape(tile: TilePlan) -> tuple[int, int]:
    """返回逻辑网格切片里实际可见的行数和列数。"""

    rows = tile.header_context_rows + tile.logical_row_end - tile.logical_row_start
    columns = (
        tile.stub_context_columns
        + tile.logical_column_end
        - tile.logical_column_start
    )
    return rows, columns


def _mask_has_no_text(ink_mask: list[list[bool]] | None) -> bool:
    """判断预处理是否已确认所有逻辑格内都没有文字。

    ``None`` 表示没有可靠的逻辑网格信息，不能擅自当作空表。
    """

    return ink_mask is not None and bool(ink_mask) and all(
        not has_text for row in ink_mask for has_text in row
    )


def _empty_tile_html(tile: TilePlan) -> str:
    """根据逻辑切片的预处理行列数生成全空表。"""

    if tile.tiling_mode != "logical_grid":
        raise ValueError("只有逻辑网格切片才能生成预处理全空表")
    return render_empty_table(*_logical_tile_shape(tile))


class TablePipeline:
    """只处理已经由赛题分好的图表目录，不承担自动路由。"""

    def __init__(self, config: TableConfig, work_dir: str | Path):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.backend: ImageBackend = create_backend(config.backend)
        self.detector = create_detector(config)
        self.cache = ResultCache(self.work_dir / "cache.sqlite3")
        self._last_v6_regions: V6RegionResult | None = None

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
        if isinstance(self.detector, InkTableDetector):
            # 默认无模型流程使用 v6。detect_v6_regions 返回的是 20% 分析图
            # 坐标；这里直接映射回原图，不再额外删除或扩张边缘。
            self._last_v6_regions = detect_v6_regions(preview, self.config)
            return [
                DetectedBox(
                    self._map_preview_box(item.box, preview.width, preview.height, meta),
                    label=item.label,
                    confidence=item.confidence,
                    source=item.source,
                )
                for item in detected_boxes(self._last_v6_regions)
            ]
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

    def _make_detection_preview(
        self, image_path: Path, meta: ImageMeta, image_dir: Path
    ) -> Image.Image:
        """默认 v6 使用固定 20% 分析图；历史检测器仍使用最长边预览。"""

        preview_path = image_dir / "preview.png"
        if isinstance(self.detector, InkTableDetector):
            longest = round(max(meta.width, meta.height) * self.config.table_analysis_scale)
            if longest > self.config.table_analysis_max_side:
                raise RuntimeError(
                    f"{image_path.name} 固定缩放后的最长边为 {longest}，超过安全上限 "
                    f"{self.config.table_analysis_max_side}；请显式调整 table_analysis_scale"
                )
            self.backend.save_crop(
                image_path,
                Box(0, 0, meta.width, meta.height),
                preview_path,
                scale=self.config.table_analysis_scale,
            )
            with Image.open(preview_path) as source:
                return source.convert("RGB").copy()
        preview = self.backend.make_preview(image_path, self.config.preview_max_side)
        preview.save(preview_path, format="PNG", optimize=True)
        return preview

    def _save_v6_detection_debug(self, preview: Image.Image, image_dir: Path) -> None:
        """保存密度分表带、分表框和最终分析框，方便定位分表错误。"""

        result = self._last_v6_regions
        if result is None:
            return
        output_dir = image_dir / "density_detection"
        output_dir.mkdir(parents=True, exist_ok=True)
        density = result.ink_result.coarse_density
        density_visualization(density).resize(
            preview.size, Image.Resampling.NEAREST
        ).save(output_dir / "density.png")
        overlay = preview.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        for band in result.horizontal_bands:
            y1 = round(band.start * preview.height / density.shape[0])
            y2 = round(band.end * preview.height / density.shape[0])
            draw.rectangle((0, y1, preview.width, y2), fill=(255, 0, 0, 65))
        for band in result.vertical_bands:
            x1 = round(band.start * preview.width / density.shape[1])
            x2 = round(band.end * preview.width / density.shape[1])
            draw.rectangle((x1, 0, x2, preview.height), fill=(255, 0, 0, 65))
        for index, box in enumerate(result.split_boxes):
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(0, 80, 255, 255), width=3)
            draw.text((box.x1 + 4, box.y1 + 4), f"split-{index + 1}", fill=(0, 80, 255, 255))
        for index, box in enumerate(result.analysis_boxes):
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(160, 0, 255, 255), width=3)
            draw.text((box.x1 + 4, box.y1 + 22), f"table-{index + 1}", fill=(160, 0, 255, 255))
        overlay.save(output_dir / "split_and_analysis_boxes.png")

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

    def _analyze_grid(
        self, image_path: Path, region: Box, region_index: int, image_dir: Path
    ) -> GridStructure:
        """从原图区域生成受控尺寸分析图，再把网格坐标映射回原图。"""

        analysis_dir = image_dir / "grid_analysis"
        analysis_path = analysis_dir / f"region_{region_index:03d}.png"
        if isinstance(self.detector, InkTableDetector):
            scale = self.config.table_analysis_scale
        else:
            scale = min(
                1.0, self.config.grid_analysis_max_side / max(region.width, region.height)
            )
        self.backend.save_crop(image_path, region, analysis_path, scale=scale)
        with Image.open(analysis_path) as source:
            analysis = source.convert("RGB").copy()
        diagnostics = None
        if isinstance(self.detector, InkTableDetector):
            grid, diagnostics = detect_v6_grid(analysis, region, self.config)
        else:
            grid = detect_grid_structure(analysis, region, self.config)
        overlay = analysis.copy()
        draw = ImageDraw.Draw(overlay)
        color = (0, 180, 0) if grid.source == "ruled-lines" else (255, 128, 0)
        for boundary in grid.row_boundaries[1:-1]:
            y = round((boundary - region.y1) * analysis.height / region.height)
            draw.line((0, y, analysis.width, y), fill=color, width=2)
        for boundary in grid.column_boundaries[1:-1]:
            x = round((boundary - region.x1) * analysis.width / region.width)
            draw.line((x, 0, x, analysis.height), fill=color, width=2)
        draw.text((8, 8), grid.source, fill=color)
        overlay.save(
            analysis_dir / f"region_{region_index:03d}_boundaries.png"
        )
        if diagnostics is not None:
            diagnostic_data = diagnostics.to_dict()
            diagnostic_data.update(
                {
                    "analysis_size": list(analysis.size),
                    "source_region": region.to_dict(),
                    "grid": grid.to_dict(),
                }
            )
            _json_dump(analysis_dir / f"region_{region_index:03d}_diagnostics.json", diagnostic_data)
        return grid


    def _save_tile(
        self, image_path: Path, output_path: Path, plan: TilePlan,
        row_boundaries: tuple[int, ...], column_boundaries: tuple[int, ...],
    ) -> None:
        """保存普通裁片，或拼出“左上角 + 表头 + 行名列 + 主体”图片。"""

        if plan.header_context_rows == 0 and plan.stub_context_columns == 0:
            self.backend.save_crop(image_path, plan.source_box, output_path, plan.scale)
            with Image.open(output_path) as source:
                tile = source.convert("RGB").copy()
            if tile.size != (plan.output_width, plan.output_height):
                tile = tile.resize(
                    (plan.output_width, plan.output_height),
                    Image.Resampling.LANCZOS,
                )
                tile.save(output_path, format="PNG", optimize=True)
            return
        with TemporaryDirectory(dir=output_path.parent) as temporary:
            temporary_dir = Path(temporary)

            def crop(name: str, box: Box) -> Image.Image:
                path = temporary_dir / f"{name}.png"
                self.backend.save_crop(image_path, box, path)
                with Image.open(path) as source:
                    return source.convert("RGB").copy()

            body = crop("body", plan.source_box)
            top_height = row_boundaries[plan.header_context_rows] - row_boundaries[0]
            left_width = column_boundaries[plan.stub_context_columns] - column_boundaries[0]
            canvas = Image.new("RGB", (body.width + left_width, body.height + top_height), "white")
            canvas.paste(body, (left_width, top_height))
            if top_height:
                top_box = Box(plan.source_box.x1, row_boundaries[0], plan.source_box.x2, row_boundaries[plan.header_context_rows])
                canvas.paste(crop("top", top_box), (left_width, 0))
            if left_width:
                left_box = Box(column_boundaries[0], plan.source_box.y1, column_boundaries[plan.stub_context_columns], plan.source_box.y2)
                canvas.paste(crop("left", left_box), (0, top_height))
            if top_height and left_width:
                corner_box = Box(column_boundaries[0], row_boundaries[0], column_boundaries[plan.stub_context_columns], row_boundaries[plan.header_context_rows])
                canvas.paste(crop("corner", corner_box), (0, 0))
            if canvas.size != (plan.output_width, plan.output_height):
                canvas = canvas.resize(
                    (plan.output_width, plan.output_height),
                    Image.Resampling.LANCZOS,
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, format="PNG", optimize=True)


    @staticmethod
    def _preview_box(box: Box, preview: Image.Image, meta: ImageMeta) -> Box:
        """把原图矩形映射到审计预览图。"""

        return Box(
            round(box.x1 * preview.width / meta.width),
            round(box.y1 * preview.height / meta.height),
            round(box.x2 * preview.width / meta.width),
            round(box.y2 * preview.height / meta.height),
        ).clamp(preview.width, preview.height)

    def _save_tile_audit(
        self,
        preview: Image.Image,
        meta: ImageMeta,
        regions: list[PreparedRegion],
        tiles_dir: Path,
        image_dir: Path,
    ) -> dict[str, object]:
        """生成总览叠加图和真实切块联系图，便于不调用 API 直接验收。"""

        overlay = preview.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        all_plans: list[TilePlan] = []
        for region in regions:
            region_box = self._preview_box(region.box, preview, meta)
            draw.rectangle(
                (region_box.x1, region_box.y1, region_box.x2, region_box.y2),
                outline=(0, 160, 0, 255),
                width=3,
            )
            # 淡色细线表示结构检测得到的所有逻辑边界；红框表示真正送给
            # 大模型负责识别的主体范围。重复表头/行名列只出现在切块图片中。
            for boundary in region.row_boundaries[1:-1]:
                y = round(boundary * preview.height / meta.height)
                draw.line((region_box.x1, y, region_box.x2, y), fill=(0, 170, 255, 100), width=1)
            for boundary in region.column_boundaries[1:-1]:
                x = round(boundary * preview.width / meta.width)
                draw.line((x, region_box.y1, x, region_box.y2), fill=(0, 170, 255, 100), width=1)
            for plan in region.tiles:
                all_plans.append(plan)
                tile_box = self._preview_box(plan.source_box, preview, meta)
                draw.rectangle(
                    (tile_box.x1, tile_box.y1, tile_box.x2, tile_box.y2),
                    outline=(255, 0, 0, 255),
                    width=3,
                )
                draw.text(
                    (tile_box.x1 + 3, tile_box.y1 + 3),
                    f"T{len(all_plans):03d}",
                    fill=(255, 0, 0, 255),
                )
        overlay_path = image_dir / "tile_overlay.png"
        overlay.save(overlay_path, format="PNG", optimize=True)

        card_width, card_height = 560, 420
        image_height = 340
        columns = min(3, max(1, len(all_plans)))
        rows = max(1, (len(all_plans) + columns - 1) // columns)
        sheet = Image.new("RGB", (columns * card_width, rows * card_height), "white")
        sheet_draw = ImageDraw.Draw(sheet)
        for index, plan in enumerate(all_plans):
            left = (index % columns) * card_width
            top = (index // columns) * card_height
            with Image.open(tiles_dir / plan.file_name) as source:
                tile = source.convert("RGB").copy()
            tile.thumbnail((card_width - 20, image_height - 10), Image.Resampling.LANCZOS)
            x = left + (card_width - tile.width) // 2
            y = top + 5 + (image_height - tile.height) // 2
            sheet.paste(tile, (x, y))
            sheet_draw.rectangle(
                (left, top, left + card_width - 1, top + card_height - 1),
                outline=(150, 150, 150),
            )
            sheet_draw.text(
                (left + 8, top + image_height + 5),
                f"T{index + 1:03d} region={plan.region_index}  R[{plan.logical_row_start},{plan.logical_row_end})  C[{plan.logical_column_start},{plan.logical_column_end})",
                fill=(0, 0, 0),
            )
            sheet_draw.text(
                (left + 8, top + image_height + 25),
                f"header={plan.header_context_rows}  stub={plan.stub_context_columns}  size={plan.output_width}x{plan.output_height}",
                fill=(0, 0, 0),
            )
        contact_path = image_dir / "tile_contact_sheet.jpg"
        sheet.save(contact_path, format="JPEG", quality=90, optimize=True)
        return {
            "tile_count": len(all_plans),
            "tile_overlay": str(overlay_path.resolve()),
            "tile_contact_sheet": str(contact_path.resolve()),
        }

    def _prepare_one(self, image_path: Path, image_sha256: str) -> Path:
        # 配置摘要进入目录名，切换检测/切片参数后不会与旧 tiles 混在一起。
        image_dir = self.work_dir / "prepared" / (
            f"{image_path.stem}_{image_sha256[:12]}_{self.config.digest()[:8]}"
        )
        tiles_dir = image_dir / "tiles"
        image_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)

        meta = self.backend.read_meta(image_path, known_sha256=image_sha256)
        preview = self._make_detection_preview(image_path, meta, image_dir)

        detected = self._detect_regions(preview, meta)
        if isinstance(self.detector, InkTableDetector):
            self._save_v6_detection_debug(preview, image_dir)
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
            region_box = item.box
            grid = self._analyze_grid(image_path, region_box, region_index, image_dir)
            plans = []
            if grid.available:
                region_box = Box(
                    grid.column_boundaries[0],
                    grid.row_boundaries[0],
                    grid.column_boundaries[-1],
                    grid.row_boundaries[-1],
                )
                plans = plan_grid_tiles(
                    region_box, region_index, grid.row_boundaries, grid.column_boundaries,
                    self.config.max_vlm_side, self.config.single_tile_min_scale,
                    self.config.repeat_header_rows, self.config.repeat_stub_columns,
                    self.config.max_logical_cells_per_tile,
                    self.config.preferred_min_logical_cells_per_tile,
                    self.config.max_tile_aspect_ratio,
                )
            if not plans:
                # 保留“检测到了边界但存在超大单元格”的原因，避免清单把它
                # 与“完全没有发现边界”混为一谈。
                fallback_source = (
                    f"{grid.source}-unplannable"
                    if grid.available
                    else "unavailable"
                )
                grid = GridStructure(
                    fallback_source, grid.row_boundaries, grid.column_boundaries
                )
                plans = plan_region_tiles(
                    region_box, region_index, self.config.max_vlm_side,
                    self.config.tile_overlap, self.config.single_tile_min_scale,
                )
            for plan in plans:
                self._save_tile(
                    image_path, tiles_dir / plan.file_name, plan,
                    grid.row_boundaries, grid.column_boundaries,
                )
            regions.append(
                PreparedRegion(
                    index=region_index,
                    box=region_box,
                    detector_source=item.source,
                    tiles=plans,
                    grid_source=grid.source,
                    row_boundaries=list(grid.row_boundaries),
                    column_boundaries=list(grid.column_boundaries),
                )
            )

        audit = self._save_tile_audit(
            preview, meta, regions, tiles_dir, image_dir
        )
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": meta.to_dict(),
            "backend": self.backend.name,
            "detector": self.detector.name,
            "config": self.config.to_dict(),
            "regions": [region.to_dict() for region in regions],
            "audit": audit,
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
            logical_row_start=int(raw.get("logical_row_start", 0)),
            logical_row_end=int(raw.get("logical_row_end", 1)),
            logical_column_start=int(raw.get("logical_column_start", 0)),
            logical_column_end=int(raw.get("logical_column_end", 1)),
            header_context_rows=int(raw.get("header_context_rows", 0)),
            stub_context_columns=int(raw.get("stub_context_columns", 0)),
            tiling_mode=str(raw.get("tiling_mode", "pixel_overlap")),
        )


    def _tile_ink_mask(
        self,
        gray: np.ndarray,
        row_boundaries: list[int],
        column_boundaries: list[int],
        tile: TilePlan,
    ) -> list[list[bool]]:
        """判断每个逻辑格是否真的有文字墨迹，供模型省略空格时软对齐。"""

        row_indices = [
            *range(tile.header_context_rows),
            *range(tile.logical_row_start, tile.logical_row_end),
        ]
        column_indices = [
            *range(tile.stub_context_columns),
            *range(tile.logical_column_start, tile.logical_column_end),
        ]
        result: list[list[bool]] = []
        for row in row_indices:
            values: list[bool] = []
            y1, y2 = row_boundaries[row], row_boundaries[row + 1]
            for column in column_indices:
                x1 = column_boundaries[column]
                x2 = column_boundaries[column + 1]
                # 向内缩一点，避免把单元格四周的黑色网格线当成文字。
                inset_x = min(max(1, (x2 - x1) // 10), max(1, (x2 - x1 - 1) // 3))
                inset_y = min(max(1, (y2 - y1) // 10), max(1, (y2 - y1 - 1) // 3))
                inner = gray[
                    y1 + inset_y : max(y1 + inset_y + 1, y2 - inset_y),
                    x1 + inset_x : max(x1 + inset_x + 1, x2 - inset_x),
                ]
                ink_pixels = int(
                    np.count_nonzero(inner < self.config.grid_white_threshold)
                )
                values.append(
                    ink_pixels > max(2, round(inner.size * 0.003))
                )
            result.append(values)
        return result

    def _recognize_manifest(self, manifest_path: Path, client: FinixDocClient) -> str:
        image_manifest = _load_json(manifest_path)
        region_markdowns: list[str] = []
        cache_model = _table_cache_model(client)
        total_tiles = sum(len(region["tiles"]) for region in image_manifest["regions"])
        completed_tiles = 0
        with Image.open(image_manifest["image"]["path"]) as source:
            source_gray = np.asarray(
                source.convert("L"),
                dtype=np.uint8,
            )
        for region in image_manifest["regions"]:
            contents: dict[tuple[int, int], str] = {}
            plans = [self._tile_from_dict(raw) for raw in region["tiles"]]
            row_boundaries = [
                int(value) for value in region.get("row_boundaries", [])
            ]
            column_boundaries = [
                int(value) for value in region.get("column_boundaries", [])
            ]
            tile_ink_masks = {
                (plan.row_index, plan.column_index): self._tile_ink_mask(
                    source_gray, row_boundaries, column_boundaries, plan
                )
                for plan in plans
            } if row_boundaries and column_boundaries else {}
            response_dir = manifest_path.parent / "responses"
            response_dir.mkdir(parents=True, exist_ok=True)
            for raw_tile in region["tiles"]:
                tile = self._tile_from_dict(raw_tile)
                tile_id = (tile.row_index, tile.column_index)
                tile_path = manifest_path.parent / "tiles" / tile.file_name
                # 官方 multipart 只上传图片和三个文本字段，不生成、不发送 prompt。
                prompt = select_request_prompt(
                    getattr(client, "protocol", CHAT_PROTOCOL),
                    lambda: build_table_prompt(tile),
                )
                image_bytes = tile_path.read_bytes()
                cache_key = self.cache.tile_key(image_bytes, prompt, cache_model)
                base_metadata = {
                    "tile": tile.to_dict(),
                    "model": cache_model,
                    "prompt_version": PROMPT_VERSION,
                }
                ink_mask = tile_ink_masks.get(tile_id)

                if tile.tiling_mode == "logical_grid" and _mask_has_no_text(ink_mask):
                    # 只看单元格内部；网格线本身不会触发“有文字”。这种纯空表
                    # 无需让自回归模型生成几百个重复的 <td>，直接保留预处理形状。
                    markdown = _empty_tile_html(tile)
                    source = "预处理判空，跳过模型"
                    self.cache.put_tile(
                        cache_key,
                        markdown,
                        {
                            **base_metadata,
                            "empty_table_fallback": True,
                            "empty_table_reason": "preprocess-no-cell-ink",
                        },
                    )
                else:
                    markdown = self.cache.get_tile(cache_key)
                    source = "缓存"
                    if markdown is not None and not markdown.strip():
                        if tile.tiling_mode == "logical_grid":
                            markdown = _empty_tile_html(tile)
                            source = "空缓存修复"
                            self.cache.put_tile(
                                cache_key,
                                markdown,
                                {
                                    **base_metadata,
                                    "empty_table_fallback": True,
                                    "empty_table_reason": "cached-empty-markdown",
                                },
                            )
                        else:
                            # 像素重叠切块没有可靠的逻辑行列数，不能凭空补表。
                            markdown = None

                    if markdown is None:
                        # 参数升级后尝试迁移旧成功缓存。这里只检查响应能否解析；
                        # 行列差异由后面的墨迹软对齐处理并写入 warning，避免把
                        # 只是省略空行空列的结果再次送进模型。
                        for legacy_model in _table_legacy_cache_models(client):
                            legacy_key = self.cache.tile_key(
                                image_bytes, prompt, legacy_model
                            )
                            candidate = self.cache.get_tile(legacy_key)
                            if candidate is None:
                                continue
                            legacy_metadata = {
                                **base_metadata,
                                "migrated_from": legacy_model,
                            }
                            if not candidate.strip():
                                if tile.tiling_mode != "logical_grid":
                                    continue
                                candidate = _empty_tile_html(tile)
                                legacy_metadata.update(
                                    {
                                        "empty_table_fallback": True,
                                        "empty_table_reason": "cached-empty-markdown",
                                    }
                                )
                            elif tile.tiling_mode == "logical_grid":
                                try:
                                    normalize_table_response(candidate)
                                except HtmlTableMergeError:
                                    continue
                            markdown = candidate
                            source = "兼容缓存"
                            self.cache.put_tile(
                                cache_key, markdown, legacy_metadata
                            )
                            break

                    if markdown is None:
                        source = "API"
                        print(
                            f"[图表识别 {completed_tiles + 1:02d}/{total_tiles:02d}] "
                            f"{tile.file_name}：请求 API",
                            flush=True,
                        )
                        empty_reason: str | None = None
                        try:
                            markdown = client.recognize(tile_path, prompt)
                        except RuntimeError as error:
                            if (
                                tile.tiling_mode == "logical_grid"
                                and "返回了空 Markdown" in str(error)
                            ):
                                markdown = _empty_tile_html(tile)
                                empty_reason = "model-empty-markdown"
                                source = "模型空响应，按预处理补空表"
                            else:
                                raise
                        if not markdown.strip():
                            if tile.tiling_mode != "logical_grid":
                                raise RuntimeError(
                                    "模型返回了空结果，且该切片没有可靠的预处理行列数"
                                )
                            markdown = _empty_tile_html(tile)
                            empty_reason = "model-empty-markdown"
                            source = "模型空响应，按预处理补空表"
                        metadata = dict(base_metadata)
                        if empty_reason is not None:
                            metadata.update(
                                {
                                    "empty_table_fallback": True,
                                    "empty_table_reason": empty_reason,
                                }
                            )
                        self.cache.put_tile(cache_key, markdown, metadata)

                (response_dir / f"{Path(tile.file_name).stem}.md").write_text(
                    markdown, encoding="utf-8"
                )
                completed_tiles += 1
                print(
                    f"[图表识别 {completed_tiles:02d}/{total_tiles:02d}] "
                    f"{tile.file_name}：{source}完成，结果 {len(markdown)} 字符",
                    flush=True,
                )
                contents[tile_id] = markdown

            try:
                if plans and plans[0].tiling_mode == "logical_grid":
                    if len(contents) == 1:
                        expected_rows = len(region.get("row_boundaries", [])) - 1
                        expected_columns = len(region.get("column_boundaries", [])) - 1
                        key = next(iter(contents))
                        region_markdown, actual = normalize_table_response_soft(
                            contents[key],
                            expected_rows,
                            expected_columns,
                            tile_ink_masks.get(key),
                        )
                        quality = {
                            "logical_rows": expected_rows,
                            "logical_columns": expected_columns,
                            "actual_rows": actual["rows"],
                            "actual_columns": actual["columns"],
                            "warnings": actual["warnings"],
                            "status": "warning" if actual["warnings"] else "ok",
                        }
                    else:
                        region_markdown, quality = merge_logical_tiles(
                            contents, plans,
                            len(region["row_boundaries"]) - 1,
                            len(region["column_boundaries"]) - 1,
                            tile_ink_masks,
                        )
                    _json_dump(
                        manifest_path.parent / "quality" / f"region_{region['index']:03d}.json",
                        quality,
                    )
                    for warning in quality.get("warnings", []):
                        print(
                            f"[图表警告] 区域 {region['index'] + 1}：{warning}",
                            flush=True,
                        )
                else:
                    region_markdown = (
                        next(iter(contents.values()))
                        if len(contents) == 1
                        else merge_markdown_grid(contents)
                    )
            except (MarkdownMergeError, HtmlTableMergeError) as error:
                warning_data = {
                    "status": "warning",
                    "region_index": region["index"],
                    "error": str(error),
                    "fallback": "按切片顺序保留模型原始输出",
                    "response_files": [
                        f"responses/{Path(plan.file_name).stem}.md"
                        for plan in plans
                    ],
                }
                _json_dump(
                    manifest_path.parent
                    / "quality"
                    / f"region_{region['index']:03d}.json",
                    warning_data,
                )
                _json_dump(
                    manifest_path.parent / "merge_warning.json",
                    {
                        "region_index": region["index"],
                        "error": str(error),
                        "responses": {
                            f"r{row}_c{column}": text
                            for (row, column), text in contents.items()
                        },
                    },
                )
                print(
                    f"[图表警告] 区域 {region['index'] + 1} 无法可靠合并："
                    f"{error}；已保留模型原始输出继续生成。",
                    flush=True,
                )
                region_markdown = "\n\n".join(
                    contents[key] for key in sorted(contents)
                )
            region_markdowns.append(region_markdown.strip())
        return "\n\n".join(text for text in region_markdowns if text).strip()

    def recognize_dataset(
        self,
        dataset_manifest_path: str | Path,
        client: FinixDocClient,
        output_csv: str | Path,
        max_workers: int = 1,
    ) -> dict[str, str]:
        dataset_manifest = _load_json(Path(dataset_manifest_path))
        cache_model = _table_cache_model(client)
        recognition_digest = hashlib.sha256(
            (
                dataset_manifest["config_digest"]
                + "\0"
                + cache_model
                + "\0"
                + PROMPT_VERSION
            ).encode("utf-8")
        ).hexdigest()

        if max_workers <= 0:
            raise ValueError("max_workers 必须大于 0")

        unique_items: dict[str, dict[str, Any]] = {}
        for item in dataset_manifest["items"]:
            unique_items.setdefault(item["canonical_file_name"], item)

        def recognize_one(item: dict[str, Any]) -> tuple[str, str]:
            canonical_name = item["canonical_file_name"]
            cached = self.cache.get_image(item["sha256"], recognition_digest)
            if cached is None:
                cached = self._recognize_manifest(
                    Path(item["image_manifest"]), client
                )
                self.cache.put_image(
                    item["sha256"],
                    recognition_digest,
                    cached,
                    {
                        "canonical_file_name": canonical_name,
                        "model": cache_model,
                        "prompt_version": PROMPT_VERSION,
                    },
                )
            return canonical_name, cached

        canonical_items = list(unique_items.values())
        if max_workers == 1 or len(canonical_items) <= 1:
            pairs = [recognize_one(item) for item in canonical_items]
        else:
            worker_count = min(max_workers, len(canonical_items))
            print(
                f"[图表并行] {worker_count} 个任务并发识别 "
                f"{len(canonical_items)} 张唯一图片",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pairs = list(executor.map(recognize_one, canonical_items))
        canonical_results = dict(pairs)

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
