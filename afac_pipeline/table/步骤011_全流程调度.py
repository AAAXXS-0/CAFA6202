"""图表分支端到端编排。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory

from ..common.cache import ResultCache
from ..common.recognition_errors import IncompleteImageRecognitionError
from ..common.preprocessing_audit import (
    建立中文中间产物,
    写入预处理检查点,
)
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
    normalize_table_response_soft,
    render_empty_table,
)
from .步骤001_墨水密度定位 import density_visualization
from .步骤008_Markdown表格合并 import MarkdownMergeError, merge_markdown_grid
from .步骤005_黑线白带结构检测 import (
    V6RegionResult,
    build_column_smear_mask,
    build_row_smear_mask,
    detect_v6_grid,
    detect_v6_regions,
    detected_boxes,
    smeared_content_box,
)
from ..common.models import Box, DetectedBox, ImageMeta, PreparedRegion, TilePlan
from ..common.submission import write_submission
from ..common.vlm_client import (
    CHAT_PROTOCOL,
    FinixDocClient,
    select_request_prompt,
)


PROMPT_VERSION = "table-physical-grid-v6-top-context-title"


class TablePreprocessingError(RuntimeError):
    """物理网格不可信或无法原尺寸切块时立即终止预处理。"""


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _raw_attempt_number(path: Path) -> int:
    """从原始响应名读取 attempt 数字，避免 attempt_10 排在 attempt_9 前面。"""

    match = re.search(r"_attempt_(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def _raw_response_candidates(raw_dir: Path, tile_stem: str) -> list[Path]:
    """按真实尝试次数从新到旧返回全部历史原始响应。"""

    return sorted(
        raw_dir.glob(f"{tile_stem}_attempt_*.md"),
        key=_raw_attempt_number,
        reverse=True,
    )


def _raw_metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """只有图片、切块、模型和提示词签名全相同时才允许恢复。"""

    keys = ("tile_sha256", "prompt_sha256", "cache_key", "model")
    return all(metadata.get(key) == expected.get(key) for key in keys)


def _is_truncated_empty_html(response: str) -> bool:
    """判断响应是否只有空 HTML 单元格，且恰好在末尾标签中途截断。

    这不是普通 HTML 自动修复：只要出现一个可见字符就返回 False，交回严格
    质量闸门重试。连续复核后，上层仅按预处理 R×C 生成全空矩阵。
    """

    if len(re.findall(r"<table\b", response, re.IGNORECASE)) != 1:
        return False
    if re.search(r"</table\s*>", response, re.IGNORECASE):
        return False
    if not re.search(r"</t[dh]\s*>", response, re.IGNORECASE):
        return False
    normalized = re.sub(
        r"^```(?:markdown|html)?\s*",
        "",
        response.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(
        r"</?(?:table|tr|td|th)\b[^>]*>",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    # API 的 token 截断通常停在“<td”或“</tr”中间，只允许末尾这一段残标签。
    remainder = re.sub(
        r"</?(?:table|tr|td|th)?[^<>]*$",
        "",
        remainder,
        flags=re.IGNORECASE,
    )
    return not remainder.strip()


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
        visible_rows = (
            tile.header_context_rows + tile.logical_row_end - tile.logical_row_start
        )
        visible_columns = (
            tile.stub_context_columns
            + tile.logical_column_end
            - tile.logical_column_start
        )
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


def build_top_context_prompt() -> str:
    """顶部候选区只识别可见标题、单位和注释，不虚构表格。"""

    return """你正在识别金融文档表格上方的一小块候选标题区域。

只输出图片中真实可见的标题、单位、日期或注释文字，保持阅读顺序。
如果图片中没有可读文字，输出空内容；严禁补全，不要生成表格，不要解释。
直接输出 Markdown 文本："""


def _logical_tile_shape(tile: TilePlan) -> tuple[int, int]:
    """返回逻辑网格切片里实际可见的行数和列数。"""

    rows = tile.header_context_rows + tile.logical_row_end - tile.logical_row_start
    columns = (
        tile.stub_context_columns + tile.logical_column_end - tile.logical_column_start
    )
    return rows, columns


def _mask_has_no_text(ink_mask: list[list[bool]] | None) -> bool:
    """判断预处理是否已确认所有逻辑格内都没有文字。

    ``None`` 表示没有可靠的逻辑网格信息，不能擅自当作空表。
    """

    return (
        ink_mask is not None
        and bool(ink_mask)
        and all(not has_text for row in ink_mask for has_text in row)
    )


def _mask_has_any_text(ink_mask: list[list[bool]] | None) -> bool:
    """墨迹 bool 是否认为至少一个物理格内存在文字。"""

    return bool(ink_mask and any(has_text for row in ink_mask for has_text in row))


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
    def _map_preview_box(
        box: Box, preview_width: int, preview_height: int, meta: ImageMeta
    ) -> Box:
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
                    self._map_preview_box(
                        item.box, preview.width, preview.height, meta
                    ),
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
            longest = round(
                max(meta.width, meta.height) * self.config.table_analysis_scale
            )
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
            draw.rectangle(
                (box.x1, box.y1, box.x2, box.y2), outline=(0, 80, 255, 255), width=3
            )
            draw.text(
                (box.x1 + 4, box.y1 + 4), f"split-{index + 1}", fill=(0, 80, 255, 255)
            )
        for index, box in enumerate(result.analysis_boxes):
            draw.rectangle(
                (box.x1, box.y1, box.x2, box.y2), outline=(160, 0, 255, 255), width=3
            )
            draw.text(
                (box.x1 + 4, box.y1 + 22), f"table-{index + 1}", fill=(160, 0, 255, 255)
            )
        overlay.save(output_dir / "split_and_analysis_boxes.png")

        # 正式测试阶段保留每张分表从原块到强晕染分析框的全部中间产物。
        gray = np.asarray(preview.convert("L"), dtype=np.uint8)
        for index, split_box in enumerate(result.split_boxes):
            table_dir = output_dir / f"第{index + 1:03d}表_分析框"
            table_dir.mkdir(parents=True, exist_ok=True)
            crop = preview.crop((split_box.x1, split_box.y1, split_box.x2, split_box.y2))
            crop.save(table_dir / "01_横向分表块.png")
            local_ink = gray[
                split_box.y1 : split_box.y2,
                split_box.x1 : split_box.x2,
            ] < self.config.ink_threshold
            local_box, smeared, component, info = smeared_content_box(
                local_ink,
                self.config,
            )
            Image.fromarray(np.where(local_ink, 0, 255).astype(np.uint8)).save(
                table_dir / "02_灰度二值图.png"
            )
            Image.fromarray(np.where(smeared, 0, 255).astype(np.uint8)).save(
                table_dir / "03_强二维晕染.png"
            )
            Image.fromarray(np.where(component, 0, 255).astype(np.uint8)).save(
                table_dir / "04_最大主体连通块.png"
            )
            local_overlay = crop.copy()
            ImageDraw.Draw(local_overlay).rectangle(
                (local_box.x1, local_box.y1, local_box.x2, local_box.y2),
                outline=(0, 200, 60),
                width=3,
            )
            local_overlay.save(table_dir / "05_最终分析框.png")
            _json_dump(table_dir / "06_分析框判定数据.json", info)

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
        """每张子表只生成一张50%分析图，在同一坐标系检测黑线和白缝。"""

        analysis_dir = image_dir / "grid_analysis"
        analysis_path = analysis_dir / f"region_{region_index:03d}.png"
        black_analysis = None
        black_analysis_path = analysis_path
        if isinstance(self.detector, InkTableDetector):
            scale = self.config.table_black_line_scale
        else:
            scale = min(
                1.0,
                self.config.grid_analysis_max_side / max(region.width, region.height),
            )
        self.backend.save_crop(image_path, region, analysis_path, scale=scale)
        with Image.open(analysis_path) as source:
            analysis = source.convert("RGB").copy()
        diagnostics = None
        if isinstance(self.detector, InkTableDetector):
            analysis_longest = round(max(region.width, region.height) * scale)
            if analysis_longest > self.config.table_black_analysis_max_side:
                raise TablePreprocessingError(
                    f"{image_path.name} 的表格区域 {region_index} 在50%统一分析时"
                    f"最长边为 {analysis_longest}，超过安全上限 "
                    f"{self.config.table_black_analysis_max_side}"
                )
            black_analysis = analysis
            grid, diagnostics = detect_v6_grid(
                analysis,
                region,
                self.config,
                black_analysis_image=black_analysis,
            )
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
        overlay.save(analysis_dir / f"region_{region_index:03d}_boundaries.png")
        if diagnostics is not None:
            # 行、列白缝均保存“擦黑线前后”和“晕染后”图，避免只看到
            # 最终边界却无法判断白缝在哪一步消失。
            ink50 = (
                np.asarray(black_analysis.convert("L"), dtype=np.uint8)
                < self.config.grid_white_threshold
            )
            Image.fromarray(np.where(ink50, 0, 255).astype(np.uint8)).save(
                analysis_dir / f"region_{region_index:03d}_01_灰度二值图.png"
            )
            row_smear = build_row_smear_mask(
                ink50,
                list(diagnostics.black_columns_at_whitespace_scale),
                self.config,
            )
            Image.fromarray(
                np.where(row_smear["erased"], 0, 255).astype(np.uint8)
            ).save(
                analysis_dir / f"region_{region_index:03d}_02_行白缝_擦除竖黑线.png"
            )
            Image.fromarray(
                np.where(row_smear["mask"], 0, 255).astype(np.uint8)
            ).save(
                analysis_dir / f"region_{region_index:03d}_03_行白缝_左右晕染.png"
            )
            column_smear = build_column_smear_mask(
                ink50,
                list(diagnostics.black_rows),
                list(diagnostics.used_black_columns),
                self.config,
            )
            Image.fromarray(
                np.where(column_smear["erased"], 0, 255).astype(np.uint8)
            ).save(
                analysis_dir / f"region_{region_index:03d}_04_列白缝_擦除全部黑线.png"
            )
            Image.fromarray(
                np.where(column_smear["mask"], 0, 255).astype(np.uint8)
            ).save(
                analysis_dir / f"region_{region_index:03d}_05_列白缝_二维自适应晕染.png"
            )
            smear_diagnostics = {
                "行白缝": {
                    key: value for key, value in row_smear.items()
                    if key not in {"erased", "mask", "rows"}
                }
                | {"检测到的白缝数量": len(row_smear["rows"])},
                "列白缝": {
                    key: value for key, value in column_smear.items()
                    if key not in {"erased", "mask"}
                },
            }
            # 表体滑窗中间图：红框是不稳定/未选窗口，绿框是最终连续表体段。
            if diagnostics.body_window_results:
                body_overlay = black_analysis.copy()
                body_draw = ImageDraw.Draw(body_overlay)
                selected_indices = (
                    set(range(*diagnostics.body_window_selected))
                    if diagnostics.body_window_selected is not None
                    else set()
                )
                for window_index, window in enumerate(
                    diagnostics.body_window_results
                ):
                    color = (
                        (0, 180, 0)
                        if window_index in selected_indices
                        else (220, 60, 60)
                    )
                    start = int(window["start"])
                    end = int(window["end"])
                    body_draw.rectangle(
                        (0, start, body_overlay.width - 1, end - 1),
                        outline=color,
                        width=4,
                    )
                    body_draw.text(
                        (8, start + 4),
                        (
                            f"W{window_index:02d} "
                            f"bands={window['band_count']} "
                            f"threshold={window['threshold']}"
                        ),
                        fill=color,
                    )
                if diagnostics.body_window_box is not None:
                    x1, y1, x2, y2 = diagnostics.body_window_box
                    body_draw.rectangle(
                        (x1, y1, x2 - 1, y2 - 1),
                        outline=(255, 180, 0),
                        width=7,
                    )
                    black_analysis.crop(
                        diagnostics.body_window_box
                    ).save(
                        analysis_dir
                        / f"region_{region_index:03d}_body_selected.png"
                    )
                body_overlay.save(
                    analysis_dir
                    / f"region_{region_index:03d}_body_windows.png"
                )

            # 在统一50%分析图上画出黑线候选及最终采用边界。
            black_overlay = black_analysis.copy()
            black_draw = ImageDraw.Draw(black_overlay)
            for line in diagnostics.black_rows:
                black_draw.line(
                    (line.start, line.position, line.end, line.position),
                    fill=(255, 0, 0),
                    width=2,
                )
            for line in diagnostics.black_columns:
                black_draw.line(
                    (line.position, line.start, line.position, line.end),
                    fill=(0, 80, 255),
                    width=2,
                )
            for item in diagnostics.rejected_black_columns:
                line = item.line
                black_draw.line(
                    (line.position, line.start, line.position, line.end),
                    fill=(255, 0, 255),
                    width=4,
                )
            for boundary in grid.row_boundaries[1:-1]:
                y = round(
                    (boundary - region.y1) * black_analysis.height / region.height
                )
                black_draw.line(
                    (0, y, black_analysis.width, y), fill=(0, 180, 0), width=2
                )
            for boundary in grid.column_boundaries[1:-1]:
                x = round((boundary - region.x1) * black_analysis.width / region.width)
                black_draw.line(
                    (x, 0, x, black_analysis.height), fill=(0, 180, 0), width=2
                )
            black_overlay.save(
                analysis_dir / f"region_{region_index:03d}_black_candidates.png"
            )
            cleanup_overlay = black_analysis.copy()
            cleanup_draw = ImageDraw.Draw(cleanup_overlay)
            for line in diagnostics.used_black_columns:
                cleanup_draw.line(
                    (line.position, line.start, line.position, line.end),
                    fill=(0, 180, 0),
                    width=2,
                )
            for item in diagnostics.rejected_black_columns:
                line = item.line
                cleanup_draw.line(
                    (line.position, line.start, line.position, line.end),
                    fill=(255, 0, 255),
                    width=4,
                )
            cleanup_overlay.save(
                analysis_dir / f"region_{region_index:03d}_black_cleanup.png"
            )
            # 黑线和白缝现在都来自同一张50%分析图；清理图沿用同一坐标系。
            white_cleanup_base = (
                black_analysis if diagnostics.white_column_uses_black_scale else analysis
            )
            white_cleanup = white_cleanup_base.copy()
            white_cleanup_draw = ImageDraw.Draw(white_cleanup)
            for band in diagnostics.raw_white_column_bands:
                white_cleanup_draw.rectangle(
                    (
                        band.start,
                        0,
                        max(band.start, band.end - 1),
                        white_cleanup.height - 1,
                    ),
                    outline=(0, 80, 255),
                    width=1,
                )
            for band in diagnostics.used_white_column_bands:
                white_cleanup_draw.line(
                    (band.position, 0, band.position, white_cleanup.height),
                    fill=(0, 180, 0),
                    width=2,
                )
            for item in diagnostics.rejected_white_column_bands:
                white_cleanup_draw.line(
                    (
                        item.band.position,
                        0,
                        item.band.position,
                        white_cleanup.height,
                    ),
                    fill=(255, 0, 255),
                    width=3,
                )
            white_cleanup.save(
                analysis_dir
                / f"region_{region_index:03d}_white_column_cleanup.png"
            )
            diagnostic_data = diagnostics.to_dict()
            diagnostic_data.update(
                {
                    "analysis_size": list(analysis.size),
                    "whitespace_analysis_size": list(analysis.size),
                    "black_analysis_size": list(black_analysis.size),
                    "whitespace_analysis_path": str(analysis_path.resolve()),
                    "black_analysis_path": str(black_analysis_path.resolve()),
                    "white_column_analysis_size": list(white_cleanup_base.size),
                    "white_column_analysis_path": str(
                        (black_analysis_path if diagnostics.white_column_uses_black_scale else analysis_path).resolve()
                    ),
                    "body_window_overlay_path": (
                        str(
                            (
                                analysis_dir
                                / f"region_{region_index:03d}_body_windows.png"
                            ).resolve()
                        )
                        if diagnostics.body_window_results
                        else None
                    ),
                    "body_selected_path": (
                        str(
                            (
                                analysis_dir
                                / f"region_{region_index:03d}_body_selected.png"
                            ).resolve()
                        )
                        if diagnostics.body_window_box is not None
                        else None
                    ),
                    "source_region": region.to_dict(),
                    "grid": grid.to_dict(),
                    "smear_diagnostics": smear_diagnostics,
                }
            )
            _json_dump(
                analysis_dir / f"region_{region_index:03d}_diagnostics.json",
                diagnostic_data,
            )
        return grid

    def _save_top_context(
        self,
        image_path: Path,
        analysis_region: Box,
        grid: GridStructure,
        region_index: int,
        image_dir: Path,
    ) -> dict[str, Any] | None:
        """保存分析框顶部到第一根横线的候选标题区。"""

        if not self.config.top_context_enabled or not grid.row_boundaries:
            return None
        first_line = int(grid.row_boundaries[0])
        # 黑线边界记录的是线芯中心，向上留保护像素，避免把半根横线
        # 当成标题文字；白带模式通常从分析框外沿开始，这里自然为空。
        y2 = min(
            first_line - self.config.top_context_line_guard_px,
            analysis_region.y2,
        )
        x1 = max(analysis_region.x1, int(grid.column_boundaries[0]))
        x2 = min(analysis_region.x2, int(grid.column_boundaries[-1]))
        box = Box(x1, analysis_region.y1, x2, y2)
        if box.width <= 0 or box.height <= 0:
            return None

        relative = Path("top_context") / f"region_{region_index:03d}.png"
        output_path = image_dir / relative
        self.backend.save_crop(image_path, box, output_path)
        with Image.open(output_path) as source:
            image = source.convert("L").copy()
        pixels = np.asarray(image)
        ink_pixels = int(np.count_nonzero(pixels < self.config.grid_white_threshold))
        pixel_count = max(1, image.width * image.height)
        minimum_ink = max(
            8,
            round(pixel_count * self.config.top_context_min_ink_ratio),
        )
        has_text = ink_pixels >= minimum_ink
        recognition_relative: Path | None = None
        recognition_box: Box | None = None
        if has_text:
            ink_y, ink_x = np.nonzero(pixels < self.config.grid_white_threshold)
            padding = max(
                8,
                round(min(image.width, image.height) * 0.15),
            )
            recognition_box = Box(
                box.x1 + max(0, int(ink_x.min()) - padding),
                box.y1 + max(0, int(ink_y.min()) - padding),
                box.x1 + min(image.width, int(ink_x.max()) + 1 + padding),
                box.y1 + min(image.height, int(ink_y.max()) + 1 + padding),
            )
            recognition_relative = (
                Path("top_context") / f"region_{region_index:03d}_content.png"
            )
            self.backend.save_crop(
                image_path,
                recognition_box,
                image_dir / recognition_relative,
            )
        return {
            "file_name": relative.as_posix(),
            "box": box.to_dict(),
            "image_size": [image.width, image.height],
            "ink_pixels": ink_pixels,
            "ink_ratio": ink_pixels / pixel_count,
            "minimum_ink_pixels": minimum_ink,
            "has_text": has_text,
            "recognition_file_name": (
                recognition_relative.as_posix()
                if recognition_relative is not None
                else None
            ),
            "recognition_box": (
                recognition_box.to_dict() if recognition_box is not None else None
            ),
            "source": "analysis-top-to-first-horizontal-line",
            "line_guard_px": self.config.top_context_line_guard_px,
        }

    def _save_tile(
        self,
        image_path: Path,
        output_path: Path,
        plan: TilePlan,
        row_boundaries: tuple[int, ...],
        column_boundaries: tuple[int, ...],
    ) -> None:
        """保存普通裁片，或拼出“左上角 + 表头 + 行名列 + 主体”图片。"""

        if plan.scale != 1.0:
            raise TablePreprocessingError(
                f"正式图表切片禁止缩放，但计划中的 scale={plan.scale}"
            )
        if plan.header_context_rows == 0 and plan.stub_context_columns == 0:
            self.backend.save_crop(image_path, plan.source_box, output_path)
            with Image.open(output_path) as source:
                tile = source.convert("RGB").copy()
            if tile.size != (plan.output_width, plan.output_height):
                raise TablePreprocessingError(
                    f"实际裁片尺寸{tile.size}与计划尺寸"
                    f"{(plan.output_width, plan.output_height)}不一致；禁止自动缩放修正"
                )
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
            left_width = (
                column_boundaries[plan.stub_context_columns] - column_boundaries[0]
            )
            canvas = Image.new(
                "RGB", (body.width + left_width, body.height + top_height), "white"
            )
            canvas.paste(body, (left_width, top_height))
            if top_height:
                top_box = Box(
                    plan.source_box.x1,
                    row_boundaries[0],
                    plan.source_box.x2,
                    row_boundaries[plan.header_context_rows],
                )
                canvas.paste(crop("top", top_box), (left_width, 0))
            if left_width:
                left_box = Box(
                    column_boundaries[0],
                    plan.source_box.y1,
                    column_boundaries[plan.stub_context_columns],
                    plan.source_box.y2,
                )
                canvas.paste(crop("left", left_box), (0, top_height))
            if top_height and left_width:
                corner_box = Box(
                    column_boundaries[0],
                    row_boundaries[0],
                    column_boundaries[plan.stub_context_columns],
                    row_boundaries[plan.header_context_rows],
                )
                canvas.paste(crop("corner", corner_box), (0, 0))
            if canvas.size != (plan.output_width, plan.output_height):
                raise TablePreprocessingError(
                    f"上下文拼接尺寸{canvas.size}与计划尺寸"
                    f"{(plan.output_width, plan.output_height)}不一致；禁止自动缩放修正"
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
            if region.top_context:
                top_box = self._preview_box(
                    Box.from_dict(region.top_context["box"]),
                    preview,
                    meta,
                )
                draw.rectangle(
                    (top_box.x1, top_box.y1, top_box.x2, top_box.y2),
                    outline=(180, 0, 180, 255),
                    width=3,
                )
                draw.text(
                    (top_box.x1 + 3, top_box.y1 + 3),
                    "TOP",
                    fill=(180, 0, 180, 255),
                )
            # 淡色细线表示结构检测得到的所有逻辑边界；红框表示真正送给
            # 大模型负责识别的主体范围。重复表头/行名列只出现在切块图片中。
            for boundary in region.row_boundaries[1:-1]:
                y = round(boundary * preview.height / meta.height)
                draw.line(
                    (region_box.x1, y, region_box.x2, y),
                    fill=(0, 170, 255, 100),
                    width=1,
                )
            for boundary in region.column_boundaries[1:-1]:
                x = round(boundary * preview.width / meta.width)
                draw.line(
                    (x, region_box.y1, x, region_box.y2),
                    fill=(0, 170, 255, 100),
                    width=1,
                )
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
            tile.thumbnail(
                (card_width - 20, image_height - 10), Image.Resampling.LANCZOS
            )
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

    def _single_image_directory(
        self,
        image_path: Path,
        image_sha256: str,
    ) -> Path:
        """配置摘要进入目录名，参数变化时不会误复用旧产物。"""

        return (
            self.work_dir
            / "prepared"
            / (
                f"{image_path.stem}_{image_sha256[:12]}_"
                f"{self.config.digest()[:8]}"
            )
        )

    def _find_reusable_manifest(
        self,
        manifest_path: Path,
        image_sha256: str,
    ) -> Path | None:
        """确认 R×C、墨迹矩阵和所有 API 切块完整后才复用单图缓存。"""

        if not manifest_path.is_file():
            return None
        try:
            manifest = _load_json(manifest_path)
            if manifest.get("schema_version") != 4:
                return None
            if manifest.get("config") != self.config.to_dict():
                return None
            if manifest.get("image", {}).get("sha256") != image_sha256:
                return None
            regions = manifest.get("regions")
            if not isinstance(regions, list) or not regions:
                return None
            for region in regions:
                rows = region.get("row_boundaries")
                columns = region.get("column_boundaries")
                mask = region.get("cell_ink_mask")
                tiles = region.get("tiles")
                if (
                    not isinstance(rows, list)
                    or len(rows) < 2
                    or not isinstance(columns, list)
                    or len(columns) < 2
                    or not isinstance(mask, list)
                    or len(mask) != len(rows) - 1
                    or any(
                        not isinstance(row, list)
                        or len(row) != len(columns) - 1
                        for row in mask
                    )
                    or not isinstance(tiles, list)
                    or not tiles
                ):
                    return None
                for tile in tiles:
                    file_name = tile.get("file_name")
                    if (
                        not file_name
                        or not (
                            manifest_path.parent / "tiles" / file_name
                        ).is_file()
                    ):
                        return None
                context = region.get("top_context") or {}
                context_name = (
                    context.get("recognition_file_name")
                    or context.get("file_name")
                )
                if context.get("has_text") and (
                    not context_name
                    or not (manifest_path.parent / context_name).is_file()
                ):
                    return None
        except (OSError, ValueError, TypeError, KeyError):
            return None
        return manifest_path

    def _prepare_one(self, image_path: Path, image_sha256: str) -> Path:
        image_dir = self._single_image_directory(image_path, image_sha256)
        manifest_path = image_dir / "manifest.json"
        reusable = self._find_reusable_manifest(manifest_path, image_sha256)
        if reusable is not None:
            print(
                f"[图表准备复用] {image_path.name}：R×C、墨迹矩阵和切块均完整",
                flush=True,
            )
            return reusable

        tiles_dir = image_dir / "tiles"
        image_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        stale_error = image_dir / "预处理致命错误.json"
        if stale_error.exists():
            stale_error.unlink()

        meta = self.backend.read_meta(image_path, known_sha256=image_sha256)
        preview = self._make_detection_preview(image_path, meta, image_dir)

        detected = self._detect_regions(preview, meta)
        if isinstance(self.detector, InkTableDetector):
            self._save_v6_detection_debug(preview, image_dir)
        if not detected:
            error_path = image_dir / "预处理致命错误.json"
            _json_dump(
                error_path,
                {
                    "image_name": image_path.name,
                    "stage": "分表",
                    "reason": "没有检测到任何表格区域",
                    "diagnostic_directory": str(image_dir.resolve()),
                },
            )
            raise TablePreprocessingError(
                f"{image_path.name} 图表预处理失败：没有检测到任何表格区域；"
                f"诊断：{error_path.resolve()}"
            )
        self._draw_preview_boxes(preview, detected, meta).save(
            image_dir / "preview_detected.png", format="PNG", optimize=True
        )

        # R×C确定后立即逐格判断墨迹；准备阶段只读取一次原图灰度。
        with Image.open(image_path) as source:
            source_gray = np.asarray(source.convert("L"), dtype=np.uint8)

        regions: list[PreparedRegion] = []
        for region_index, item in enumerate(detected):
            region_box = item.box
            grid = self._analyze_grid(image_path, region_box, region_index, image_dir)
            if not grid.available:
                error_path = image_dir / "预处理致命错误.json"
                _json_dump(
                    error_path,
                    {
                        "image_name": image_path.name,
                        "stage": "物理网格检测",
                        "region_index": region_index,
                        "region_box": region_box.to_dict(),
                        "grid_source": grid.source,
                        "reason": "没有同时得到可信的行边界和列边界",
                        "boundary_image": str(
                            (
                                image_dir
                                / "grid_analysis"
                                / f"region_{region_index:03d}_boundaries.png"
                            ).resolve()
                        ),
                        "diagnostics": str(
                            (
                                image_dir
                                / "grid_analysis"
                                / f"region_{region_index:03d}_diagnostics.json"
                            ).resolve()
                        ),
                    },
                )
                raise TablePreprocessingError(
                    f"{image_path.name} 的表格区域 {region_index} 没有可信的 R×C 物理网格；"
                    f"禁止缩放或像素兜底。诊断：{error_path.resolve()}"
                )
            analysis_region_box = region_box
            top_context = self._save_top_context(
                image_path,
                analysis_region_box,
                grid,
                region_index,
                image_dir,
            )
            region_box = Box(
                grid.column_boundaries[0],
                grid.row_boundaries[0],
                grid.column_boundaries[-1],
                grid.row_boundaries[-1],
            )
            cell_ink_mask = self._physical_cell_ink_mask(
                source_gray,
                list(grid.row_boundaries),
                list(grid.column_boundaries),
            )
            cell_ink_mask_path = (
                image_dir
                / "grid_analysis"
                / f"region_{region_index:03d}_cell_ink_mask.json"
            )
            _json_dump(
                cell_ink_mask_path,
                {
                    "physical_shape": [grid.row_count, grid.column_count],
                    "ink_cell_count": sum(
                        int(value)
                        for row in cell_ink_mask
                        for value in row
                    ),
                    "cell_ink_mask": cell_ink_mask,
                    "meaning": "true表示该R×C物理格内部存在墨迹",
                },
            )
            plans = plan_grid_tiles(
                region_box,
                region_index,
                grid.row_boundaries,
                grid.column_boundaries,
                self.config.max_vlm_side,
                self.config.repeat_header_rows,
                self.config.repeat_stub_columns,
                self.config.max_logical_cells_per_tile,
                self.config.preferred_min_logical_cells_per_tile,
                self.config.max_tile_aspect_ratio,
            )
            if not plans:
                row_sizes = [
                    right - left
                    for left, right in zip(grid.row_boundaries, grid.row_boundaries[1:])
                ]
                column_sizes = [
                    right - left
                    for left, right in zip(
                        grid.column_boundaries, grid.column_boundaries[1:]
                    )
                ]
                error_path = image_dir / "预处理致命错误.json"
                _json_dump(
                    error_path,
                    {
                        "image_name": image_path.name,
                        "stage": "物理网格切块",
                        "region_index": region_index,
                        "region_box": region_box.to_dict(),
                        "grid_source": grid.source,
                        "physical_shape": [grid.row_count, grid.column_count],
                        "largest_row_height": max(row_sizes, default=0),
                        "largest_column_width": max(column_sizes, default=0),
                        "max_allowed_side": self.config.max_vlm_side,
                        "reason": "无法只沿物理网格边界切出尺寸合规的原尺寸图片",
                        "boundary_image": str(
                            (
                                image_dir
                                / "grid_analysis"
                                / f"region_{region_index:03d}_boundaries.png"
                            ).resolve()
                        ),
                        "diagnostics": str(
                            (
                                image_dir
                                / "grid_analysis"
                                / f"region_{region_index:03d}_diagnostics.json"
                            ).resolve()
                        ),
                    },
                )
                raise TablePreprocessingError(
                    f"{image_path.name} 的表格区域 {region_index} 虽有 {grid.row_count}×{grid.column_count} 网格，"
                    f"但无法原尺寸切块；最大行高={max(row_sizes, default=0)}，"
                    f"最大列宽={max(column_sizes, default=0)}，上限={self.config.max_vlm_side}。"
                    f"这表示前置网格有误；诊断：{error_path.resolve()}"
                )
            for plan in plans:
                self._save_tile(
                    image_path,
                    tiles_dir / plan.file_name,
                    plan,
                    grid.row_boundaries,
                    grid.column_boundaries,
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
                    raw_column_boundaries=list(grid.raw_column_boundaries),
                    rejected_column_boundaries=list(
                        grid.rejected_column_boundaries
                    ),
                    cell_ink_mask=cell_ink_mask,
                    top_context=top_context,
                )
            )

        audit = self._save_tile_audit(preview, meta, regions, tiles_dir, image_dir)
        manifest = {
            "schema_version": 4,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": meta.to_dict(),
            "backend": self.backend.name,
            "detector": self.detector.name,
            "config": self.config.to_dict(),
            "regions": [region.to_dict() for region in regions],
            "audit": audit,
        }
        _json_dump(manifest_path, manifest)
        return manifest_path

    def prepare_directory(
        self,
        input_dir: str | Path,
        *,
        continue_on_error: bool = False,
    ) -> Path:
        """逐图断点准备；成功图复用，fatal 图重试并持续写中文汇总。"""

        paths = discover_images(input_dir)
        if not paths:
            raise RuntimeError(f"图表目录中没有图片：{input_dir}")
        groups = group_exact_duplicates(paths)
        items: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        sorted_groups = sorted(groups.items(), key=lambda pair: pair[1][0].name)

        def checkpoint(processed: int, *, complete: bool) -> Path:
            return 写入预处理检查点(
                work_dir=self.work_dir,
                branch="图表",
                input_dir=input_dir,
                config=self.config.to_dict(),
                config_digest=self.config.digest(),
                image_count=len(paths),
                unique_image_count=len(groups),
                duplicate_reuse_count=len(paths) - len(groups),
                processed_unique_count=processed,
                items=items,
                failures=failures,
                complete=complete,
            )

        checkpoint(0, complete=False)
        for group_index, (image_sha256, group) in enumerate(
            sorted_groups, start=1
        ):
            canonical = sorted(group, key=lambda path: path.name)[0]
            image_dir = self._single_image_directory(canonical, image_sha256)
            candidate_manifest = image_dir / "manifest.json"
            was_reused = (
                self._find_reusable_manifest(
                    candidate_manifest,
                    image_sha256,
                )
                is not None
            )
            print(
                f"[图表准备 {group_index:02d}/{len(sorted_groups):02d}] "
                f"{canonical.name}（同内容文件 {len(group)} 张）",
                flush=True,
            )
            try:
                manifest_path = self._prepare_one(canonical, image_sha256)
                chinese_artifacts = 建立中文中间产物(
                    manifest_path.parent,
                    "图表",
                    original_image=canonical,
                )
            except Exception as error:
                error_path = image_dir / "预处理致命错误.json"
                if not error_path.is_file():
                    _json_dump(
                        error_path,
                        {
                            "image_name": canonical.name,
                            "stage": "未知预处理阶段",
                            "reason": str(error),
                            "error_type": type(error).__name__,
                            "diagnostic_directory": str(image_dir.resolve()),
                        },
                    )
                chinese_artifacts = 建立中文中间产物(
                    image_dir,
                    "图表",
                    original_image=canonical,
                )
                failure = {
                    "canonical_file_name": canonical.name,
                    "file_names": [
                        path.name
                        for path in sorted(group, key=lambda item: item.name)
                    ],
                    "paths": [
                        str(path.resolve())
                        for path in sorted(group, key=lambda item: item.name)
                    ],
                    "sha256": image_sha256,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "chinese_artifacts": str(chinese_artifacts.resolve()),
                }
                failures.append(failure)
                checkpoint(group_index, complete=False)
                print(
                    f"[图表预处理fatal] {canonical.name}：{error}\n"
                    f"  中文中间产物：{chinese_artifacts.resolve()}\n"
                    f"  错误已写入汇总，继续下一张。",
                    flush=True,
                )
                if not continue_on_error:
                    raise
                continue

            for member in sorted(group, key=lambda item: item.name):
                items.append(
                    {
                        "file_name": member.name,
                        "path": str(member.resolve()),
                        "sha256": image_sha256,
                        "canonical_file_name": canonical.name,
                        "duplicate_of": (
                            None if member == canonical else canonical.name
                        ),
                        "image_manifest": str(manifest_path.resolve()),
                        "preprocessing_cache": (
                            "reused" if was_reused else "new"
                        ),
                        "chinese_artifacts": str(
                            chinese_artifacts.resolve()
                        ),
                    }
                )
            checkpoint(group_index, complete=False)

        output = checkpoint(len(sorted_groups), complete=True)
        reused = sum(
            1 for item in items if item["preprocessing_cache"] == "reused"
        )
        print(
            f"[图表预处理汇总] 成功 {len(items)}/{len(paths)}，"
            f"fatal {sum(len(item['file_names']) for item in failures)}，"
            f"逐图缓存复用 {reused}。\n"
            f"  HTML：{(self.work_dir / '预处理进度与错误汇总.html').resolve()}\n"
            f"  CSV：{(self.work_dir / '预处理进度与错误汇总.csv').resolve()}",
            flush=True,
        )
        return output

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

    def _physical_cell_ink_mask(
        self,
        gray: np.ndarray,
        row_boundaries: list[int],
        column_boundaries: list[int],
    ) -> list[list[bool]]:
        """R×C形成后，逐个物理格判断其内部是否存在真实墨迹。"""

        result: list[list[bool]] = []
        for row, (y1, y2) in enumerate(
            zip(row_boundaries, row_boundaries[1:])
        ):
            values: list[bool] = []
            for column, (x1, x2) in enumerate(
                zip(column_boundaries, column_boundaries[1:])
            ):
                # 向内缩约10%，避免把四周物理表格线记成单元格内容。
                inset_x = min(
                    max(1, (x2 - x1) // 10),
                    max(1, (x2 - x1 - 1) // 3),
                )
                inset_y = min(
                    max(1, (y2 - y1) // 10),
                    max(1, (y2 - y1 - 1) // 3),
                )
                inner = gray[
                    y1 + inset_y : max(y1 + inset_y + 1, y2 - inset_y),
                    x1 + inset_x : max(x1 + inset_x + 1, x2 - inset_x),
                ]
                ink_pixels = int(
                    np.count_nonzero(inner < self.config.grid_white_threshold)
                )
                values.append(ink_pixels > max(2, round(inner.size * 0.003)))
            result.append(values)
        return result

    @staticmethod
    def _slice_cell_ink_mask(
        cell_ink_mask: list[list[bool]],
        tile: TilePlan,
    ) -> list[list[bool]]:
        """从整张R×C墨迹矩阵中切出当前逻辑切片负责的子矩阵。"""

        row_indices = [
            *range(tile.header_context_rows),
            *range(tile.logical_row_start, tile.logical_row_end),
        ]
        column_indices = [
            *range(tile.stub_context_columns),
            *range(tile.logical_column_start, tile.logical_column_end),
        ]
        return [
            [bool(cell_ink_mask[row][column]) for column in column_indices]
            for row in row_indices
        ]
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
                values.append(ink_pixels > max(2, round(inner.size * 0.003)))
            result.append(values)
        return result

    def _recognize_top_context(
        self,
        manifest_path: Path,
        region: dict[str, Any],
        image_name: str,
        client: FinixDocClient,
        cache_model: str,
    ) -> str:
        """识别表格顶部候选区；失败只记警告，不阻断主体表格。"""

        context = region.get("top_context") or {}
        if not context.get("has_text"):
            return ""
        title_path = manifest_path.parent / str(
            context.get("recognition_file_name") or context["file_name"]
        )
        warning_path = (
            manifest_path.parent
            / "quality"
            / f"top_context_region_{int(region['index']):03d}_warning.json"
        )
        prompt = select_request_prompt(
            getattr(client, "protocol", CHAT_PROTOCOL),
            build_top_context_prompt,
        )
        title_model = f"{cache_model};top-context-v1"
        cache_key = self.cache.tile_key(
            title_path.read_bytes(),
            prompt,
            title_model,
        )
        cached = self.cache.get_tile_entry(cache_key)
        if cached is not None:
            warning_path.unlink(missing_ok=True)
            return cached[0].strip()

        label = f"原图 {image_name} / 区域 {int(region['index']) + 1} / 顶部候选标题"
        try:
            if isinstance(client, FinixDocClient):
                text = client.recognize(
                    title_path,
                    prompt,
                    request_label=label,
                    empty_retry_limit=min(3, client.max_retries),
                    return_empty_after_limit=True,
                )
            else:
                text = client.recognize(title_path, prompt)
        except Exception as error:
            _json_dump(
                warning_path,
                {
                    "source_image": image_name,
                    "region_index": region["index"],
                    "error": str(error),
                    "policy": "标题失败不阻断主体表格",
                },
            )
            print(
                f"[图表标题警告] {label}：{error}；继续主体表格。",
                flush=True,
            )
            return ""

        text = str(text or "").strip()
        warning_path.unlink(missing_ok=True)
        response_path = (
            manifest_path.parent
            / "responses"
            / "top_context"
            / f"region_{int(region['index']):03d}.md"
        )
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(text, encoding="utf-8")
        self.cache.put_tile(
            cache_key,
            text,
            {
                "source_image": image_name,
                "region_index": region["index"],
                "response_status": "empty" if not text else "valid",
                "context": True,
                "model": title_model,
            },
        )
        return text

    def _recognize_manifest(
        self,
        manifest_path: Path,
        client: FinixDocClient,
        *,
        allow_degraded_output: bool = False,
    ) -> str:
        image_manifest = _load_json(manifest_path)
        image_info = image_manifest["image"]
        image_name = str(image_info.get("file_name") or Path(image_info["path"]).name)
        region_markdowns: list[str] = []
        tile_failures: list[dict[str, Any]] = []
        degraded_parts: list[dict[str, Any]] = []
        cache_model = _table_cache_model(client)
        total_tiles = sum(len(region["tiles"]) for region in image_manifest["regions"])
        completed_tiles = 0
        source_gray = None
        if any(
            not region.get("cell_ink_mask")
            for region in image_manifest["regions"]
        ):
            # 仅兼容旧manifest；新版预处理已把整张R×C墨迹矩阵写入清单。
            with Image.open(image_manifest["image"]["path"]) as source:
                source_gray = np.asarray(
                    source.convert("L"),
                    dtype=np.uint8,
                )
        for region in image_manifest["regions"]:
            contents: dict[tuple[int, int], str] = {}
            plans = [self._tile_from_dict(raw) for raw in region["tiles"]]
            top_context_markdown = self._recognize_top_context(
                manifest_path, region, image_name, client, cache_model
            )
            row_boundaries = [int(value) for value in region.get("row_boundaries", [])]
            column_boundaries = [
                int(value) for value in region.get("column_boundaries", [])
            ]
            region_cell_ink_mask = region.get("cell_ink_mask") or []
            if region_cell_ink_mask:
                tile_ink_masks = {
                    (plan.row_index, plan.column_index): self._slice_cell_ink_mask(
                        region_cell_ink_mask,
                        plan,
                    )
                    for plan in plans
                }
            elif (
                source_gray is not None
                and row_boundaries
                and column_boundaries
            ):
                tile_ink_masks = {
                    (plan.row_index, plan.column_index): self._tile_ink_mask(
                        source_gray,
                        row_boundaries,
                        column_boundaries,
                        plan,
                    )
                    for plan in plans
                }
            else:
                tile_ink_masks = {}
            response_dir = manifest_path.parent / "responses"
            response_dir.mkdir(parents=True, exist_ok=True)
            for raw_tile in region["tiles"]:
                tile = self._tile_from_dict(raw_tile)
                tile_id = (tile.row_index, tile.column_index)
                tile_path = manifest_path.parent / "tiles" / tile.file_name
                request_label = (
                    f"原图 {image_name} / 区域 {region['index'] + 1} / "
                    f"切块 {tile.file_name}"
                )
                # 官方 multipart 只上传图片和三个文本字段，不生成、不发送 prompt。
                prompt = select_request_prompt(
                    getattr(client, "protocol", CHAT_PROTOCOL),
                    lambda: build_table_prompt(tile),
                )
                image_bytes = tile_path.read_bytes()
                cache_key = self.cache.tile_key(image_bytes, prompt, cache_model)
                base_metadata = {
                    "source_image": image_name,
                    "tile": tile.to_dict(),
                    "model": cache_model,
                    "prompt_version": PROMPT_VERSION,
                    "tile_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "cache_key": cache_key,
                }
                ink_mask = tile_ink_masks.get(tile_id)

                if tile.tiling_mode == "logical_grid" and _mask_has_no_text(ink_mask):
                    # 墨迹 bool 只判断内容是否为空，不改变预处理确定的物理行列。
                    markdown = _empty_tile_html(tile)
                    source = "预处理判空，跳过模型"
                    self.cache.put_tile(
                        cache_key,
                        markdown,
                        {
                            **base_metadata,
                            "response_status": "preprocess-confirmed-empty",
                            "empty_table_fallback": True,
                            "empty_table_reason": "preprocess-no-cell-ink",
                        },
                    )
                else:
                    markdown: str | None = None
                    quality_report: dict[str, object] | None = None
                    forced_degradation = False
                    cached_entry = self.cache.get_tile_entry(cache_key)
                    cached = cached_entry[0] if cached_entry else None
                    cached_metadata = cached_entry[1] if cached_entry else {}
                    if cached is not None:
                        if not cached.strip():
                            # 历史空字符串没有“三次复核”状态，删除后重新识别。
                            self.cache.delete_tile(cache_key)
                            print(
                                f"[图表坏缓存失效] {request_label}："
                                "历史缓存为空字符串，重新识别。",
                                flush=True,
                            )
                        elif tile.tiling_mode == "logical_grid":
                            try:
                                markdown, quality_report = (
                                    normalize_table_response_soft(
                                        cached, *_logical_tile_shape(tile), ink_mask
                                    )
                                )
                                if (
                                    _mask_has_any_text(ink_mask)
                                    and quality_report["nonempty_cells"] == 0
                                    and cached_metadata.get("response_status")
                                    not in {
                                        "empty-fallback",
                                        "preprocess-confirmed-empty",
                                        *(
                                            ("error-fallback",)
                                            if allow_degraded_output
                                            else ()
                                        ),
                                    }
                                ):
                                    raise HtmlTableMergeError(
                                        "历史缓存是全空表，但没有经过三次全空复核"
                                    )
                                source = "缓存"
                            except HtmlTableMergeError as error:
                                self.cache.delete_tile(cache_key)
                                print(
                                    f"[图表坏缓存失效] {request_label}："
                                    f"{error}；重新识别。",
                                    flush=True,
                                )
                        else:
                            markdown = cached
                            source = "缓存"

                    if markdown is None:
                        # 旧缓存也必须通过当前质量闸门，坏结果不迁移。
                        for legacy_model in _table_legacy_cache_models(client):
                            legacy_key = self.cache.tile_key(
                                image_bytes, prompt, legacy_model
                            )
                            legacy_entry = self.cache.get_tile_entry(legacy_key)
                            candidate = legacy_entry[0] if legacy_entry else None
                            legacy_metadata = legacy_entry[1] if legacy_entry else {}
                            if candidate is None or not candidate.strip():
                                continue
                            legacy_quality = None
                            if tile.tiling_mode == "logical_grid":
                                try:
                                    candidate, legacy_quality = (
                                        normalize_table_response_soft(
                                            candidate,
                                            *_logical_tile_shape(tile),
                                            ink_mask,
                                        )
                                    )
                                    if (
                                        _mask_has_any_text(ink_mask)
                                        and legacy_quality["nonempty_cells"] == 0
                                        and legacy_metadata.get("response_status")
                                        not in {
                                            "empty-fallback",
                                            "preprocess-confirmed-empty",
                                            *(
                                                ("error-fallback",)
                                                if allow_degraded_output
                                                else ()
                                            ),
                                        }
                                    ):
                                        continue
                                except HtmlTableMergeError:
                                    continue
                            markdown = candidate
                            quality_report = legacy_quality
                            source = "兼容缓存"
                            self.cache.put_tile(
                                cache_key,
                                markdown,
                                {
                                    **base_metadata,
                                    "response_status": "valid-migrated-cache",
                                    "migrated_from": legacy_model,
                                    "quality": legacy_quality,
                                },
                            )
                            break

                    if markdown is None:
                        # 原始响应不等于缓存。只有旁边的元数据证明图片字节、
                        # 模型和提示词完全一致时，才从新到旧逐个尝试恢复。
                        raw_dir = response_dir / "模型原始"
                        for raw_path in _raw_response_candidates(
                            raw_dir, Path(tile.file_name).stem
                        ):
                            metadata_path = raw_path.with_suffix(".json")
                            try:
                                raw_metadata = _load_json(metadata_path)
                                raw = raw_path.read_text(encoding="utf-8")
                            except (OSError, json.JSONDecodeError):
                                continue
                            if not _raw_metadata_matches(raw_metadata, base_metadata):
                                continue
                            if raw_metadata.get("response_sha256") != hashlib.sha256(
                                raw.encode("utf-8")
                            ).hexdigest():
                                continue
                            if not raw.strip() or tile.tiling_mode != "logical_grid":
                                continue
                            try:
                                recovered, recovered_quality = (
                                    normalize_table_response_soft(
                                        raw, *_logical_tile_shape(tile), ink_mask
                                    )
                                )
                            except HtmlTableMergeError:
                                continue
                            if (
                                _mask_has_any_text(ink_mask)
                                and recovered_quality["nonempty_cells"] == 0
                            ):
                                continue
                            markdown = recovered
                            quality_report = recovered_quality
                            source = "已校验的历史原始响应"
                            self.cache.put_tile(
                                cache_key,
                                markdown,
                                {
                                    **base_metadata,
                                    "response_status": "recovered",
                                    "recovered_from": str(raw_path.resolve()),
                                    "quality": quality_report,
                                },
                            )
                            print(
                                f"[图表恢复 {completed_tiles + 1:02d}/{total_tiles:02d}] "
                                f"{request_label}：{source}",
                                flush=True,
                            )
                            break

                    if markdown is None:
                        print(
                            f"[图表识别 {completed_tiles + 1:02d}/{total_tiles:02d}] "
                            f"{request_label}：请求模型",
                            flush=True,
                        )
                        retry_limit = 3 if tile.tiling_mode == "logical_grid" else 0
                        empty_reason: str | None = None
                        last_error: Exception | None = None
                        raw_dir = response_dir / "模型原始"
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        last_raw_path: Path | None = None

                        for attempt in range(retry_limit + 1):
                            try:
                                if isinstance(client, FinixDocClient):
                                    raw = client.recognize(
                                        tile_path,
                                        prompt,
                                        request_label=request_label,
                                        empty_retry_limit=min(3, client.max_retries),
                                        return_empty_after_limit=True,
                                    )
                                else:
                                    raw = client.recognize(tile_path, prompt)
                            except RuntimeError as error:
                                # 本地后端用异常表示“正常完成但输出为空”。
                                if (
                                    tile.tiling_mode == "logical_grid"
                                    and "空 Markdown" in str(error)
                                ):
                                    raw = ""
                                else:
                                    last_error = error
                                    break

                            last_raw_path = (
                                raw_dir
                                / f"{Path(tile.file_name).stem}_attempt_{attempt + 1}.md"
                            )
                            last_raw_path.write_text(raw, encoding="utf-8")
                            _json_dump(
                                last_raw_path.with_suffix(".json"),
                                {
                                    **base_metadata,
                                    "attempt": attempt + 1,
                                    "response_sha256": hashlib.sha256(
                                        raw.encode("utf-8")
                                    ).hexdigest(),
                                    "saved_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )

                            if not raw.strip():
                                empty_reason = "model-empty-markdown"
                                # 官方客户端内部已经完成三次平方退让。
                                if isinstance(client, FinixDocClient):
                                    break
                                if attempt < retry_limit:
                                    print(
                                        f"[图表全空复核 {attempt + 1}/{retry_limit}] "
                                        f"{request_label}：没有 HTML/Markdown，"
                                        "再试一次。",
                                        flush=True,
                                    )
                                    continue
                                break

                            if tile.tiling_mode != "logical_grid":
                                markdown = raw
                                source = "模型"
                                break

                            try:
                                normalized, candidate_quality = (
                                    normalize_table_response_soft(
                                        raw, *_logical_tile_shape(tile), ink_mask
                                    )
                                )
                            except HtmlTableMergeError as error:
                                if _is_truncated_empty_html(raw):
                                    empty_reason = "model-empty-truncated-html"
                                    last_error = None
                                else:
                                    last_error = error
                                if attempt < retry_limit:
                                    print(
                                        f"[图表内容异常复核 {attempt + 1}/{retry_limit}] "
                                        f"{request_label}：{error}；换一次请求重试。",
                                        flush=True,
                                    )
                                    continue
                                break

                            if (
                                _mask_has_any_text(ink_mask)
                                and candidate_quality["nonempty_cells"] == 0
                            ):
                                # 这可能是墨迹 bool 约 0.1% 的误判，也可能是模型漏识。
                                empty_reason = "model-empty-table"
                                last_error = None
                                if attempt < retry_limit:
                                    print(
                                        f"[图表全空复核 {attempt + 1}/{retry_limit}] "
                                        f"{request_label}：返回全空表但墨迹 bool 有内容，"
                                        "再试一次。",
                                        flush=True,
                                    )
                                    continue
                                break

                            markdown = normalized
                            quality_report = candidate_quality
                            source = "模型"
                            empty_reason = None
                            last_error = None
                            break

                        if markdown is None and empty_reason is not None:
                            if tile.tiling_mode == "logical_grid":
                                markdown = _empty_tile_html(tile)
                                source = "模型连续全空，按预处理补空表"
                                rows, columns = _logical_tile_shape(tile)
                                quality_report = {
                                    "physical_rows": rows,
                                    "physical_columns": columns,
                                    "warnings": [
                                        "模型复核后仍为全空；保留物理结构并清空内容"
                                    ],
                                }
                            else:
                                last_error = RuntimeError(
                                    "全空切片没有可靠预处理行列，无法补矩阵"
                                )

                        if markdown is None:
                            error = last_error or RuntimeError(
                                "模型响应未通过图表质量检查"
                            )
                            failure = {
                                "source_image": image_name,
                                "region_index": region["index"],
                                "tile_file_name": tile.file_name,
                                "tile_path": str(tile_path.resolve()),
                                "raw_response": (
                                    str(last_raw_path.resolve())
                                    if last_raw_path
                                    else None
                                ),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                            if allow_degraded_output and tile.tiling_mode == "logical_grid":
                                markdown = _empty_tile_html(tile)
                                forced_degradation = True
                                source = "识别失败，按预处理物理结构补空表"
                                rows, columns = _logical_tile_shape(tile)
                                quality_report = {
                                    "physical_rows": rows,
                                    "physical_columns": columns,
                                    "warnings": [
                                        "识别重试耗尽；该切块内容已降级为空，物理结构保留"
                                    ],
                                }
                                failure["fallback"] = "按预处理 R×C 生成全空切块"
                                failure["degraded_output"] = True
                                degraded_parts.append(failure)
                                print(
                                    f"[图表强制补全 {completed_tiles + 1:02d}/{total_tiles:02d}] "
                                    f"{request_label}：{error}；保留 R×C 并清空该块内容。",
                                    flush=True,
                                )
                            else:
                                tile_failures.append(failure)
                                completed_tiles += 1
                                print(
                                    f"[图表切块失败 {completed_tiles:02d}/{total_tiles:02d}] "
                                    f"{request_label}：{error}；不缓存损坏响应，"
                                    "继续此图下一块。",
                                    flush=True,
                                )
                                continue

                        metadata = {
                            **base_metadata,
                            "response_status": (
                                "error-fallback"
                                if forced_degradation
                                else ("empty-fallback" if empty_reason else "valid")
                            ),
                            "quality": quality_report,
                        }
                        if forced_degradation:
                            metadata.update(
                                {
                                    "degraded_output": True,
                                    "degraded_reason": degraded_parts[-1]["error"],
                                }
                            )
                        if empty_reason is not None:
                            metadata.update(
                                {
                                    "empty_table_fallback": True,
                                    "empty_table_reason": empty_reason,
                                    "empty_retry_limit": 3,
                                }
                            )
                        self.cache.put_tile(cache_key, markdown, metadata)

                (response_dir / f"{Path(tile.file_name).stem}.md").write_text(
                    markdown, encoding="utf-8"
                )
                completed_tiles += 1
                print(
                    f"[图表识别 {completed_tiles:02d}/{total_tiles:02d}] "
                    f"{request_label}：{source}完成，结果 {len(markdown)} 字符",
                    flush=True,
                )
                contents[tile_id] = markdown

            if len(contents) != len(plans):
                print(
                    f"[图表暂缓合并] 原图 {image_name} / "
                    f"区域 {region['index'] + 1}：存在失败切块，继续下一区域。",
                    flush=True,
                )
                continue
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
                            contents,
                            plans,
                            len(region["row_boundaries"]) - 1,
                            len(region["column_boundaries"]) - 1,
                            tile_ink_masks,
                        )
                    _json_dump(
                        manifest_path.parent
                        / "quality"
                        / f"region_{region['index']:03d}.json",
                        quality,
                    )
                    for warning in quality.get("warnings", []):
                        print(
                            f"[图表警告] 原图 {image_name} / "
                            f"区域 {region['index'] + 1}：{warning}",
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
                    "source_image": image_name,
                    "region_index": region["index"],
                    "error": str(error),
                    "fallback": "禁止损坏响应进入最终结果，区域标记为不完整",
                    "response_files": [
                        f"responses/{Path(plan.file_name).stem}.md" for plan in plans
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
                        "source_image": image_name,
                        "region_index": region["index"],
                        "error": str(error),
                        "responses": {
                            f"r{row}_c{column}": text
                            for (row, column), text in contents.items()
                        },
                    },
                )
                merge_failure = {
                    "source_image": image_name,
                    "region_index": region["index"],
                    "tile_file_name": "<区域合并>",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                if allow_degraded_output and row_boundaries and column_boundaries:
                    region_markdown = render_empty_table(
                        len(row_boundaries) - 1,
                        len(column_boundaries) - 1,
                    )
                    merge_failure["fallback"] = "按区域完整 R×C 生成全空表"
                    merge_failure["degraded_output"] = True
                    degraded_parts.append(merge_failure)
                    print(
                        f"[图表区域强制补全] 原图 {image_name} / "
                        f"区域 {region['index'] + 1}：{error}；"
                        "保留整区物理结构并清空内容。",
                        flush=True,
                    )
                else:
                    tile_failures.append(merge_failure)
                    print(
                        f"[图表区域损坏] 原图 {image_name} / "
                        f"区域 {region['index'] + 1} 无法可靠合并："
                        f"{error}；禁止把损坏响应拼入最终结果，继续下一区域。",
                        flush=True,
                    )
                    continue
            region_text = region_markdown.strip()
            if top_context_markdown:
                region_text = f"{top_context_markdown.strip()}\n\n{region_text}"
            region_markdowns.append(region_text)
        part_failure_path = manifest_path.parent / "recognition_failures.json"
        _json_dump(
            part_failure_path,
            {
                "status": (
                    "incomplete"
                    if tile_failures
                    else ("degraded" if degraded_parts else "ok")
                ),
                "source_image": image_name,
                "failure_count": len(tile_failures),
                "failed_parts": tile_failures,
                "degraded_count": len(degraded_parts),
                "degraded_parts": degraded_parts,
            },
        )
        if tile_failures:
            raise IncompleteImageRecognitionError(
                f"原图 {image_name} 有 {len(tile_failures)} 个切块失败；"
                f"详情：{part_failure_path}",
                tile_failures,
            )

        return "\n\n".join(text for text in region_markdowns if text).strip()

    def recognize_dataset(
        self,
        dataset_manifest_path: str | Path,
        client: FinixDocClient,
        output_csv: str | Path,
        max_workers: int = 1,
        allow_degraded_output: bool = False,
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
                + "\0degraded="
                + str(allow_degraded_output)
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
            cached = self.cache.get_image(item["sha256"], recognition_digest)
            if cached is None:
                image_manifest_path = Path(item["image_manifest"])
                cached = (
                    self._recognize_manifest(
                        image_manifest_path,
                        client,
                        allow_degraded_output=True,
                    )
                    if allow_degraded_output
                    else self._recognize_manifest(image_manifest_path, client)
                )
                title_warnings = list(
                    (image_manifest_path.parent / "quality").glob(
                        "top_context_region_*_warning.json"
                    )
                )
                if not title_warnings:
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
                else:
                    print(
                        f"[图表整图暂不缓存] 原图 {canonical_name}："
                        "顶部候选标题识别失败，主体结果仍正常输出；下次只重试标题。",
                        flush=True,
                    )
            return canonical_name, cached

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
                if isinstance(error, IncompleteImageRecognitionError):
                    failure["failed_parts"] = error.failed_parts
                if allow_degraded_output:
                    fallback = ""
                    failure["degraded_output"] = True
                    failure["fallback"] = "整图异常，提交结果降级为空字符串"
                    self.cache.put_image(
                        item["sha256"],
                        recognition_digest,
                        fallback,
                        {
                            "canonical_file_name": canonical_name,
                            "model": cache_model,
                            "prompt_version": PROMPT_VERSION,
                            "degraded_output": True,
                            "degraded_reason": str(error),
                        },
                    )
                    print(
                        f"[图表整图强制补全] 原图 {canonical_name}：{error}；"
                        "结果降级为空字符串，继续生成完整 CSV。",
                        flush=True,
                    )
                    return canonical_name, fallback, failure
                print(
                    f"[图表失败] 原图 {canonical_name}：{error}；"
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
                f"[图表并行] {worker_count} 个任务并发识别 "
                f"{len(canonical_items)} 张唯一图片",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                outcomes = list(executor.map(recognize_safely, canonical_items))
        canonical_results = {
            name: markdown for name, markdown, _ in outcomes if markdown is not None
        }
        failures = [failure for _, _, failure in outcomes if failure is not None]
        degraded_images: list[dict[str, Any]] = []
        for item in canonical_items:
            report_path = (
                Path(item["image_manifest"]).parent / "recognition_failures.json"
            )
            try:
                report = _load_json(report_path)
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("status") == "degraded":
                degraded_images.append(
                    {
                        "canonical_file_name": item["canonical_file_name"],
                        "report_path": str(report_path.resolve()),
                        "degraded_count": int(report.get("degraded_count", 0)),
                        "degraded_parts": report.get("degraded_parts", []),
                    }
                )
        for failure in failures:
            if failure.get("degraded_output"):
                degraded_images.append(
                    {
                        "canonical_file_name": failure["canonical_file_name"],
                        "report_path": failure.get("image_manifest"),
                        "degraded_count": 1,
                        "degraded_parts": [failure],
                    }
                )

        results = {
            item["file_name"]: canonical_results[item["canonical_file_name"]]
            for item in dataset_manifest["items"]
            if item["canonical_file_name"] in canonical_results
        }
        results_dir = self.work_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for file_name, markdown in results.items():
            result_path = results_dir / f"{Path(file_name).stem}.md"
            result_path.write_text(markdown, encoding="utf-8")

        failure_report_path = self.work_dir / "recognition_failures.json"
        _json_dump(
            failure_report_path,
            {
                "status": (
                    "degraded"
                    if degraded_images or (failures and allow_degraded_output)
                    else ("incomplete" if failures else "ok")
                ),
                "allow_degraded_output": allow_degraded_output,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "failure_count": len(failures),
                "degraded_image_count": len(degraded_images),
                "successful_unique_images": len(canonical_results),
                "failures": failures,
                "degraded_images": degraded_images,
            },
        )
        if failures and not allow_degraded_output:
            partial_csv = self.work_dir / "partial_results.csv"
            write_submission(results, partial_csv)
            failed_names = ", ".join(
                failure["canonical_file_name"] for failure in failures
            )
            raise RuntimeError(
                f"图表识别跑完其他图片后仍有 {len(failures)} 张失败；"
                f"失败原图：{failed_names}；详情：{failure_report_path}"
            )
        write_submission(results, output_csv)
        return results
