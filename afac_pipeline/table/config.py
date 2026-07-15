"""配置加载与校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TableConfig:
    """图表分支配置。

    detector=auto 时使用低清墨水密度和连通外轮廓，不加载模型。
    yolo 与 projection 只保留为历史对比模式。
    """

    backend: str = "auto"
    detector: str = "auto"
    yolo_model_path: str = "360LayoutAnalysis/weights/report-8n.pt"
    yolo_confidence: float = 0.25
    yolo_imgsz: int = 1280
    preview_max_side: int = 1800
    max_vlm_side: int = 3900
    tile_overlap: int = 160
    single_tile_min_scale: float = 0.65
    table_box_padding_ratio: float = 0.015
    ink_coarse_max_side: int = 384
    ink_threshold: int = 245
    ink_minimum_density: float = 0.008
    ink_blur_ratio: float = 0.012
    ink_closing_ratio: float = 0.018
    ink_minimum_box_area_ratio: float = 0.01
    grid_analysis_max_side: int = 2400
    grid_white_threshold: int = 225
    grid_line_min_ratio: float = 0.42
    grid_min_line_count: int = 2
    grid_min_cell_size: int = 18
    whitespace_blank_ratio: float = 0.002
    whitespace_min_band: int = 8
    whitespace_dilate_ratio: float = 0.008
    whitespace_horizontal_dilate_ratio: float | None = None
    whitespace_vertical_dilate_ratio: float | None = None
    repeat_header_rows: int = 1
    repeat_stub_columns: int = 1
    projection_threshold: int = 225
    projection_min_line_ratio: float = 0.22
    projection_min_lines: int = 3
    projection_max_line_gap_ratio: float = 0.10
    pipeline_version: str = "table-v3-model-free-region"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "pillow", "vips"}:
            raise ValueError("backend 只能是 auto、pillow 或 vips")
        if self.detector not in {"auto", "ink", "projection", "yolo"}:
            raise ValueError("detector 只能是 auto、ink、projection 或 yolo")
        if not 512 <= self.preview_max_side <= 4096:
            raise ValueError("preview_max_side 应位于 512 到 4096 之间")
        if not 512 <= self.max_vlm_side <= 4096:
            raise ValueError("max_vlm_side 应位于 512 到 4096 之间")
        if not 0 <= self.tile_overlap < self.max_vlm_side:
            raise ValueError("tile_overlap 必须小于 max_vlm_side")
        if not 0 < self.single_tile_min_scale <= 1:
            raise ValueError("single_tile_min_scale 必须位于 (0, 1] 内")
        if not 64 <= self.ink_coarse_max_side <= 2048:
            raise ValueError("ink_coarse_max_side 应位于 64 到 2048 之间")
        if not 0 <= self.ink_threshold <= 255:
            raise ValueError("ink_threshold 必须位于 0 到 255 之间")
        if not 0 < self.ink_minimum_density < 1:
            raise ValueError("ink_minimum_density 必须位于 (0, 1) 内")
        if self.ink_blur_ratio <= 0 or self.ink_closing_ratio <= 0:
            raise ValueError("墨水模糊和闭运算比例必须大于 0")
        if not 0 < self.ink_minimum_box_area_ratio <= 1:
            raise ValueError("ink_minimum_box_area_ratio 必须位于 (0, 1] 内")
        if not 512 <= self.grid_analysis_max_side <= 4096:
            raise ValueError("grid_analysis_max_side 应位于 512 到 4096 之间")
        if not 0 <= self.grid_white_threshold <= 255:
            raise ValueError("grid_white_threshold 必须位于 0 到 255 之间")
        if not 0 < self.grid_line_min_ratio <= 1:
            raise ValueError("grid_line_min_ratio 必须位于 (0, 1] 内")
        if self.grid_min_line_count < 2 or self.grid_min_cell_size <= 0:
            raise ValueError("网格线数量和最小单元格尺寸配置不合法")
        if not 0 <= self.whitespace_blank_ratio < 1:
            raise ValueError("whitespace_blank_ratio 必须位于 [0, 1) 内")
        if self.whitespace_min_band <= 0 or self.whitespace_dilate_ratio <= 0:
            raise ValueError("空白带最小宽度和墨水扩张比例必须大于 0")
        if (
            self.whitespace_horizontal_dilate_ratio is not None
            and self.whitespace_horizontal_dilate_ratio <= 0
        ):
            raise ValueError("横向文字扩张比例必须大于 0")
        if (
            self.whitespace_vertical_dilate_ratio is not None
            and self.whitespace_vertical_dilate_ratio <= 0
        ):
            raise ValueError("纵向文字扩张比例必须大于 0")
        if self.repeat_header_rows < 0 or self.repeat_stub_columns < 0:
            raise ValueError("重复表头行数和行名列数不能为负数")

    @classmethod
    def from_json(cls, path: str | Path | None) -> "TableConfig":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as file:
            raw: dict[str, Any] = json.load(file)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"配置中存在未知字段：{', '.join(unknown)}")
        return cls(**raw)

    def digest(self) -> str:
        """缓存键使用稳定配置摘要，修改配置后不会误用旧结果。"""

        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
