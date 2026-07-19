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
    # v6 固定使用原图 20% 做边界分析，再把分析图缩到 25%，即原图 5%，
    # 生成用于分开同图异表的低清密度图。固定比例可避免同一套像素参数随
    # 原图尺寸变化而漂移。
    table_analysis_scale: float = 0.20
    # 白带仍使用上面的 20% 分析图；只有容易被缩灰、缩断的黑表格线
    # 单独回到分表后的原图区域，以 50% 分辨率检测。
    table_black_line_scale: float = 0.50
    table_black_analysis_max_side: int = 12000
    table_density_scale: float = 0.25
    table_analysis_max_side: int = 4608
    max_vlm_side: int = 3900
    tile_overlap: int = 160
    # 仅为兼容旧 manifest/config 保留。v13 起请求图片严禁缩放，这个值
    # 不再参与任何切块决策；新配置文件不必再填写。
    single_tile_min_scale: float = 0.65
    table_box_padding_ratio: float = 0.015
    ink_coarse_max_side: int = 384
    ink_threshold: int = 225
    ink_minimum_density: float = 0.008
    ink_blur_ratio: float = 0.012
    ink_closing_ratio: float = 0.018
    ink_minimum_box_area_ratio: float = 0.01
    grid_analysis_max_side: int = 4096
    grid_white_threshold: int = 225
    grid_line_min_ratio: float = 0.90
    grid_min_line_count: int = 5
    grid_black_line_ratio: float = 0.90
    # 短表中同列数字“1”的竖笔画可能恰好超过 90%。竖线因此改用更严格的
    # 中段覆盖率，并要求候选线明显深于左右邻域；横线仍沿用上面的 90%。
    grid_black_column_line_ratio: float = 0.95
    grid_black_column_endpoint_trim_ratio: float = 0.05
    grid_black_column_min_contrast: float = 30.0
    # 竖线若已有至少 98% 的绝对黑像素覆盖率，本身足以说明它是一条连续
    # 物理线，此时不再让容易受邻近底色影响的灰度对比规则误杀。
    grid_black_column_contrast_bypass_ratio: float = 0.98
    grid_reliable_line_count: int = 5
    # 黑线不能只靠“数量够多”就成为物理网格：至少要有一条边界处在表格
    # 中部，且任何一个逻辑格都不能独占所在方向绝大部分长度。
    grid_interior_margin_ratio: float = 0.10
    grid_max_cell_span_ratio: float = 0.95
    grid_min_cell_size: int = 18
    whitespace_blank_ratio: float = 0.01
    whitespace_min_band: int = 1
    whitespace_dilate_ratio: float = 0.004
    whitespace_horizontal_dilate_ratio: float | None = 0.0015
    whitespace_vertical_dilate_ratio: float | None = 0.004
    repeat_header_rows: int = 0
    repeat_stub_columns: int = 0
    # 空单元格也会输出 HTML 标签，因此按逻辑总格子数限制输出规模。
    # 图片尺寸能装下，不代表几千个单元格也能一次写完。
    # 280 格会拆开已知会截断的 304 格复杂表，又比 240 格少很多请求。
    max_logical_cells_per_tile: int = 280
    # 这是规划优选目标，不是硬限制；整张小表仍可少于 80 格。
    preferred_min_logical_cells_per_tile: int = 80
    # 极端细长块即使格子不多，也继续沿长边拆分以保留表格语义。
    max_tile_aspect_ratio: float = 8.0
    projection_threshold: int = 225
    projection_min_line_ratio: float = 0.22
    projection_min_lines: int = 3
    projection_max_line_gap_ratio: float = 0.10
    pipeline_version: str = "table-v15-98-percent-bypasses-column-contrast"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "pillow", "vips"}:
            raise ValueError("backend 只能是 auto、pillow 或 vips")
        if self.detector not in {"auto", "ink", "projection", "yolo"}:
            raise ValueError("detector 只能是 auto、ink、projection 或 yolo")
        if not 512 <= self.preview_max_side <= 4096:
            raise ValueError("preview_max_side 应位于 512 到 4096 之间")
        if not 0 < self.table_analysis_scale <= 1:
            raise ValueError("table_analysis_scale 必须位于 (0, 1] 内")
        if not 0 < self.table_black_line_scale <= 1:
            raise ValueError("table_black_line_scale 必须位于 (0, 1] 内")
        if self.table_black_line_scale < self.table_analysis_scale:
            raise ValueError("黑线分析比例不能低于白带分析比例")
        if not 512 <= self.table_black_analysis_max_side <= 20000:
            raise ValueError("黑线分析图安全上限应位于 512 到 20000 之间")
        if not 0 < self.table_density_scale <= 1:
            raise ValueError("table_density_scale 必须位于 (0, 1] 内")
        if not 512 <= self.table_analysis_max_side <= 8192:
            raise ValueError("table_analysis_max_side 应位于 512 到 8192 之间")
        if not 512 <= self.max_vlm_side <= 4096:
            raise ValueError("max_vlm_side 应位于 512 到 4096 之间")
        if not 0 <= self.tile_overlap < self.max_vlm_side:
            raise ValueError("tile_overlap 必须小于 max_vlm_side")
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
        if not 0 < self.grid_black_line_ratio <= 1:
            raise ValueError("grid_black_line_ratio 必须位于 (0, 1] 内")
        if not 0 < self.grid_black_column_line_ratio <= 1:
            raise ValueError("grid_black_column_line_ratio 必须位于 (0, 1] 内")
        if not (
            self.grid_black_column_line_ratio
            <= self.grid_black_column_contrast_bypass_ratio
            <= 1
        ):
            raise ValueError("竖线免灰度对比覆盖率必须位于竖线阈值和 1 之间")
        if not 0 <= self.grid_black_column_endpoint_trim_ratio < 0.5:
            raise ValueError(
                "grid_black_column_endpoint_trim_ratio 必须位于 [0, 0.5) 内"
            )
        if self.grid_black_column_min_contrast < 0:
            raise ValueError("grid_black_column_min_contrast 不能小于 0")
        if self.grid_reliable_line_count < 1:
            raise ValueError("grid_reliable_line_count 至少为 1")
        if not 0 <= self.grid_interior_margin_ratio < 0.5:
            raise ValueError("网格内部边界留白比例必须位于 [0, 0.5) 内")
        if not 0 < self.grid_max_cell_span_ratio < 1:
            raise ValueError("最大单元格跨度比例必须位于 (0, 1) 内")
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
        if self.max_logical_cells_per_tile < 32:
            raise ValueError("单个图表切片的逻辑单元格上限至少为 32")
        if not (
            1
            <= self.preferred_min_logical_cells_per_tile
            <= self.max_logical_cells_per_tile
        ):
            raise ValueError("优选最小逻辑格数必须位于 1 和单块上限之间")
        if self.max_tile_aspect_ratio < 1:
            raise ValueError("逻辑切片最大宽高比不能小于 1")

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

        config = asdict(self)
        config.pop("single_tile_min_scale", None)
        payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        config = asdict(self)
        config.pop("single_tile_min_scale", None)
        return config
