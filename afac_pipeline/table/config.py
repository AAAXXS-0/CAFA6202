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
    # 正式流程固定使用整图20%预览做横向分表。分表只允许上下切；
    # 每张子表的黑线与白缝划线另行统一使用50%分析图。
    table_analysis_scale: float = 0.20
    # 分表后的每张子表只生成这一张50%分析图，黑线和白缝共用。
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
    # 横向分表后，用固定强度二维晕染连接同一张表，只保留最大主体。
    # 外接框宁可略带空白，也不能裁掉边缘稀疏数据。
    analysis_box_smear_horizontal_ratio: float = 0.06
    analysis_box_smear_vertical_ratio: float = 0.04
    analysis_box_close_iterations: int = 2
    analysis_box_padding_ratio: float = 0.03
    ink_coarse_max_side: int = 384
    ink_threshold: int = 225
    ink_minimum_density: float = 0.008
    ink_blur_ratio: float = 0.012
    ink_closing_ratio: float = 0.018
    ink_minimum_box_area_ratio: float = 0.01
    grid_analysis_max_side: int = 4096
    grid_white_threshold: int = 225
    # 分析框顶部到第一根横线之间单独保存为候选标题区。它不参与
    # 物理网格，只在最终表格 Markdown 前输出一次。
    top_context_enabled: bool = True
    top_context_line_guard_px: int = 4
    top_context_min_ink_ratio: float = 0.00002
    grid_line_min_ratio: float = 0.90
    grid_min_line_count: int = 5
    grid_black_line_ratio: float = 0.90
    # 短表中同列数字“1”的竖笔画可能恰好超过 90%。竖线因此改用更严格的
    # 中段覆盖率，并要求候选线明显深于左右邻域；横线仍沿用上面的 90%。
    grid_black_column_line_ratio: float = 0.95
    grid_black_column_endpoint_trim_ratio: float = 0.05
    grid_black_column_min_contrast: float = 30.0
    # 压字伪线常会在局部制造一串远小于全表典型列宽的窄格。V6 使用该
    # 比例做自适应复核，固定像素上限仍沿用 grid_min_cell_size，避免
    # 密集小表被统一按 18 像素误删。
    grid_black_column_min_gap_ratio: float = 0.60
    # 若窄间距在全表所占比例超过该值，它更像真实的密集列规格而非孤立
    # 压字误检，此时整张表不执行候选清理。
    grid_black_column_close_gap_max_fraction: float = 0.12
    # 竖线若已有至少 98% 的绝对黑像素覆盖率，本身足以说明它是一条连续
    # 物理线，此时不再让容易受邻近底色影响的灰度对比规则误杀。
    grid_black_column_contrast_bypass_ratio: float = 0.98
    grid_reliable_line_count: int = 5
    # 有些表只在表头和分段处画横线，并不是每一行都有网格线。若稳定的
    # 行白带数量远多于黑线，则把黑线视为局部分隔线，行结构改用白带。
    grid_partial_line_min_white_bands: int = 12
    grid_partial_line_white_band_multiplier: float = 3.0
    grid_partial_line_min_white_regularity: float = 0.80
    # 黑线不能只靠“数量够多”就成为物理网格：至少要有一条边界处在表格
    # 中部，且任何一个逻辑格都不能独占所在方向绝大部分长度。
    grid_interior_margin_ratio: float = 0.10
    grid_max_cell_span_ratio: float = 0.95
    grid_min_cell_size: int = 18
    whitespace_blank_ratio: float = 0.01
    # 横向行白带始终保留1像素能力；只有纵向列白带会按每个分表区域从
    # 1～7像素中选择一个固定最小宽度。
    whitespace_min_band: int = 1
    whitespace_column_max_min_band: int = 7
    # 原始列间距已有90%稳定度时保持1像素，不触发清理。否则新宽度至少
    # 改善8个百分点、保留25%候选，并且保留白带要明显宽于删除白带。
    whitespace_column_regular_spacing_ratio: float = 0.90
    whitespace_column_min_regularity_gain: float = 0.08
    whitespace_column_min_retention_ratio: float = 0.25
    whitespace_column_min_width_separation_ratio: float = 1.60
    # 行很多的无框表通常具有稳定的数据主体。只在行白带达到该数量时，
    # 才允许去掉顶部/底部少量表头页脚后重新寻找列白带。
    whitespace_column_body_min_row_bands: int = 20
    whitespace_column_body_trim_ratio: float = 0.05
    whitespace_dilate_ratio: float = 0.004
    whitespace_horizontal_dilate_ratio: float | None = 0.0015
    whitespace_vertical_dilate_ratio: float | None = 0.004
    # 表体滑动窗口：在50%分辨率图上寻找列结构稳定的主体区。
    body_window_height_ratio: float = 0.18
    body_window_step_ratio: float = 0.05
    body_window_min_height: int = 120
    body_window_min_count: int = 3
    body_column_stable_max_width: int = 20
    body_column_stable_repeat: int = 3
    body_column_max_count_delta: int = 4
    body_column_min_position_match: float = 0.80
    body_column_position_tolerance_px: int = 8
    # 列白缝二维自适应晕染：上下负责连接同列文字，左右负责抹掉
    # 中文、数字内部过于整齐的伪列缝。左右比例采用人工梯度确认后的0%～1%。
    body_column_dilate_min_ratio: float = 0.01
    body_column_dilate_max_ratio: float = 0.03
    body_column_sparse_dilate_max_ratio: float = 0.06
    body_column_horizontal_dilate_min_ratio: float = 0.0
    body_column_horizontal_dilate_max_ratio: float = 0.01
    body_row_dilate_min_ratio: float = 0.015
    body_row_dilate_max_ratio: float = 0.04
    repeat_header_rows: int = 0
    repeat_stub_columns: int = 0
    # 空单元格也会输出 HTML 标签，因此按逻辑总格子数限制输出规模。
    # 图片尺寸能装下，不代表几千个单元格也能一次写完。
    # 官方 API 对接近 280 格的块识别不稳定；正式值减半到 140 格，
    # 同时保留 80 格优选下限，避免切出大量没有意义的小碎片。
    max_logical_cells_per_tile: int = 140
    # 这是规划优选目标，不是硬限制；整张小表仍可少于 80 格。
    preferred_min_logical_cells_per_tile: int = 80
    # 极端细长块即使格子不多，也继续沿长边拆分以保留表格语义。
    max_tile_aspect_ratio: float = 8.0
    projection_threshold: int = 225
    projection_min_line_ratio: float = 0.22
    projection_min_lines: int = 3
    projection_max_line_gap_ratio: float = 0.10
    # 密度分表偶尔会把独立表题切成一个很浅的小区域。满足这两个比例时，
    # 小区域会并回下一张表，由 top_context 链路识别标题。
    density_title_strip_max_page_height_ratio: float = 0.05
    density_title_strip_max_next_height_ratio: float = 0.10
    pipeline_version: str = "table-v25-v7-formal-r1"

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
        if not 0 < self.analysis_box_smear_horizontal_ratio <= 0.25:
            raise ValueError("分析框左右晕染比例必须位于 (0, 0.25] 内")
        if not 0 < self.analysis_box_smear_vertical_ratio <= 0.25:
            raise ValueError("分析框上下晕染比例必须位于 (0, 0.25] 内")
        if not 1 <= self.analysis_box_close_iterations <= 8:
            raise ValueError("分析框闭运算次数必须位于1到8之间")
        if not 0 <= self.analysis_box_padding_ratio <= 0.20:
            raise ValueError("分析框外扩比例必须位于 [0, 0.20] 内")
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
        if self.top_context_line_guard_px < 0:
            raise ValueError("顶部候选区的横线保护像素不能为负数")
        if not 0 <= self.top_context_min_ink_ratio < 1:
            raise ValueError("顶部候选区最小墨迹比例必须位于 [0, 1) 内")
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
        if not 0 < self.grid_black_column_min_gap_ratio < 1:
            raise ValueError("竖线最小间距比例必须位于 (0, 1) 内")
        if not 0 < self.grid_black_column_close_gap_max_fraction <= 1:
            raise ValueError(
                "竖线窄间距占比上限必须位于 (0, 1] 内"
            )
        if self.grid_reliable_line_count < 1:
            raise ValueError("grid_reliable_line_count 至少为 1")
        if self.grid_partial_line_min_white_bands < 3:
            raise ValueError("局部分隔线复核至少需要3根行白带")
        if self.grid_partial_line_white_band_multiplier <= 1:
            raise ValueError("行白带相对黑线的数量倍数必须大于1")
        if not 0 <= self.grid_partial_line_min_white_regularity <= 1:
            raise ValueError("行白带最小稳定度必须位于 [0, 1] 内")
        if not 0 <= self.grid_interior_margin_ratio < 0.5:
            raise ValueError("网格内部边界留白比例必须位于 [0, 0.5) 内")
        if not 0 < self.grid_max_cell_span_ratio < 1:
            raise ValueError("最大单元格跨度比例必须位于 (0, 1) 内")
        if not 0 <= self.whitespace_blank_ratio < 1:
            raise ValueError("whitespace_blank_ratio 必须位于 [0, 1) 内")
        if self.whitespace_min_band <= 0 or self.whitespace_dilate_ratio <= 0:
            raise ValueError("空白带最小宽度和墨水扩张比例必须大于 0")
        if not 1 <= self.whitespace_column_max_min_band <= 64:
            raise ValueError("列白带最大最小宽度必须位于1和64之间")
        if not 0 < self.whitespace_column_regular_spacing_ratio <= 1:
            raise ValueError("列间距稳定度阈值必须位于 (0, 1] 内")
        if not 0 <= self.whitespace_column_min_regularity_gain <= 1:
            raise ValueError("列间距最小改善值必须位于 [0, 1] 内")
        if not 0 < self.whitespace_column_min_retention_ratio <= 1:
            raise ValueError("列白带最小保留比例必须位于 (0, 1] 内")
        if self.whitespace_column_min_width_separation_ratio <= 1:
            raise ValueError("保留与删除白带宽度比必须大于1")
        if self.whitespace_column_body_min_row_bands < 6:
            raise ValueError("主体区列复核至少需要6根行白带")
        if not 0 < self.whitespace_column_body_trim_ratio < 0.25:
            raise ValueError("主体区首尾裁剪比例必须位于 (0, 0.25) 内")
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
        if not 0 < self.body_window_height_ratio <= 1:
            raise ValueError("表体窗口高度比例必须位于 (0, 1] 内")
        if not 0 < self.body_window_step_ratio <= 1:
            raise ValueError("表体窗口步长比例必须位于 (0, 1] 内")
        if self.body_window_min_height < 20:
            raise ValueError("表体窗口最小高度不能小于20")
        if self.body_window_min_count < 2:
            raise ValueError("表体至少需要连续两个稳定窗口")
        if not 1 <= self.body_column_stable_max_width <= 64:
            raise ValueError("列白缝稳定扫描上限必须位于1和64之间")
        if self.body_column_stable_repeat < 2:
            raise ValueError("列白缝稳定重复次数至少为2")
        if self.body_column_max_count_delta < 0:
            raise ValueError("窗口列数允许差值不能为负数")
        if not 0 < self.body_column_min_position_match <= 1:
            raise ValueError("窗口列位置最小匹配比例必须位于 (0, 1] 内")
        if self.body_column_position_tolerance_px < 0:
            raise ValueError("窗口列位置容差不能为负数")
        if not 0 < self.body_column_dilate_min_ratio <= self.body_column_dilate_max_ratio:
            raise ValueError("表体列上下晕染基础比例范围不合法")
        if not (
            self.body_column_dilate_max_ratio
            <= self.body_column_sparse_dilate_max_ratio
            <= 0.25
        ):
            raise ValueError("稀疏表列上下晕染上限必须不低于基础上限且不超过25%")
        if not (
            0
            <= self.body_column_horizontal_dilate_min_ratio
            <= self.body_column_horizontal_dilate_max_ratio
            <= 0.10
        ):
            raise ValueError("表体列左右晕染比例必须满足 0≤最小值≤最大值≤10%")
        if not 0 < self.body_row_dilate_min_ratio <= self.body_row_dilate_max_ratio:
            raise ValueError("行白缝左右晕染比例范围不合法")
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
        if not 0 < self.density_title_strip_max_page_height_ratio < 0.25:
            raise ValueError("标题条最大页面高度比例必须位于 (0, 0.25) 内")
        if not 0 < self.density_title_strip_max_next_height_ratio < 0.5:
            raise ValueError("标题条相对下表高度比例必须位于 (0, 0.5) 内")

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
