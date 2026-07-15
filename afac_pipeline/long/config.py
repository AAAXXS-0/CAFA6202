"""长图分支配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LongConfig:
    """固定检测窗口、语义样式分析与最终安全切块参数。"""

    strategy: str = "semantic"
    backend: str = "auto"
    yolo_model_path: str = "360LayoutAnalysis/general6-8n.pt"
    window_height: int = 2048
    window_step: int = 1792
    yolo_imgsz: int = 640
    yolo_batch_size: int = 8
    yolo_base_confidence: float = 0.25
    save_yolo_debug: bool = False
    title_confidence: float = 0.60
    text_confidence: float = 0.50
    other_confidence: float = 0.50
    cut_protection_confidence: float = 0.25
    cut_protection_padding: int = 16
    deduplicate_iou: float = 0.50
    # 以下三个字段只供归档算法和本地 OCR 备用线读取。
    logical_title_gap_ratio: float = 0.35
    logical_title_width_ratio: float = 0.60
    center_tolerance_ratio: float = 0.10
    projection_sample_width: int = 256
    projection_white_threshold: int = 245
    projection_blank_ratio: float = 0.01
    minimum_blank_band: int = 8
    adaptive_target_height: int = 3200
    adaptive_min_height: int = 2200
    max_vlm_height: int = 3900
    vlm_overlap: int = 200
    safe_cut_search: int = 600
    # v1 清单兼容字段；v2 不再使用加权 H2 分数。
    semantic_h2_min_score: float = 0.62
    semantic_ink_threshold: int = 225
    semantic_active_row_ratio: float = 0.01
    semantic_full_width_active_ratio: float = 0.002
    semantic_line_merge_gap: int = 2
    semantic_min_ink_line_height: int = 6
    semantic_min_ink_width_ratio: float = 0.02
    semantic_multiline_gap_ratio: float = 0.35
    semantic_multiline_overlap_ratio: float = 0.65
    semantic_model_title_min_ratio: float = 1.05
    semantic_ink_only_title_ratio: float = 1.35
    semantic_ink_only_min_whitespace_ratio: float = 0.60
    # 单行标题即使字号很大，高度通常也不会超过正文的 2.6 倍。超过该值的
    # 连通墨迹更可能是表格线、插图或多行内容粘成的大块，不能参与标题投票。
    semantic_title_max_height_ratio: float = 2.60
    # H3 可以只比正文略大，但 H2 是章节边界，要求应更严格。达不到该值时
    # 宁可回退安全切块，也不把几十个普通小标题全部提升为 H2。
    semantic_h2_min_style_ratio: float = 1.20
    semantic_candidate_overlap_ratio: float = 0.65
    semantic_toc_anchor_max_width_ratio: float = 0.82
    semantic_toc_second_anchor_min_height_ratio: float = 2.00
    semantic_toc_min_line_count: int = 8
    semantic_min_heading_ratio: float = 1.08
    semantic_style_height_tolerance: float = 0.10
    semantic_style_indent_tolerance: float = 0.06
    semantic_h2_cluster_height_tolerance: float = 0.08
    semantic_center_max_width_ratio: float = 0.55
    semantic_title_padding: int = 12
    semantic_context_gap: int = 10
    semantic_audit_windows: bool = True
    minimum_part_height: int = 512
    pipeline_version: str = "long-v8-whole-toc-request"

    def __post_init__(self) -> None:
        if self.strategy not in {"semantic", "legacy"}:
            raise ValueError("strategy 只能是 semantic 或 legacy")
        if self.backend not in {"auto", "pillow", "vips"}:
            raise ValueError("backend 只能是 auto、pillow 或 vips")
        if self.window_height <= 0 or self.window_step <= 0:
            raise ValueError("滑窗高度和步长必须大于 0")
        if self.window_step >= self.window_height:
            raise ValueError("window_step 必须小于 window_height，才能保留重叠区")
        if self.window_height > 4096:
            raise ValueError("检测窗口高度不得超过 4096")
        confidence_values = (
            self.yolo_base_confidence,
            self.title_confidence,
            self.text_confidence,
            self.other_confidence,
            self.cut_protection_confidence,
        )
        if any(not 0 <= value <= 1 for value in confidence_values):
            raise ValueError("YOLO 与切割保护置信度必须位于 0 到 1 之间")
        if self.cut_protection_padding < 0:
            raise ValueError("cut_protection_padding 不能小于 0")
        if self.projection_sample_width <= 0:
            raise ValueError("projection_sample_width 必须大于 0")
        if not 0 <= self.projection_white_threshold <= 255:
            raise ValueError("projection_white_threshold 必须位于 0 到 255 之间")
        if not 0 <= self.projection_blank_ratio <= 1:
            raise ValueError("projection_blank_ratio 必须位于 0 到 1 之间")
        if self.minimum_blank_band <= 0:
            raise ValueError("minimum_blank_band 必须大于 0")
        if not 512 <= self.max_vlm_height <= 4096:
            raise ValueError("max_vlm_height 应位于 512 到 4096 之间")
        if not 0 < self.adaptive_min_height <= self.adaptive_target_height:
            raise ValueError("自适应最小高度必须大于 0 且不超过目标高度")
        if not self.adaptive_target_height <= self.max_vlm_height:
            raise ValueError("自适应目标高度不能超过 max_vlm_height")
        if self.safe_cut_search < 0:
            raise ValueError("safe_cut_search 不能小于 0")
        if not 0 <= self.semantic_h2_min_score <= 1:
            raise ValueError("semantic_h2_min_score 必须位于 0 到 1 之间")
        if not 0 <= self.semantic_ink_threshold <= 255:
            raise ValueError("semantic_ink_threshold 必须位于 0 到 255 之间")
        if not 0 <= self.semantic_active_row_ratio <= 1:
            raise ValueError("semantic_active_row_ratio 必须位于 0 到 1 之间")
        if not 0 < self.semantic_full_width_active_ratio <= 1:
            raise ValueError("semantic_full_width_active_ratio 必须位于 0 到 1 之间")
        if not 0 < self.semantic_min_ink_width_ratio <= 1:
            raise ValueError("semantic_min_ink_width_ratio 必须位于 0 到 1 之间")
        if not 0 < self.semantic_multiline_overlap_ratio <= 1:
            raise ValueError("semantic_multiline_overlap_ratio 必须位于 0 到 1 之间")
        if not 0 < self.semantic_center_max_width_ratio <= 1:
            raise ValueError("semantic_center_max_width_ratio 必须位于 0 到 1 之间")
        if not 0 < self.semantic_toc_anchor_max_width_ratio <= 1:
            raise ValueError("semantic_toc_anchor_max_width_ratio 必须位于 0 到 1 之间")
        if not 0 < self.semantic_candidate_overlap_ratio <= 1:
            raise ValueError("semantic_candidate_overlap_ratio 必须位于 0 到 1 之间")
        nonnegative = (
            self.semantic_multiline_gap_ratio,
            self.semantic_model_title_min_ratio,
            self.semantic_ink_only_title_ratio,
            self.semantic_ink_only_min_whitespace_ratio,
            self.semantic_min_heading_ratio,
            self.semantic_style_height_tolerance,
            self.semantic_style_indent_tolerance,
            self.semantic_h2_cluster_height_tolerance,
            self.semantic_title_max_height_ratio,
            self.semantic_h2_min_style_ratio,
            self.semantic_toc_second_anchor_min_height_ratio,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("语义墨迹和样式比例参数不能小于 0")
        if self.semantic_line_merge_gap <= 0:
            raise ValueError("semantic_line_merge_gap 必须大于 0")
        if self.semantic_min_ink_line_height <= 0:
            raise ValueError("semantic_min_ink_line_height 必须大于 0")
        if self.semantic_toc_min_line_count <= 0:
            raise ValueError("semantic_toc_min_line_count 必须大于 0")
        if self.semantic_title_padding < 0 or self.semantic_context_gap < 0:
            raise ValueError("语义标题留白和上下文间隔不能小于 0")
        if not 0 <= self.vlm_overlap < self.adaptive_min_height:
            raise ValueError("vlm_overlap 必须小于自适应最小高度")
        if self.minimum_part_height >= self.max_vlm_height:
            raise ValueError("minimum_part_height 必须小于 max_vlm_height")

    @property
    def window_overlap(self) -> int:
        return self.window_height - self.window_step

    @classmethod
    def from_json(cls, path: str | Path | None) -> "LongConfig":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as file:
            raw: dict[str, Any] = json.load(file)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"长图配置中存在未知字段：{', '.join(unknown)}")
        return cls(**raw)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
