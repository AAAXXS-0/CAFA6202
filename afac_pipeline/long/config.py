"""长图分支配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LongConfig:
    """固定检测滑窗与自适应最终安全切块参数。"""

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
    # 保留旧字段以便读取旧清单；新流程不再按标题语义段物理切块。
    minimum_part_height: int = 512
    pipeline_version: str = "long-v4-adaptive-safe-cut"

    def __post_init__(self) -> None:
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
