"""长图分支配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LongConfig:
    """长图滑窗、标题分析和二次切块参数。"""

    backend: str = "auto"
    yolo_model_path: str = "360LayoutAnalysis/general6-8n.pt"
    window_height: int = 2048
    window_step: int = 1792
    yolo_imgsz: int = 1280
    yolo_batch_size: int = 8
    yolo_base_confidence: float = 0.15
    save_yolo_debug: bool = False
    title_confidence: float = 0.20
    text_confidence: float = 0.25
    other_confidence: float = 0.25
    deduplicate_iou: float = 0.50
    logical_title_gap_ratio: float = 0.35
    logical_title_width_ratio: float = 0.60
    center_tolerance_ratio: float = 0.10
    max_vlm_height: int = 3900
    vlm_overlap: int = 200
    safe_cut_search: int = 400
    minimum_part_height: int = 512
    pipeline_version: str = "long-v1"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "pillow", "vips"}:
            raise ValueError("backend 只能是 auto、pillow 或 vips")
        if self.window_height <= 0 or self.window_step <= 0:
            raise ValueError("滑窗高度和步长必须大于 0")
        if self.window_step >= self.window_height:
            raise ValueError("window_step 必须小于 window_height，才能保留重叠区")
        if self.window_height > 4096:
            raise ValueError("检测窗口高度不得超过 4096")
        if not 512 <= self.max_vlm_height <= 4096:
            raise ValueError("max_vlm_height 应位于 512 到 4096 之间")
        if not 0 <= self.vlm_overlap < self.max_vlm_height:
            raise ValueError("vlm_overlap 必须小于 max_vlm_height")
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
