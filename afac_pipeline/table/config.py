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

    detector=auto 时优先使用已有 YOLO 权重；依赖或权重不可用时，
    自动退回无参数的水平线投影检测器。
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
    projection_threshold: int = 225
    projection_min_line_ratio: float = 0.22
    projection_min_lines: int = 3
    projection_max_line_gap_ratio: float = 0.10
    pipeline_version: str = "table-v1"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "pillow", "vips"}:
            raise ValueError("backend 只能是 auto、pillow 或 vips")
        if self.detector not in {"auto", "projection", "yolo"}:
            raise ValueError("detector 只能是 auto、projection 或 yolo")
        if not 512 <= self.preview_max_side <= 4096:
            raise ValueError("preview_max_side 应位于 512 到 4096 之间")
        if not 512 <= self.max_vlm_side <= 4096:
            raise ValueError("max_vlm_side 应位于 512 到 4096 之间")
        if not 0 <= self.tile_overlap < self.max_vlm_side:
            raise ValueError("tile_overlap 必须小于 max_vlm_side")
        if not 0 < self.single_tile_min_scale <= 1:
            raise ValueError("single_tile_min_scale 必须位于 (0, 1] 内")

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
