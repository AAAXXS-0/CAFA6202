"""工作流各模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Box:
    """使用原图像素坐标表示的矩形，右下边界不包含在矩形内。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "Box":
        """将矩形限制在图像内部，并保证坐标顺序合法。"""

        x1 = min(max(0, self.x1), width)
        y1 = min(max(0, self.y1), height)
        x2 = min(max(x1, self.x2), width)
        y2 = min(max(y1, self.y2), height)
        return Box(x1, y1, x2, y2)

    def expand(self, padding_x: int, padding_y: int, width: int, height: int) -> "Box":
        """向四周扩张矩形，用于保留表题、单位与脚注。"""

        return Box(
            self.x1 - padding_x,
            self.y1 - padding_y,
            self.x2 + padding_x,
            self.y2 + padding_y,
        ).clamp(width, height)

    def intersection_area(self, other: "Box") -> int:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(self, other: "Box") -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Box":
        return cls(**{key: int(value[key]) for key in ("x1", "y1", "x2", "y2")})


@dataclass(frozen=True)
class ImageMeta:
    path: Path
    file_name: str
    width: int
    height: int
    actual_format: str
    file_size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class DetectedBox:
    box: Box
    label: str = "table"
    confidence: float = 1.0
    source: str = "projection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "label": self.label,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class TilePlan:
    """一个送入视觉模型的表格切片计划。"""

    region_index: int
    row_index: int
    column_index: int
    row_count: int
    column_count: int
    source_box: Box
    output_width: int
    output_height: int
    scale: float
    file_name: str
    # 以下逻辑坐标只由图表结构化切块使用。坐标采用左闭右开区间，
    # 表示该图片块真正负责识别的表格行列；重复表头和行名列不计入责任区。
    logical_row_start: int = 0
    logical_row_end: int = 1
    logical_column_start: int = 0
    logical_column_end: int = 1
    header_context_rows: int = 0
    stub_context_columns: int = 0
    tiling_mode: str = "pixel_overlap"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_box"] = self.source_box.to_dict()
        return result


@dataclass
class PreparedRegion:
    index: int
    box: Box
    detector_source: str
    tiles: list[TilePlan] = field(default_factory=list)
    grid_source: str = "unavailable"
    row_boundaries: list[int] = field(default_factory=list)
    column_boundaries: list[int] = field(default_factory=list)
    raw_column_boundaries: list[int] = field(default_factory=list)
    rejected_column_boundaries: list[int] = field(default_factory=list)
    # 表格物理网格上方、分析框顶部到第一根横线之间的候选标题区。
    # 它只作为表前 Markdown 识别，不进入 R×C 网格。
    top_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "box": self.box.to_dict(),
            "detector_source": self.detector_source,
            "grid_source": self.grid_source,
            "row_boundaries": self.row_boundaries,
            "column_boundaries": self.column_boundaries,
            "raw_column_boundaries": self.raw_column_boundaries,
            "rejected_column_boundaries": self.rejected_column_boundaries,
            "top_context": self.top_context,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }
