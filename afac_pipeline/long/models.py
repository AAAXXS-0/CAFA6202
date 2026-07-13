"""长图分支使用的数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..common.models import Box


@dataclass(frozen=True)
class DetectionWindow:
    index: int
    start_y: int
    end_y: int
    ownership_start_y: int
    ownership_end_y: int
    file_name: str

    @property
    def height(self) -> int:
        return self.end_y - self.start_y

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayoutBlock:
    id: str
    label: str
    box: Box
    confidence: float
    source_window: int
    member_ids: tuple[str, ...] = ()

    @property
    def center_x(self) -> float:
        return (self.box.x1 + self.box.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.box.y1 + self.box.y2) / 2

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_dict()
        result["member_ids"] = list(self.member_ids)
        return result


@dataclass(frozen=True)
class Heading:
    id: str
    level: int
    role: str
    box: Box
    parent_id: str | None
    confidence: float
    centered: bool = False
    member_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_dict()
        result["member_ids"] = list(self.member_ids)
        return result


@dataclass(frozen=True)
class SemanticPart:
    id: str
    segment_id: str
    role: str
    source_box: Box
    part_index: int
    part_count: int
    h1_id: str | None
    h2_id: str | None
    h3_id: str | None
    expected_heading_levels: tuple[int, ...]
    file_name: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_box"] = self.source_box.to_dict()
        result["expected_heading_levels"] = list(self.expected_heading_levels)
        return result


@dataclass
class SemanticSegment:
    id: str
    role: str
    start_y: int
    end_y: int
    h1_id: str | None = None
    h2_id: str | None = None
    h3_id: str | None = None
    expected_heading_levels: tuple[int, ...] = ()
    parts: list[SemanticPart] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.end_y - self.start_y

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "start_y": self.start_y,
            "end_y": self.end_y,
            "h1_id": self.h1_id,
            "h2_id": self.h2_id,
            "h3_id": self.h3_id,
            "expected_heading_levels": list(self.expected_heading_levels),
            "parts": [part.to_dict() for part in self.parts],
        }
