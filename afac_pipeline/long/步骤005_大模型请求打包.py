"""将自适应安全块封装为 FinixDoc-VL 请求，并整理 Markdown 标题。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .步骤001_数据定义 import SafeCutChunk, SemanticSegment
from ..common.models import Box


@dataclass(frozen=True)
class RecognitionPack:
    id: str
    source_box: Box
    segment_ids: tuple[str, ...]
    part_ids: tuple[str, ...]
    heading_hints: tuple[dict[str, Any], ...]
    file_name: str
    cut_method: str = "legacy"
    overlap_top: int = -1
    overlap_bottom: int = -1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_box"] = self.source_box.to_dict()
        result["segment_ids"] = list(self.segment_ids)
        result["part_ids"] = list(self.part_ids)
        result["heading_hints"] = list(self.heading_hints)
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecognitionPack":
        return cls(
            id=str(raw["id"]),
            source_box=Box.from_dict(raw["source_box"]),
            segment_ids=tuple(str(value) for value in raw["segment_ids"]),
            part_ids=tuple(str(value) for value in raw["part_ids"]),
            heading_hints=tuple(dict(value) for value in raw["heading_hints"]),
            file_name=str(raw["file_name"]),
            cut_method=str(raw.get("cut_method", "legacy")),
            overlap_top=int(raw.get("overlap_top", -1)),
            overlap_bottom=int(raw.get("overlap_bottom", -1)),
        )


def build_recognition_packs(
    segments: list[SemanticSegment], image_width: int, max_height: int
) -> list[RecognitionPack]:
    """合并坐标连续的小语义块，同时保留每个 H2/H3 的逻辑元数据。

    同一超长语义段的物理块存在重叠，不与前一请求强行合并；普通相邻语义块
    则尽量填满 3900 像素，从而显著减少 API 调用次数。
    """

    units: list[tuple[SemanticSegment, Any]] = [
        (segment, part)
        for segment in segments
        for part in segment.parts
    ]
    units.sort(key=lambda item: (item[1].source_box.y1, item[1].source_box.y2))
    grouped: list[list[tuple[SemanticSegment, Any]]] = []
    for unit in units:
        if not grouped:
            grouped.append([unit])
            continue
        current = grouped[-1]
        current_start = current[0][1].source_box.y1
        current_end = max(item[1].source_box.y2 for item in current)
        candidate = unit[1].source_box
        can_pack = (
            candidate.y1 >= current_end
            and candidate.y2 - current_start <= max_height
        )
        if can_pack:
            current.append(unit)
        else:
            grouped.append([unit])

    packs: list[RecognitionPack] = []
    for index, group in enumerate(grouped):
        start_y = min(part.source_box.y1 for _, part in group)
        end_y = max(part.source_box.y2 for _, part in group)
        hints = tuple(
            {
                "segment_id": segment.id,
                "role": segment.role,
                "h2_id": segment.h2_id,
                "h3_id": segment.h3_id,
                "expected_heading_levels": list(part.expected_heading_levels),
            }
            for segment, part in group
        )
        packs.append(
            RecognitionPack(
                id=f"request_{index:05d}",
                source_box=Box(0, start_y, image_width, end_y),
                segment_ids=tuple(dict.fromkeys(segment.id for segment, _ in group)),
                part_ids=tuple(part.id for _, part in group),
                heading_hints=hints,
                file_name=f"request_{index:05d}_y{start_y:07d}_{end_y:07d}.png",
            )
        )
    return packs


def build_adaptive_recognition_packs(
    chunks: list[SafeCutChunk], image_width: int
) -> list[RecognitionPack]:
    """一个安全块对应一个请求，不再按推测出的 H2/H3 结构拆分。"""

    packs: list[RecognitionPack] = []
    for index, chunk in enumerate(chunks):
        packs.append(
            RecognitionPack(
                id=f"request_{index:05d}",
                source_box=chunk.source_box,
                segment_ids=(chunk.id,),
                part_ids=(),
                heading_hints=(),
                file_name=(
                    f"request_{index:05d}_y{chunk.source_box.y1:07d}_"
                    f"{chunk.source_box.y2:07d}.png"
                ),
                cut_method=chunk.cut_method,
                overlap_top=chunk.overlap_top,
                overlap_bottom=chunk.overlap_bottom,
            )
        )
    return packs


def build_pack_prompt(pack: RecognitionPack) -> str:
    boundary = (
        f"本块顶部与前一块重叠约 {pack.overlap_top} 像素，重复内容照实输出。"
        if pack.overlap_top > 0
        else "本块从安全空白边界开始，与前一块没有正文重叠。"
    )
    return f"""请把这张金融文档长图请求块转换为与原图严格对应的 Markdown。

切块方式：{pack.cut_method}
边界说明：{boundary}

要求：
1. 只输出图片中真实可见的内容，严禁补全、总结、解释或改写。
2. 必须保留图片里的全部标题，并根据可见编号、字号和排版判断 #、##、###。
3. 保持文字、数字、编号、标点、特殊符号、列表和表格的原始顺序。
4. 普通加粗文字不要擅自提升为标题，空白区域不要生成内容。
5. 若图片从段落中间开始或结束，只抄录可见部分，不猜测缺失内容。
6. 不要输出 Markdown 代码围栏、识别说明、页码统计或置信度。
7. 只有边界说明明确存在重叠时，才可能与相邻请求重复，程序会处理接缝。

直接输出 Markdown："""


def _numbered_heading_level(text: str) -> int | None:
    """从已经被 VLM 判断为标题的文字编号中推断层级。"""

    import re

    value = text.strip()
    arabic = re.match(r"^(\d+(?:\.\d+){0,5})(?:\s|[、．]|$)", value)
    if arabic:
        number = arabic.group(1)
        if "." not in number and int(number) > 99:
            return None
        return min(3, number.count(".") + 1)
    if re.match(r"^[一二三四五六七八九十百]+[、．.]", value):
        return 1
    if re.match(r"^[（(][一二三四五六七八九十百]+[）)]", value):
        return 2
    if re.match(r"^[（(]?\d+[）)]", value):
        return 3
    return None


def normalize_markdown_heading_levels(markdown: str) -> str:
    """仅纠正 VLM 已输出的标题井号，不把普通正文擅自提升为标题。"""

    import re

    output: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            output.append(line)
            continue
        level = _numbered_heading_level(match.group(2))
        if level is None:
            output.append(line)
        else:
            output.append(f"{'#' * level} {match.group(2)}")
    return "\n".join(output).strip()
