"""把 legacy 安全块或 semantic H2 章节封装为 FinixDoc-VL 请求。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
from typing import Any

from .config import LongConfig
from .步骤001_数据定义 import Heading, LayoutBlock, SafeCutChunk, SemanticSegment
from .步骤004_自适应安全切块 import build_adaptive_chunks, find_blank_bands
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
    # context_boxes 是从原图裁出的祖先标题条，按 H2、H3 顺序拼在正文块上方。
    context_boxes: tuple[Box, ...] = ()
    context_heading_ids: tuple[str, ...] = ()
    visible_heading_ids: tuple[str, ...] = ()
    semantic_role: str = "legacy"
    sequence: int = -1
    # 原图坐标保持不变，只在生成请求 PNG 时缩放。目录可因此整块送入。
    body_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_box"] = self.source_box.to_dict()
        result["segment_ids"] = list(self.segment_ids)
        result["part_ids"] = list(self.part_ids)
        result["heading_hints"] = list(self.heading_hints)
        result["context_boxes"] = [box.to_dict() for box in self.context_boxes]
        result["context_heading_ids"] = list(self.context_heading_ids)
        result["visible_heading_ids"] = list(self.visible_heading_ids)
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecognitionPack":
        return cls(
            id=str(raw["id"]),
            source_box=Box.from_dict(raw["source_box"]),
            segment_ids=tuple(str(value) for value in raw.get("segment_ids", [])),
            part_ids=tuple(str(value) for value in raw.get("part_ids", [])),
            heading_hints=tuple(dict(value) for value in raw.get("heading_hints", [])),
            file_name=str(raw["file_name"]),
            cut_method=str(raw.get("cut_method", "legacy")),
            overlap_top=int(raw.get("overlap_top", -1)),
            overlap_bottom=int(raw.get("overlap_bottom", -1)),
            context_boxes=tuple(
                Box.from_dict(value) for value in raw.get("context_boxes", [])
            ),
            context_heading_ids=tuple(
                str(value) for value in raw.get("context_heading_ids", [])
            ),
            visible_heading_ids=tuple(
                str(value) for value in raw.get("visible_heading_ids", [])
            ),
            semantic_role=str(raw.get("semantic_role", "legacy")),
            sequence=int(raw.get("sequence", -1)),
            body_scale=float(raw.get("body_scale", 1.0)),
        )


def build_recognition_packs(
    segments: list[SemanticSegment], image_width: int, max_height: int
) -> list[RecognitionPack]:
    """旧标题树测试仍使用的兼容打包器。"""

    units: list[tuple[SemanticSegment, Any]] = [
        (segment, part) for segment in segments for part in segment.parts
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
        if candidate.y1 >= current_end and candidate.y2 - current_start <= max_height:
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
    """legacy 策略：一个安全块对应一个请求。"""

    return [
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
            sequence=index,
        )
        for index, chunk in enumerate(chunks)
    ]


def _context_box(
    heading: Heading, image_width: int, image_height: int, padding: int
) -> Box:
    return Box(
        0,
        max(0, heading.box.y1 - padding),
        image_width,
        min(image_height, heading.box.y2 + padding),
    )


def _visible_ids(headings: list[Heading], start_y: int, end_y: int) -> tuple[str, ...]:
    return tuple(
        item.id for item in headings if start_y <= item.box.y1 < end_y
    )


def _split_range_safely(
    start_y: int,
    end_y: int,
    maximum_height: int,
    projection: list[float],
    blocks: list[LayoutBlock],
    config: LongConfig,
) -> list[tuple[int, int, str, int, int]]:
    """在指定语义段内部切块；空白优先，完全无空白时才保留物理重叠。"""

    if end_y <= start_y:
        return []
    maximum_height = max(256, maximum_height)
    if end_y - start_y <= maximum_height:
        return [(start_y, end_y, "semantic_whole", 0, 0)]
    bands = find_blank_bands(
        projection,
        blank_ratio=config.projection_blank_ratio,
        minimum_height=config.minimum_blank_band,
    )
    protected = [
        (
            max(start_y, block.box.y1 - config.cut_protection_padding),
            min(end_y, block.box.y2 + config.cut_protection_padding),
        )
        for block in blocks
        if block.confidence >= config.cut_protection_confidence
        and block.box.y2 > start_y
        and block.box.y1 < end_y
    ]

    def is_protected(y: int) -> bool:
        return any(first <= y < second for first, second in protected)

    records: list[tuple[int, int, str, int, int]] = []
    cursor = start_y
    previous_end = start_y
    while end_y - cursor > maximum_height:
        target = cursor + min(config.adaptive_target_height, maximum_height)
        minimum = min(config.adaptive_min_height, maximum_height // 2)
        lower = min(end_y - 1, cursor + max(256, minimum))
        upper = min(end_y - 1, cursor + maximum_height)
        candidates: list[tuple[int, int]] = []
        for band in bands:
            if band.end_y <= lower or band.start_y >= upper:
                continue
            y = min(max(target, max(lower, band.start_y)), min(upper, band.end_y - 1))
            if not is_protected(y):
                candidates.append((abs(y - target), y))
        if candidates:
            boundary = min(candidates)[1]
            current_end = boundary
            next_cursor = boundary
            method = "semantic_blank_band"
        else:
            search_start = max(lower, target - config.safe_cut_search)
            search_end = min(upper, target + config.safe_cut_search)
            possible = list(range(search_start, search_end + 1))
            possible.sort(key=lambda y: (is_protected(y), projection[y], abs(y - target)))
            boundary = possible[0]
            half = min(config.vlm_overlap // 2, maximum_height // 8)
            # boundary 后补半个重叠区时仍必须服从正文可用高度。
            # 否则再拼上 H2/H3 标题条会出现 3900+十几像素的越界。
            current_end = min(
                end_y,
                cursor + maximum_height,
                boundary + half,
            )
            next_cursor = max(cursor + 1, boundary - half)
            method = "semantic_fallback_overlap"
        if current_end <= cursor or next_cursor <= cursor:
            raise RuntimeError("语义安全切块没有向下推进")
        overlap_top = max(0, previous_end - cursor)
        records.append((cursor, current_end, method, overlap_top, max(0, current_end - next_cursor)))
        previous_end = current_end
        cursor = next_cursor
    records.append((cursor, end_y, "semantic_end", max(0, previous_end - cursor), 0))
    return records


def build_semantic_recognition_packs(
    headings: list[Heading],
    blocks: list[LayoutBlock],
    projection: list[float],
    image_width: int,
    image_height: int,
    config: LongConfig,
) -> tuple[list[RecognitionPack], dict[str, Any]]:
    """整 H2 优先；超长 H2 按 H3；超长 H3 再按空白带切割。"""

    h2s = sorted((item for item in headings if item.level == 2), key=lambda item: item.box.y1)
    if not h2s:
        chunks, legacy_debug = build_adaptive_chunks(
            image_width, image_height, projection, blocks, config
        )
        return build_adaptive_recognition_packs(chunks, image_width), {
            "mode": "semantic-fallback-legacy",
            "reason": "没有可靠 H2",
            "legacy": legacy_debug,
        }

    packs: list[RecognitionPack] = []

    def append_pack(
        box: Box,
        role: str,
        contexts: tuple[Heading, ...] = (),
        *,
        cut_method: str = "semantic_whole",
        overlap_top: int = 0,
        overlap_bottom: int = 0,
        body_scale: float = 1.0,
    ) -> None:
        if box.height <= 0:
            return
        if not 0 < body_scale <= 1:
            raise ValueError("请求正文缩放比例必须位于 0 到 1 之间")
        index = len(packs)
        context_boxes = tuple(
            _context_box(item, image_width, image_height, config.semantic_title_padding)
            for item in contexts
        )
        total_height = round(box.height * body_scale)
        total_height += sum(item.height for item in context_boxes)
        total_height += config.semantic_context_gap * len(context_boxes)
        if total_height > config.max_vlm_height:
            raise RuntimeError(
                f"语义请求 {role} 拼接标题后高 {total_height}px，超过 VLM 限制"
            )
        visible = _visible_ids(headings, box.y1, box.y2)
        hints = tuple(
            {
                "heading_id": item.id,
                "level": item.level,
                "role": item.role,
                "source": "context" if item in contexts else "body",
            }
            for item in [*contexts, *[h for h in headings if h.id in visible]]
        )
        packs.append(
            RecognitionPack(
                id=f"request_{index:05d}",
                source_box=box,
                segment_ids=tuple(item.id for item in contexts) or (role,),
                part_ids=(),
                heading_hints=hints,
                file_name=f"request_{index:05d}_y{box.y1:07d}_{box.y2:07d}.png",
                cut_method=cut_method,
                overlap_top=overlap_top,
                overlap_bottom=overlap_bottom,
                context_boxes=context_boxes,
                context_heading_ids=tuple(item.id for item in contexts),
                visible_heading_ids=visible,
                semantic_role=role,
                sequence=index,
                body_scale=body_scale,
            )
        )

    def append_split_range(
        start_y: int,
        end_y: int,
        role: str,
        first_contexts: tuple[Heading, ...],
        later_contexts: tuple[Heading, ...],
    ) -> None:
        later_context_boxes = [
            _context_box(item, image_width, image_height, config.semantic_title_padding)
            for item in later_contexts
        ]
        available = config.max_vlm_height - sum(box.height for box in later_context_boxes)
        available -= config.semantic_context_gap * len(later_context_boxes)
        ranges = _split_range_safely(
            start_y, end_y, available, projection, blocks, config
        )
        for index, (first, second, method, top, bottom) in enumerate(ranges):
            append_pack(
                Box(0, first, image_width, second),
                role if index == 0 else f"{role}_continuation",
                first_contexts if index == 0 else later_contexts,
                cut_method=method,
                overlap_top=top,
                overlap_bottom=bottom,
            )

    # 检测到双居中锚点时，把目录和正文 H1 前言明确分开。目录仍会完整送给
    # VLM，只是不再参与 H2/H3 样式推断。没有目录锚点时维持原有头部切法。
    toc_headings = sorted(
        (item for item in headings if item.role == "semantic_toc"),
        key=lambda item: item.box.y1,
    )
    document_h1s = sorted(
        (item for item in headings if item.role == "semantic_h1"),
        key=lambda item: item.box.y1,
    )
    first_h2_y = h2s[0].box.y1
    separated_toc = (
        toc_headings
        and document_h1s
        and toc_headings[0].box.y1 < document_h1s[0].box.y1 < first_h2_y
    )
    if separated_toc:
        toc_heading = toc_headings[0]
        document_h1 = document_h1s[0]
        if any(
            value > config.projection_blank_ratio
            for value in projection[: toc_heading.box.y1]
        ):
            append_split_range(0, toc_heading.box.y1, "front_prefix", (), ())
        toc_box = Box(
            0,
            toc_heading.box.y1,
            image_width,
            document_h1.box.y1,
        )
        # 目录只有编号和短标题，缩小后仍很清楚。这里保留完整原图范围，
        # 只缩放最终 PNG，不再套用正文的安全切块逻辑。
        toc_scale = min(1.0, config.max_vlm_height / toc_box.height)
        append_pack(
            toc_box,
            "table_of_contents",
            cut_method=(
                "semantic_toc_whole_resize"
                if toc_scale < 1.0
                else "semantic_toc_whole"
            ),
            body_scale=toc_scale,
        )
        append_split_range(
            document_h1.box.y1,
            first_h2_y,
            "front_matter",
            (),
            (),
        )
    elif (
        first_h2_y > 0
        and any(
            value > config.projection_blank_ratio
            for value in projection[:first_h2_y]
        )
    ):
        append_split_range(0, first_h2_y, "front_matter", (), ())

    section_debug: list[dict[str, Any]] = []
    for h2_index, h2 in enumerate(h2s):
        section_end = h2s[h2_index + 1].box.y1 if h2_index + 1 < len(h2s) else image_height
        children = sorted(
            (
                item
                for item in headings
                if item.level == 3
                and h2.box.y1 < item.box.y1 < section_end
                and (item.parent_id == h2.id or item.parent_id is None)
            ),
            key=lambda item: item.box.y1,
        )
        section_height = section_end - h2.box.y1
        if section_height <= config.max_vlm_height:
            append_pack(
                Box(0, h2.box.y1, image_width, section_end),
                "h2_whole",
            )
            section_debug.append(
                {"h2_id": h2.id, "mode": "whole", "height": section_height, "h3_count": len(children)}
            )
            continue

        if not children:
            append_split_range(
                h2.box.y1,
                section_end,
                "h2_body",
                (),
                (h2,),
            )
        else:
            units: list[tuple[int, int, Heading | None]] = []
            if children[0].box.y1 > h2.box.y1:
                units.append((h2.box.y1, children[0].box.y1, None))
            for child_index, h3 in enumerate(children):
                child_end = (
                    children[child_index + 1].box.y1
                    if child_index + 1 < len(children)
                    else section_end
                )
                units.append((h3.box.y1, child_end, h3))

            group_start: int | None = None
            group_end = 0

            def flush_group() -> None:
                nonlocal group_start, group_end
                if group_start is None:
                    return
                contexts = () if group_start == h2.box.y1 else (h2,)
                append_pack(
                    Box(0, group_start, image_width, group_end),
                    "h2_first_group" if not contexts else "h3_group",
                    contexts,
                )
                group_start = None
                group_end = 0

            for unit_start, unit_end, h3 in units:
                candidate_start = group_start if group_start is not None else unit_start
                contexts = () if candidate_start == h2.box.y1 else (h2,)
                context_height = sum(
                    _context_box(
                        item,
                        image_width,
                        image_height,
                        config.semantic_title_padding,
                    ).height
                    for item in contexts
                ) + config.semantic_context_gap * len(contexts)
                if unit_end - candidate_start + context_height <= config.max_vlm_height:
                    group_start = candidate_start
                    group_end = unit_end
                    continue

                flush_group()
                contexts = () if unit_start == h2.box.y1 else (h2,)
                context_height = sum(
                    _context_box(
                        item,
                        image_width,
                        image_height,
                        config.semantic_title_padding,
                    ).height
                    for item in contexts
                ) + config.semantic_context_gap * len(contexts)
                if unit_end - unit_start + context_height <= config.max_vlm_height:
                    group_start = unit_start
                    group_end = unit_end
                    continue

                if h3 is None:
                    append_split_range(
                        unit_start,
                        unit_end,
                        "h2_intro",
                        (),
                        (h2,),
                    )
                else:
                    # 超长 H3 的标题只放在上下文条里；正文从标题框下沿开始，
                    # 每个续块都携带 H2+H3，聚合时按稳定 ID 去掉重复标题。
                    append_split_range(
                        h3.box.y2,
                        unit_end,
                        "h3_body",
                        (h2, h3),
                        (h2, h3),
                    )
            flush_group()
        section_debug.append(
            {"h2_id": h2.id, "mode": "split-by-h3", "height": section_height, "h3_count": len(children)}
        )

    debug = {
        "mode": "semantic-h2",
        "h2_count": len(h2s),
        "request_count": len(packs),
        "context_request_count": sum(bool(item.context_boxes) for item in packs),
        "fallback_overlap_count": sum(item.overlap_top > 0 for item in packs),
        "toc_request_count": sum(
            item.semantic_role.startswith("table_of_contents") for item in packs
        ),
        "toc_scaled_request_count": sum(
            item.semantic_role.startswith("table_of_contents")
            and item.body_scale < 1.0
            for item in packs
        ),
        "sections": section_debug,
    }
    return packs, debug


def build_pack_prompt(pack: RecognitionPack) -> str:
    if pack.context_boxes:
        context = (
            "图片顶部有程序从同一原图复制的上级标题条，标题条之间用细灰线分隔。"
            "这些标题只用于说明当前正文所属章节；请在 Markdown 中按顺序输出每个标题一次，"
            "不要描述分隔线，也不要因为标题重复出现而生成两份。"
        )
    else:
        context = "图片没有额外复制的标题上下文，按原图可见结构识别。"
    boundary = (
        f"正文块顶部与前一块重叠约 {pack.overlap_top} 像素，重复内容照实输出。"
        if pack.overlap_top > 0
        else "正文块从语义边界或安全空白边界开始，没有人为省略中间内容。"
    )
    return f"""请把这张金融文档长图请求块转换为与原图严格对应的 Markdown。

语义角色：{pack.semantic_role}
切块方式：{pack.cut_method}
标题上下文：{context}
边界说明：{boundary}

要求：
1. 只输出图片中真实可见的内容，严禁补全、总结、解释或改写。
2. 必须保留全部标题，并根据可见编号、字号和排版判断 #、##、###、####。
3. 保持文字、数字、编号、标点、特殊符号、列表和表格的原始顺序。
4. 普通加粗文字不要擅自提升为标题，空白区域不要生成内容。
5. 若正文从段落中间开始或结束，只抄录可见部分，不猜测缺失内容。
6. 不要输出 Markdown 代码围栏、识别说明、页码统计或置信度。
7. 只有边界说明明确存在重叠时，才可能与相邻请求重复，程序会处理接缝。

直接输出 Markdown："""


def _normalized_heading_text(text: str) -> str:
    return re.sub(r"[\s#：:，,。．.、（）()\-—_]", "", text).lower()


def strip_backend_prompt_echo(markdown: str) -> str:
    """删除官方模型偶尔复述的英文任务说明，保留其后的真实文档。

    这里只识别固定的任务说明句式，不会按保险产品名或测试图片名删文字。
    若一个空白小块只返回了任务说明，清理后的空字符串会被聚合层自然跳过。
    """

    lines = markdown.strip().splitlines()
    if not lines or not re.fullmatch(
        r"#{1,2}\s+Markdown\s+Parsing\s+Task",
        lines[0].strip(),
        re.IGNORECASE,
    ):
        return markdown.strip()

    known_prefixes = (
        "extract all the text content",
        "convert tables",
        "convert all the text content",
        "ignore figures",
        "sort all text content",
        "sort all the text content",
    )
    index = 1
    while index < len(lines):
        value = lines[index].strip()
        lowered = value.lstrip("-* ").lower()
        if not value or re.fullmatch(
            r"#{1,3}\s+requirements", value, re.IGNORECASE
        ):
            index += 1
            continue
        if any(lowered.startswith(prefix) for prefix in known_prefixes):
            index += 1
            continue
        break
    return "\n".join(lines[index:]).strip()


def _heading_line_text(line: str) -> str:
    match = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
    return match.group(1).strip() if match else line.strip()


def _force_heading_level(lines: list[str], index: int, level: int) -> None:
    text = _heading_line_text(lines[index])
    lines[index] = f"{'#' * level} {text}"


def _ordered_pack_hints(
    pack: RecognitionPack,
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """按稳定 ID 取出上下文标题和正文可见标题。"""

    hint_by_id: dict[str, tuple[int, str]] = {}
    for raw in pack.heading_hints:
        heading_id = str(raw.get("heading_id", ""))
        level = int(raw.get("level", 0))
        if heading_id and 1 <= level <= 6 and heading_id not in hint_by_id:
            hint_by_id[heading_id] = (level, str(raw.get("role", "")))

    def collect(ids: tuple[str, ...]) -> list[tuple[str, int, str]]:
        result: list[tuple[str, int, str]] = []
        for heading_id in ids:
            value = hint_by_id.get(heading_id)
            if value is not None:
                result.append((heading_id, value[0], value[1]))
        return result

    return collect(pack.context_heading_ids), collect(pack.visible_heading_ids)


def _is_short_plain_heading(text: str) -> bool:
    value = text.strip()
    return bool(
        value
        and len(value) <= 100
        and not value.startswith(("<", "|", chr(96) * 3))
        and value[-1:] not in "。；;，,"
    )


def _plain_level_matches(text: str, expected_level: int) -> bool:
    observed = _numbered_heading_level(text)
    if observed == expected_level:
        return True
    # 保险释义常用“一、保险人”，在 H2 章节下实际是 H3。
    return expected_level == 3 and bool(
        re.match(r"^[一二三四五六七八九十百]+[、．.]", text.strip())
    )


def _looks_like_metadata_or_instruction(text: str) -> bool:
    value = text.strip()
    return bool(
        re.search(r"注册|备案|编号|扫描|查询|验证|以下简称", value)
        or re.match(r"^(注意|为了|请(?:仔细|扫描|查询|阅读))", value)
        or re.fullmatch(r"[（(].*[）)]", value)
        or re.search(r"〔[^〕]+〕.*号$", value)
    )


def _pick_h1_candidate(lines: list[str], used: set[int]) -> int | None:
    """在第一个正文小标题前，保守寻找文档主标题行。"""

    limit = min(len(lines), 48)
    for index, line in enumerate(lines[:limit]):
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if match and (_numbered_heading_level(match.group(1)) or 0) >= 3:
            limit = index
            break

    candidates: list[tuple[int, int, int]] = []
    for index, line in enumerate(lines[:limit]):
        if index in used:
            continue
        text = _heading_line_text(line)
        if not _is_short_plain_heading(text):
            continue
        if _looks_like_metadata_or_instruction(text):
            continue
        score = 0
        if re.match(r"^#{1,6}\s+", line.strip()):
            score += 5
        if "条款" in text:
            score += 4
        if "保险" in text:
            score += 2
        next_text = ""
        for following in lines[index + 1 : min(limit, index + 5)]:
            if following.strip():
                next_text = _heading_line_text(following)
                break
        if re.search(r"注册|备案|编号|版|〔[^〕]+〕.*号$", next_text):
            score += 5
        if score > 0:
            candidates.append((score, len(text), -index))
    if not candidates:
        return None
    _, _, negative_index = max(candidates)
    return -negative_index


def repair_pack_heading_levels(markdown: str, pack: RecognitionPack) -> str:
    """用请求包中的稳定标题 ID 校准层级，并恢复复制标题的普通文本形式。

    只处理 manifest 明确列出的标题预算，不会把全文所有编号或短句提升为标题。
    """

    if not markdown.strip() or not pack.heading_hints:
        return markdown.strip()
    lines = markdown.strip().splitlines()
    context_hints, visible_hints = _ordered_pack_hints(pack)
    used: set[int] = set()
    cursor = 0

    # 复制到图片顶部的 H2/H3 必然位于响应开头；即使模型没加井号，也可按
    # context_heading_ids 恢复，从而让后续稳定 ID 去重真正生效。
    for _, level, _ in context_hints:
        chosen: int | None = None
        for index in range(cursor, min(len(lines), cursor + 24)):
            text = _heading_line_text(lines[index])
            if not text:
                continue
            if re.match(r"^#{1,6}\s+", lines[index].strip()) or _is_short_plain_heading(text):
                chosen = index
                break
        if chosen is None:
            break
        _force_heading_level(lines, chosen, level)
        used.add(chosen)
        cursor = chosen + 1

    # 目录标题位置稳定且文字特征明确；目录条目本身仍保持普通文本或表格。
    for heading_id, level, role in visible_hints:
        if level != 1 or "toc" not in role.lower():
            continue
        for index, line in enumerate(lines[:40]):
            text = _heading_line_text(line)
            if index not in used and "目录" in text and len(text) <= 40:
                _force_heading_level(lines, index, level)
                used.add(index)
                break

    # 正文主标题通常没有编号。只在 manifest 明确给出 H1 时，从正文小标题前
    # 选择带文档标题特征的短行；找不到可靠候选则保持原样，不强行猜测。
    h1_hints = [
        item for item in visible_hints
        if item[1] == 1 and "toc" not in item[2].lower()
    ]
    for _ in h1_hints:
        chosen = _pick_h1_candidate(lines, used)
        if chosen is None:
            break
        _force_heading_level(lines, chosen, 1)
        used.add(chosen)

    body_hints = [item for item in visible_hints if item[1] != 1]
    search_from = 0
    for _, expected_level, _ in body_hints:
        chosen = None
        for index in range(search_from, len(lines)):
            if index in used:
                continue
            text = _heading_line_text(lines[index])
            if not text or not _is_short_plain_heading(text):
                continue
            if _plain_level_matches(text, expected_level):
                chosen = index
                break
        if chosen is None:
            # manifest 与模型编号形态不一致时，宁可保留模型层级，也不盲目
            # 把一个正常 H2/H3 强行降成更深标题。
            continue
        _force_heading_level(lines, chosen, expected_level)
        used.add(chosen)
        search_from = chosen + 1

    return "\n".join(lines).strip()


def strip_repeated_context_headings(
    markdown: str,
    pack: RecognitionPack,
    seen_heading_text: dict[str, str],
) -> str:
    """按稳定标题 ID 删除续块重复祖先标题；文字相似度只作为安全校验。"""

    lines = markdown.strip().splitlines()
    remove: set[int] = set()

    # 一个复制标题框偶尔会被模型读成多行，行数不一定等于稳定标题 ID 数。
    # 因此额外按 context ID 组合记录“开头短行块”；同一组合再次出现时，
    # 删除逐行相同的最长前缀。这样既能去掉普通文本形式的祖先标题，也不会
    # 对正文中稍后出现的同名短句做全局删除。
    if pack.context_heading_ids:
        prefix_rows: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            text = _heading_line_text(line)
            if not text:
                continue
            if not _is_short_plain_heading(text):
                break
            prefix_rows.append((index, _normalized_heading_text(text)))
            if len(prefix_rows) >= 12:
                break
        block_key = "__context_block__:" + "|".join(pack.context_heading_ids)
        previous_block = seen_heading_text.get(block_key)
        if previous_block is None:
            seen_heading_text[block_key] = "\x1f".join(
                signature for _, signature in prefix_rows
            )
        else:
            previous_signatures = set(previous_block.split("\x1f"))
            for line_index, signature in prefix_rows:
                # 同一复制标题框有时会漏识别其中一行，所以不能只做
                # 严格逐位比较；但仍只检查响应最开头的连续短行。
                if not signature or signature not in previous_signatures:
                    break
                remove.add(line_index)

    heading_rows = [
        (index, match.group(2).strip())
        for index, line in enumerate(lines)
        if index not in remove
        and (match := re.match(r"^(#{1,6})\s+(.+?)\s*$", line))
    ]
    for position, heading_id in enumerate(pack.context_heading_ids):
        if position >= len(heading_rows):
            break
        line_index, text = heading_rows[position]
        normalized = _normalized_heading_text(text)
        previous = seen_heading_text.get(heading_id)
        if previous is None:
            seen_heading_text[heading_id] = normalized
            continue
        similarity = SequenceMatcher(None, previous, normalized).ratio()
        if similarity >= 0.62:
            remove.add(line_index)

    remaining_heading_rows = [item for item in heading_rows if item[0] not in remove]
    # 当前正文中真正可见的标题按原图顺序记录，供后续上下文条去重。
    offset = max(0, len(pack.context_heading_ids) - len(remove))
    for position, heading_id in enumerate(pack.visible_heading_ids):
        row_index = offset + position
        if row_index >= len(remaining_heading_rows):
            break
        seen_heading_text.setdefault(
            heading_id,
            _normalized_heading_text(remaining_heading_rows[row_index][1]),
        )

    output = [line for index, line in enumerate(lines) if index not in remove]
    # 删除被移除标题附近由模型产生的多余连续空行。
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def _numbered_heading_level(text: str) -> int | None:
    """从已经被 VLM 判断为标题的文字编号中推断层级。"""

    value = text.strip()
    arabic = re.match(r"^(\d+(?:\.\d+){0,5})(?:\s|[、．]|$)", value)
    if arabic:
        number = arabic.group(1)
        if "." not in number and int(number) > 99:
            return None
        return min(5, number.count(".") + 2)
    if re.match(r"^第[一二三四五六七八九十百0-9]+章", value):
        return 2
    if re.match(r"^第[一二三四五六七八九十百0-9]+[节条]", value):
        return 3
    if re.match(r"^[一二三四五六七八九十百]+[、．.]", value):
        return 2
    if re.match(r"^[（(][一二三四五六七八九十百]+[）)]", value):
        return 4
    if re.match(r"^[（(]?\d+[）)]", value):
        return 4
    return None


def normalize_markdown_heading_levels(markdown: str) -> str:
    """仅纠正 VLM 已输出的标题井号，不把普通正文擅自提升为标题。"""

    output: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            output.append(line)
            continue
        level = _numbered_heading_level(match.group(2))
        output.append(line if level is None else f"{'#' * level} {match.group(2)}")
    return "\n".join(output).strip()
