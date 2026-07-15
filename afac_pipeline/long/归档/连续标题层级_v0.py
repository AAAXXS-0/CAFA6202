"""长图逻辑标题、层级树与二次语义切块。"""

from __future__ import annotations

from dataclasses import replace
from statistics import median

from ..config import LongConfig
from ..步骤001_数据定义 import Heading, LayoutBlock, SemanticPart, SemanticSegment
from ...common.models import Box


CONTENT_LABELS = {"Text", "Table", "Figure", "Equation", "Caption"}


def _union_box(blocks: list[LayoutBlock]) -> Box:
    return Box(
        min(block.box.x1 for block in blocks),
        min(block.box.y1 for block in blocks),
        max(block.box.x2 for block in blocks),
        max(block.box.y2 for block in blocks),
    )


def is_centered(box: Box, image_width: int, tolerance_ratio: float) -> bool:
    # 长正文框的中心也可能靠近页面中心；真正居中还要求左右边距对称且框宽适中。
    if box.width > image_width * 0.55:
        return False
    left_margin = box.x1
    right_margin = image_width - box.x2
    return abs(left_margin - right_margin) <= image_width * tolerance_ratio


def merge_multiline_titles(
    blocks: list[LayoutBlock], image_width: int, config: LongConfig
) -> list[LayoutBlock]:
    """把被 general6 拆成多行的同一个标题合成逻辑 Title。

    合并条件故意严格：间距必须明显小于标题高度，同时宽度和中心位置相近。
    这样可以尽量避免把用户规则中的“连续 H2/H3”错误合成一个标题。
    """

    titles = sorted(
        (block for block in blocks if block.label == "Title"),
        key=lambda item: (item.box.y1, item.box.x1),
    )
    if not titles:
        return []
    typical_height = max(1.0, median(block.box.height for block in titles))
    groups: list[list[LayoutBlock]] = [[titles[0]]]
    for title in titles[1:]:
        previous = groups[-1][-1]
        gap = title.box.y1 - previous.box.y2
        width_ratio = min(title.box.width, previous.box.width) / max(
            1, max(title.box.width, previous.box.width)
        )
        center_distance = abs(title.center_x - previous.center_x)
        both_centered = is_centered(
            title.box, image_width, config.center_tolerance_ratio
        ) and is_centered(previous.box, image_width, config.center_tolerance_ratio)
        height_ratio = min(title.box.height, previous.box.height) / max(
            1, max(title.box.height, previous.box.height)
        )
        should_merge = (
            0 <= gap <= typical_height * config.logical_title_gap_ratio
            and height_ratio >= 0.55
            and (
                both_centered
                or (
                    width_ratio >= config.logical_title_width_ratio
                    and center_distance <= image_width * 0.08
                )
            )
        )
        if should_merge:
            groups[-1].append(title)
        else:
            groups.append([title])

    logical: list[LayoutBlock] = []
    for index, group in enumerate(groups):
        logical.append(
            LayoutBlock(
                id=f"logical_title_{index:04d}",
                label="Title",
                box=_union_box(group),
                confidence=sum(item.confidence for item in group) / len(group),
                source_window=min(item.source_window for item in group),
                member_ids=tuple(item.id for item in group),
            )
        )
    return logical


def has_content_between(
    first: LayoutBlock | Heading,
    second: LayoutBlock | Heading,
    blocks: list[LayoutBlock],
) -> bool:
    """判断两个标题之间是否有正文、表格、图片、公式或注释。"""

    lower = first.box.y2
    upper = second.box.y1
    if upper <= lower:
        return False
    for block in blocks:
        if block.label not in CONTENT_LABELS:
            continue
        center_y = block.center_y
        if lower < center_y < upper:
            return True
    return False


def group_consecutive_titles(
    titles: list[LayoutBlock], blocks: list[LayoutBlock]
) -> list[list[LayoutBlock]]:
    """连续 Title 组：中间无内容，并限制最大空白距离以抵御 Text 漏检。"""

    if not titles:
        return []
    typical_height = max(1.0, median(title.box.height for title in titles))
    max_empty_gap = max(240.0, typical_height * 4.0)
    groups: list[list[LayoutBlock]] = [[titles[0]]]
    for title in titles[1:]:
        previous = groups[-1][-1]
        gap = title.box.y1 - previous.box.y2
        if gap <= max_empty_gap and not has_content_between(previous, title, blocks):
            groups[-1].append(title)
        else:
            groups.append([title])
    return groups


def infer_heading_hierarchy(
    blocks: list[LayoutBlock], image_width: int, config: LongConfig
) -> tuple[list[LayoutBlock], list[Heading]]:
    """按用户规则推断目录标题、正文 H1、H2 与 H3。"""

    logical_titles = merge_multiline_titles(blocks, image_width, config)
    if not logical_titles:
        return [], []
    all_runs = group_consecutive_titles(logical_titles, blocks)

    # 正文层级通常从首个非居中的连续标题组开始；正文 H1 取其前方最后一个
    # 居中标题。若没有连续组，则使用全文最后一个可信居中标题。
    first_structural_y: int | None = None
    for run in all_runs:
        if len(run) >= 2 and not is_centered(
            run[0].box, image_width, config.center_tolerance_ratio
        ):
            first_structural_y = run[0].box.y1
            break
    centered_candidates = [
        title
        for title in logical_titles
        if is_centered(title.box, image_width, config.center_tolerance_ratio)
        and (first_structural_y is None or title.box.y1 < first_structural_y)
    ]
    if not centered_candidates:
        centered_candidates = [
            title
            for title in logical_titles
            if is_centered(title.box, image_width, config.center_tolerance_ratio)
        ]

    headings: list[Heading] = []
    body_h1: LayoutBlock | None = centered_candidates[-1] if centered_candidates else None
    toc_title: LayoutBlock | None = (
        centered_candidates[-2] if len(centered_candidates) >= 2 else None
    )
    if toc_title is not None:
        headings.append(
            Heading(
                id="toc_title",
                level=1,
                role="toc_title",
                box=toc_title.box,
                parent_id=None,
                confidence=toc_title.confidence,
                centered=True,
                member_ids=toc_title.member_ids,
            )
        )
    if body_h1 is not None:
        headings.append(
            Heading(
                id="body_h1",
                level=1,
                role="body_h1",
                box=body_h1.box,
                parent_id=None,
                confidence=body_h1.confidence,
                centered=True,
                member_ids=body_h1.member_ids,
            )
        )

    body_start = body_h1.box.y2 if body_h1 is not None else 0
    body_titles = [title for title in logical_titles if title.box.y1 >= body_start]
    body_runs = group_consecutive_titles(body_titles, blocks)
    current_h2_id: str | None = None
    h2_index = 0
    h3_index = 0
    for run in body_runs:
        if len(run) >= 2:
            first = run[0]
            current_h2_id = f"h2_{h2_index:04d}"
            headings.append(
                Heading(
                    id=current_h2_id,
                    level=2,
                    role="h2",
                    box=first.box,
                    parent_id="body_h1" if body_h1 is not None else None,
                    confidence=first.confidence,
                    centered=is_centered(first.box, image_width, config.center_tolerance_ratio),
                    member_ids=first.member_ids,
                )
            )
            h2_index += 1
            for title in run[1:]:
                heading_id = f"h3_{h3_index:04d}"
                headings.append(
                    Heading(
                        id=heading_id,
                        level=3,
                        role="h3",
                        box=title.box,
                        parent_id=current_h2_id,
                        confidence=title.confidence,
                        centered=is_centered(
                            title.box, image_width, config.center_tolerance_ratio
                        ),
                        member_ids=title.member_ids,
                    )
                )
                h3_index += 1
            continue

        title = run[0]
        if current_h2_id is None:
            current_h2_id = f"h2_{h2_index:04d}"
            headings.append(
                Heading(
                    id=current_h2_id,
                    level=2,
                    role="h2_fallback",
                    box=title.box,
                    parent_id="body_h1" if body_h1 is not None else None,
                    confidence=title.confidence,
                    centered=is_centered(title.box, image_width, config.center_tolerance_ratio),
                    member_ids=title.member_ids,
                )
            )
            h2_index += 1
        else:
            heading_id = f"h3_{h3_index:04d}"
            headings.append(
                Heading(
                    id=heading_id,
                    level=3,
                    role="h3",
                    box=title.box,
                    parent_id=current_h2_id,
                    confidence=title.confidence,
                    centered=is_centered(title.box, image_width, config.center_tolerance_ratio),
                    member_ids=title.member_ids,
                )
            )
            h3_index += 1

    return logical_titles, sorted(headings, key=lambda item: (item.box.y1, item.level))


def build_semantic_segments(
    image_height: int,
    blocks: list[LayoutBlock],
    headings: list[Heading],
) -> list[SemanticSegment]:
    """把标题层级树转换为不遗漏原图纵向内容的语义段。"""

    toc = next((heading for heading in headings if heading.role == "toc_title"), None)
    h1 = next((heading for heading in headings if heading.role == "body_h1"), None)
    body_headings = [heading for heading in headings if heading.level in {2, 3}]
    h2_headings = [heading for heading in body_headings if heading.level == 2]
    segments: list[SemanticSegment] = []

    first_special_y = toc.box.y1 if toc is not None else (h1.box.y1 if h1 is not None else 0)
    if first_special_y > 0:
        segments.append(SemanticSegment("front_matter", "front_matter", 0, first_special_y))
    if toc is not None and h1 is not None and toc.box.y1 < h1.box.y1:
        segments.append(
            SemanticSegment(
                "toc",
                "toc",
                toc.box.y1,
                h1.box.y1,
                expected_heading_levels=(1,),
            )
        )

    body_start = h1.box.y1 if h1 is not None else first_special_y
    first_h2_y = h2_headings[0].box.y1 if h2_headings else image_height
    if body_start < first_h2_y:
        segments.append(
            SemanticSegment(
                "body_intro",
                "body_intro",
                body_start,
                first_h2_y,
                h1_id=h1.id if h1 else None,
                expected_heading_levels=(1,) if h1 else (),
            )
        )

    for h2_index, h2 in enumerate(h2_headings):
        h2_end = (
            h2_headings[h2_index + 1].box.y1
            if h2_index + 1 < len(h2_headings)
            else image_height
        )
        children = [
            heading
            for heading in body_headings
            if heading.level == 3
            and heading.parent_id == h2.id
            and h2.box.y1 <= heading.box.y1 < h2_end
        ]
        if not children:
            segments.append(
                SemanticSegment(
                    f"{h2.id}_body",
                    "h2_body",
                    h2.box.y1,
                    h2_end,
                    h1_id=h1.id if h1 else None,
                    h2_id=h2.id,
                    expected_heading_levels=(2,),
                )
            )
            continue

        first_child = children[0]
        content_before_first = has_content_between(h2, first_child, blocks)
        if content_before_first:
            segments.append(
                SemanticSegment(
                    f"{h2.id}_intro",
                    "h2_intro",
                    h2.box.y1,
                    first_child.box.y1,
                    h1_id=h1.id if h1 else None,
                    h2_id=h2.id,
                    expected_heading_levels=(2,),
                )
            )

        for child_index, child in enumerate(children):
            child_start = (
                h2.box.y1
                if child_index == 0 and not content_before_first
                else child.box.y1
            )
            child_end = (
                children[child_index + 1].box.y1
                if child_index + 1 < len(children)
                else h2_end
            )
            levels = (2, 3) if child_start == h2.box.y1 else (3,)
            segments.append(
                SemanticSegment(
                    f"{child.id}_body",
                    "h3_body",
                    child_start,
                    child_end,
                    h1_id=h1.id if h1 else None,
                    h2_id=h2.id,
                    h3_id=child.id,
                    expected_heading_levels=levels,
                )
            )

    if not segments:
        segments.append(SemanticSegment("whole_document", "body", 0, image_height))
    return [segment for segment in segments if segment.end_y > segment.start_y]


def _safe_boundaries(
    start_y: int,
    end_y: int,
    blocks: list[LayoutBlock],
    minimum_gap: int = 16,
) -> list[int]:
    relevant = sorted(
        (
            block.box
            for block in blocks
            if block.box.y2 > start_y and block.box.y1 < end_y
        ),
        key=lambda box: (box.y1, box.y2),
    )
    boundaries: list[int] = []
    previous_end = start_y
    for box in relevant:
        if box.y1 - previous_end >= minimum_gap:
            boundaries.append(round((previous_end + box.y1) / 2))
        previous_end = max(previous_end, box.y2)
    if end_y - previous_end >= minimum_gap:
        boundaries.append(round((previous_end + end_y) / 2))
    return boundaries


def _split_ranges(
    segment: SemanticSegment,
    blocks: list[LayoutBlock],
    config: LongConfig,
) -> list[tuple[int, int]]:
    if segment.height <= config.max_vlm_height:
        return [(segment.start_y, segment.end_y)]
    candidates = _safe_boundaries(segment.start_y, segment.end_y, blocks)
    ranges: list[tuple[int, int]] = []
    cursor = segment.start_y
    half_overlap = config.vlm_overlap // 2
    while segment.end_y - cursor > config.max_vlm_height:
        preferred = cursor + config.max_vlm_height - half_overlap
        lower = max(cursor + config.minimum_part_height, preferred - config.safe_cut_search)
        upper = min(preferred, segment.end_y - config.minimum_part_height)
        available = [value for value in candidates if lower <= value <= upper]
        boundary = max(available) if available else preferred
        part_end = min(segment.end_y, boundary + half_overlap)
        if part_end - cursor > config.max_vlm_height:
            part_end = cursor + config.max_vlm_height
            boundary = part_end - half_overlap
        ranges.append((cursor, part_end))
        next_cursor = max(cursor + 1, boundary - half_overlap)
        if next_cursor <= cursor:
            raise RuntimeError(f"语义段 {segment.id} 的切块没有前进")
        cursor = next_cursor
    ranges.append((cursor, segment.end_y))
    return ranges


def attach_physical_parts(
    segments: list[SemanticSegment],
    blocks: list[LayoutBlock],
    image_width: int,
    config: LongConfig,
) -> list[SemanticSegment]:
    """为每个语义段增加不超过视觉模型限制的物理图片切片。"""

    completed: list[SemanticSegment] = []
    for segment_index, segment in enumerate(segments):
        ranges = _split_ranges(segment, blocks, config)
        parts: list[SemanticPart] = []
        for part_index, (start_y, end_y) in enumerate(ranges):
            levels = segment.expected_heading_levels if part_index == 0 else ()
            if segment.h2_id and segment.h3_id:
                relative_path = (
                    f"{segment.h2_id}/{segment.h3_id}/part_{part_index:03d}.png"
                )
            elif segment.h2_id:
                relative_path = f"{segment.h2_id}/{segment.role}/part_{part_index:03d}.png"
            else:
                relative_path = f"_document/{segment.role}/part_{part_index:03d}.png"
            parts.append(
                SemanticPart(
                    id=f"segment_{segment_index:04d}_part_{part_index:03d}",
                    segment_id=segment.id,
                    role=segment.role,
                    source_box=Box(0, start_y, image_width, end_y),
                    part_index=part_index,
                    part_count=len(ranges),
                    h1_id=segment.h1_id,
                    h2_id=segment.h2_id,
                    h3_id=segment.h3_id,
                    expected_heading_levels=levels,
                    file_name=relative_path,
                )
            )
        completed.append(replace(segment, parts=parts))
    return completed
