"""独立墨迹扫描、严格模型候选和文档内排版样式聚类。

YOLO 与墨迹是两条真正独立的证据链：YOLO 只提供达到 Title 正式阈值的
候选框；墨迹扫描直接读取每个检测窗口的完整责任区，不依赖任何 YOLO 框。
两者只在标题样式聚类阶段汇合。没有可靠 H2 时返回空 H2，让请求规划明确
回退 legacy，而不是从低分候选中强制制造章节边界。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .config import LongConfig
from .步骤001_数据定义 import DetectionWindow, Heading, LayoutBlock
from ..common.models import Box


@dataclass(frozen=True)
class InkLine:
    """不依赖小模型、从窗口完整墨迹中检测出的一条视觉文字行。"""

    id: str
    box: Box
    ink_density: float
    gap_above: int = 0
    gap_below: int = 0

    @property
    def height(self) -> int:
        return self.box.height

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_dict()
        return result


@dataclass(frozen=True)
class TitleEvidence:
    """一个模型标题组或墨迹独立候选的全部审计信息。"""

    candidate_id: str
    source: str
    member_ids: tuple[str, ...]
    box: Box
    model_confidence: float
    matched_ink_line_ids: tuple[str, ...]
    ink_line_height: float
    body_line_height: float
    height_ratio: float
    whitespace_ratio: float
    centered: bool
    strict_model_support: bool
    independent_ink_support: bool
    eligible_for_style: bool
    style_id: str | None = None
    style_rank: int | None = None
    final_heading_id: str | None = None
    final_level: int | None = None
    used_as_h2_boundary: bool = False
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["member_ids"] = list(self.member_ids)
        result["matched_ink_line_ids"] = list(self.matched_ink_line_ids)
        result["box"] = self.box.to_dict()
        return result


@dataclass(frozen=True)
class HeadingStyle:
    id: str
    candidate_ids: tuple[str, ...]
    median_height_ratio: float
    median_left_ratio: float
    median_model_confidence: float
    median_whitespace_ratio: float
    model_supported_count: int
    ink_only_count: int
    centered_count: int
    rank: int = -1
    selected_as_h2: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_ids"] = list(self.candidate_ids)
        return result


def _active_bands(active: np.ndarray, maximum_gap: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    last_active = -10
    for index, value in enumerate(active):
        if bool(value):
            if start is None:
                start = index
            last_active = index
        elif start is not None and index - last_active > maximum_gap:
            bands.append((start, last_active + 1))
            start = None
    if start is not None:
        bands.append((start, last_active + 1))
    return bands


def scan_independent_ink_lines(
    window_paths: list[Path],
    windows: list[DetectionWindow],
    image_width: int,
    config: LongConfig,
) -> tuple[list[InkLine], float]:
    """扫描窗口整幅墨迹，并用 ownership 保证每条原图文字行只保留一次。"""

    if len(window_paths) != len(windows):
        raise ValueError("检测窗口图片和窗口元数据数量不一致")
    lines: list[InkLine] = []
    for path, window in zip(window_paths, windows):
        with Image.open(path) as source:
            gray = np.asarray(source.convert("L"), dtype=np.uint8)
        ink = gray < config.semantic_ink_threshold
        active = (
            np.mean(ink, axis=1)
            >= config.semantic_full_width_active_ratio
        )
        for local_y1, local_y2 in _active_bands(
            active, config.semantic_line_merge_gap
        ):
            height = local_y2 - local_y1
            if height < config.semantic_min_ink_line_height:
                continue
            global_y1 = window.start_y + local_y1
            global_y2 = window.start_y + local_y2
            center_y = (global_y1 + global_y2) / 2
            if not (
                window.ownership_start_y
                <= center_y
                < window.ownership_end_y
            ):
                continue
            band = ink[local_y1:local_y2]
            columns = np.flatnonzero(np.any(band, axis=0))
            if not len(columns):
                continue
            x1 = int(columns[0])
            x2 = int(columns[-1]) + 1
            if x2 - x1 < image_width * config.semantic_min_ink_width_ratio:
                continue
            lines.append(
                InkLine(
                    id=f"ink_{len(lines):05d}",
                    box=Box(x1, global_y1, x2, global_y2),
                    ink_density=float(np.mean(band)),
                )
            )

    lines.sort(key=lambda item: (item.box.y1, item.box.x1))
    with_gaps: list[InkLine] = []
    for index, line in enumerate(lines):
        previous_end = lines[index - 1].box.y2 if index else 0
        next_start = lines[index + 1].box.y1 if index + 1 < len(lines) else line.box.y2
        with_gaps.append(
            replace(
                line,
                gap_above=max(0, line.box.y1 - previous_end),
                gap_below=max(0, next_start - line.box.y2),
            )
        )
    if not with_gaps:
        return [], 1.0
    # 标题只占少数，全部视觉行高度的中位数比依赖 Text 框更稳定。
    body_height = float(median(line.height for line in with_gaps))
    return with_gaps, max(1.0, body_height)


def _axis_overlap(first1: int, first2: int, second1: int, second2: int) -> int:
    return max(0, min(first2, second2) - max(first1, second1))


def _union_box(blocks: list[LayoutBlock]) -> Box:
    return Box(
        min(item.box.x1 for item in blocks),
        min(item.box.y1 for item in blocks),
        max(item.box.x2 for item in blocks),
        max(item.box.y2 for item in blocks),
    )


def _strict_title_groups(
    blocks: list[LayoutBlock], body_height: float, config: LongConfig
) -> list[list[LayoutBlock]]:
    """严格执行 Title 0.60，并只合并高度/横向范围都很相近的多行标题。"""

    titles = sorted(
        (
            item
            for item in blocks
            if item.label == "Title"
            and item.confidence >= config.title_confidence
        ),
        key=lambda item: (item.box.y1, item.box.x1),
    )
    groups: list[list[LayoutBlock]] = []
    for title in titles:
        if not groups:
            groups.append([title])
            continue
        previous = groups[-1][-1]
        gap = title.box.y1 - previous.box.y2
        horizontal = _axis_overlap(
            title.box.x1, title.box.x2, previous.box.x1, previous.box.x2
        )
        horizontal_ratio = horizontal / max(
            1, min(title.box.width, previous.box.width)
        )
        height_ratio = min(title.box.height, previous.box.height) / max(
            1, max(title.box.height, previous.box.height)
        )
        if (
            0 <= gap <= body_height * config.semantic_multiline_gap_ratio
            and horizontal_ratio >= config.semantic_multiline_overlap_ratio
            and height_ratio >= 0.65
        ):
            groups[-1].append(title)
        else:
            groups.append([title])
    return groups


def _is_centered(box: Box, image_width: int, config: LongConfig) -> bool:
    if box.width > image_width * config.semantic_center_max_width_ratio:
        return False
    left = box.x1
    right = image_width - box.x2
    return abs(left - right) <= image_width * config.center_tolerance_ratio


def _has_balanced_center(box: Box, image_width: int, config: LongConfig) -> bool:
    """只判断左右是否对称，不因标题较宽就把多行文档主标题排除。"""

    left = box.x1
    right = image_width - box.x2
    return (
        box.width <= image_width * config.semantic_toc_anchor_max_width_ratio
        and abs(left - right) <= image_width * config.center_tolerance_ratio
    )


def _detect_toc_region(
    blocks: list[LayoutBlock],
    lines: list[InkLine],
    body_height: float,
    image_width: int,
    image_height: int,
    config: LongConfig,
) -> dict[str, Any] | None:
    """寻找“目录标题在前、文档主标题在后”的双居中锚点。

    这里只借用低阈值 Title 框找第二个大范围主标题，不把它提升为正式模型
    标题。为了避免把正文中的居中句子误当目录边界，第一锚点必须达到正式
    0.60 阈值且位于图像顶部；第二锚点必须明显更高，并且二者之间至少包含
    若干视觉文字行。
    """

    anchors = sorted(
        (
            item
            for item in blocks
            if item.label == "Title"
            and item.confidence >= config.yolo_base_confidence
            and _has_balanced_center(item.box, image_width, config)
        ),
        key=lambda item: (item.box.y1, item.box.x1),
    )
    if len(anchors) < 2:
        return None
    top_limit = min(config.window_height, int(image_height * 0.04))
    first_choices = [
        item
        for item in anchors
        if item.box.y1 <= top_limit
        and item.confidence >= config.title_confidence
        and item.box.height >= body_height * config.semantic_min_heading_ratio
    ]
    for first in first_choices:
        for second in anchors:
            if second.box.y1 <= first.box.y2 + body_height * 8:
                continue
            if second.box.y1 > first.box.y1 + config.window_height * 4:
                break
            if (
                second.box.height
                < body_height * config.semantic_toc_second_anchor_min_height_ratio
            ):
                continue
            between = [
                line
                for line in lines
                if first.box.y2 <= line.box.y1 < second.box.y1
            ]
            if len(between) < config.semantic_toc_min_line_count:
                continue
            return {
                "detected": True,
                "toc_box": Box(0, first.box.y1, image_width, second.box.y1),
                "toc_title_box": first.box,
                "document_h1_box": second.box,
                "toc_title_block_id": first.id,
                "document_h1_block_id": second.id,
                "line_count": len(between),
            }
    return None


def _matching_ink_lines(
    box: Box, lines: list[InkLine], body_height: float
) -> list[InkLine]:
    candidates: list[tuple[float, InkLine]] = []
    for line in lines:
        if line.box.y1 > box.y2 + body_height or line.box.y2 < box.y1 - body_height:
            continue
        overlap = _axis_overlap(box.y1, box.y2, line.box.y1, line.box.y2)
        distance = abs((box.y1 + box.y2) / 2 - (line.box.y1 + line.box.y2) / 2)
        if overlap > 0 or distance <= body_height:
            candidates.append((distance, line))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    nearest = candidates[0][0]
    return [line for distance, line in candidates if distance <= nearest + body_height * 0.35]


def _initial_candidates(
    blocks: list[LayoutBlock],
    lines: list[InkLine],
    body_height: float,
    image_width: int,
    config: LongConfig,
) -> list[TitleEvidence]:
    candidates: list[TitleEvidence] = []
    matched_ink_ids: set[str] = set()
    for index, group in enumerate(_strict_title_groups(blocks, body_height, config)):
        box = _union_box(group)
        matched = _matching_ink_lines(box, lines, body_height)
        evidence_box = (
            Box(
                min([box.x1, *[item.box.x1 for item in matched]]),
                min([box.y1, *[item.box.y1 for item in matched]]),
                max([box.x2, *[item.box.x2 for item in matched]]),
                max([box.y2, *[item.box.y2 for item in matched]]),
            )
            if matched else box
        )
        matched_ink_ids.update(item.id for item in matched)
        line_height = (
            float(median(item.height for item in matched))
            if matched
            else 0.0
        )
        ratio = line_height / body_height if line_height else 0.0
        whitespace = (
            float(median((item.gap_above + item.gap_below) / body_height for item in matched))
            if matched
            else 0.0
        )
        eligible = bool(
            matched
            and config.semantic_model_title_min_ratio
            <= ratio
            <= config.semantic_title_max_height_ratio
        )
        if not matched:
            rejection = "未匹配到独立墨迹行"
        elif ratio < config.semantic_model_title_min_ratio:
            rejection = "实际墨迹字号与正文过于接近"
        elif ratio > config.semantic_title_max_height_ratio:
            rejection = "墨迹块过高，更像表格、图片或多行粘连区域"
        else:
            rejection = None
        candidates.append(
            TitleEvidence(
                candidate_id=f"model_title_{index:04d}",
                source="model",
                member_ids=tuple(item.id for item in group),
                box=evidence_box,
                model_confidence=sum(item.confidence for item in group) / len(group),
                matched_ink_line_ids=tuple(item.id for item in matched),
                ink_line_height=line_height,
                body_line_height=body_height,
                height_ratio=ratio,
                whitespace_ratio=whitespace,
                centered=_is_centered(evidence_box, image_width, config),
                strict_model_support=True,
                independent_ink_support=bool(matched),
                eligible_for_style=eligible,
                rejection_reason=rejection,
            )
        )

    # 墨迹只在非常显著且上下有额外留白时独立提出候选，避免普通粗体正文
    # 大量进入标题树。与模型候选已匹配的墨迹行不重复创建。
    for line in lines:
        if line.id in matched_ink_ids:
            continue
        ratio = line.height / body_height
        whitespace = (line.gap_above + line.gap_below) / body_height
        if ratio < config.semantic_ink_only_title_ratio:
            continue
        if whitespace < config.semantic_ink_only_min_whitespace_ratio:
            continue
        too_tall = ratio > config.semantic_title_max_height_ratio
        candidates.append(
            TitleEvidence(
                candidate_id=f"ink_title_{line.id}",
                source="ink",
                member_ids=(),
                box=line.box,
                model_confidence=0.0,
                matched_ink_line_ids=(line.id,),
                ink_line_height=float(line.height),
                body_line_height=body_height,
                height_ratio=ratio,
                whitespace_ratio=whitespace,
                centered=_is_centered(line.box, image_width, config),
                strict_model_support=False,
                independent_ink_support=True,
                eligible_for_style=not too_tall,
                rejection_reason=(
                    "墨迹块过高，更像表格、图片或多行粘连区域"
                    if too_tall
                    else None
                ),
            )
        )
    return sorted(candidates, key=lambda item: (item.box.y1, item.box.x1))


def _exclude_toc_and_deduplicate(
    candidates: list[TitleEvidence],
    toc: dict[str, Any] | None,
    config: LongConfig,
) -> tuple[list[TitleEvidence], int, int]:
    """目录候选退出正文投票，并在原图坐标上删除滑窗重复候选。"""

    result = list(candidates)
    toc_rejected = 0
    if toc is not None:
        toc_box: Box = toc["toc_box"]
        for index, item in enumerate(result):
            if item.box.y2 > toc_box.y1 and item.box.y1 < toc_box.y2:
                if item.eligible_for_style:
                    toc_rejected += 1
                result[index] = replace(
                    item,
                    eligible_for_style=False,
                    rejection_reason="位于目录区域，只保留图像，不参与正文标题投票",
                )

    def evidence_priority(item: TitleEvidence) -> tuple[float, ...]:
        # 有严格模型支持优先；同类证据再比较置信度、留白和较小的框高度。
        return (
            float(item.strict_model_support),
            item.model_confidence,
            item.whitespace_ratio,
            -float(item.box.height),
        )

    accepted: list[int] = []
    overlap_rejected = 0
    order = sorted(
        range(len(result)),
        key=lambda index: (result[index].box.y1, result[index].box.x1),
    )
    for index in order:
        item = result[index]
        if not item.eligible_for_style:
            continue
        duplicate_index: int | None = None
        for other_index in accepted:
            other = result[other_index]
            vertical = _axis_overlap(
                item.box.y1, item.box.y2, other.box.y1, other.box.y2
            )
            horizontal = _axis_overlap(
                item.box.x1, item.box.x2, other.box.x1, other.box.x2
            )
            if (
                vertical / max(1, min(item.box.height, other.box.height))
                >= config.semantic_candidate_overlap_ratio
                and horizontal / max(1, min(item.box.width, other.box.width))
                >= config.semantic_candidate_overlap_ratio
            ):
                duplicate_index = other_index
                break
        if duplicate_index is None:
            accepted.append(index)
            continue
        other = result[duplicate_index]
        if evidence_priority(item) > evidence_priority(other):
            result[duplicate_index] = replace(
                other,
                eligible_for_style=False,
                rejection_reason=(
                    f"与 {item.candidate_id} 在原图中重叠，判为滑窗重复"
                ),
            )
            accepted.remove(duplicate_index)
            accepted.append(index)
        else:
            result[index] = replace(
                item,
                eligible_for_style=False,
                rejection_reason=(
                    f"与 {other.candidate_id} 在原图中重叠，判为滑窗重复"
                ),
            )
        overlap_rejected += 1
    return result, toc_rejected, overlap_rejected


def _cluster_candidates(
    candidates: list[TitleEvidence], image_width: int, config: LongConfig
) -> tuple[list[HeadingStyle], dict[str, str]]:
    eligible = [item for item in candidates if item.eligible_for_style]
    groups: list[list[TitleEvidence]] = []
    for candidate in sorted(
        eligible, key=lambda item: (-item.height_ratio, item.box.x1, item.box.y1)
    ):
        best_index: int | None = None
        best_distance = float("inf")
        for index, group in enumerate(groups):
            height = float(median(item.height_ratio for item in group))
            left = float(median(item.box.x1 / image_width for item in group))
            height_distance = abs(candidate.height_ratio - height) / max(height, 1e-6)
            left_distance = abs(candidate.box.x1 / image_width - left)
            if (
                height_distance <= config.semantic_style_height_tolerance
                and left_distance <= config.semantic_style_indent_tolerance
            ):
                distance = height_distance + left_distance
                if distance < best_distance:
                    best_index = index
                    best_distance = distance
        if best_index is None:
            groups.append([candidate])
        else:
            groups[best_index].append(candidate)

    styles: list[HeadingStyle] = []
    assignment: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda group: (
            -median(item.height_ratio for item in group),
            median(item.box.x1 / image_width for item in group),
        ),
    )
    for rank, group in enumerate(ordered):
        style_id = f"style_{rank:02d}"
        for item in group:
            assignment[item.candidate_id] = style_id
        styles.append(
            HeadingStyle(
                id=style_id,
                candidate_ids=tuple(item.candidate_id for item in group),
                median_height_ratio=float(median(item.height_ratio for item in group)),
                median_left_ratio=float(median(item.box.x1 / image_width for item in group)),
                median_model_confidence=float(median(item.model_confidence for item in group)),
                median_whitespace_ratio=float(median(item.whitespace_ratio for item in group)),
                model_supported_count=sum(item.strict_model_support for item in group),
                ink_only_count=sum(not item.strict_model_support for item in group),
                centered_count=sum(item.centered for item in group),
                rank=rank,
            )
        )
    return styles, assignment


def analyze_semantic_headings(
    blocks: list[LayoutBlock],
    image_width: int,
    window_paths: list[Path],
    windows: list[DetectionWindow],
    config: LongConfig,
) -> tuple[list[Heading], list[TitleEvidence], dict[str, Any]]:
    """按排版样式簇选择 H2；没有可信样式时明确返回零个 H2。"""

    lines, body_height = scan_independent_ink_lines(
        window_paths, windows, image_width, config
    )
    image_height = max((item.end_y for item in windows), default=0)
    toc = _detect_toc_region(
        blocks, lines, body_height, image_width, image_height, config
    )
    candidates = _initial_candidates(
        blocks, lines, body_height, image_width, config
    )
    candidates, toc_rejected_count, overlap_rejected_count = (
        _exclude_toc_and_deduplicate(candidates, toc, config)
    )
    styles, assignment = _cluster_candidates(candidates, image_width, config)
    by_id = {item.candidate_id: item for item in candidates}

    eligible_h2_styles = []
    for style in styles:
        members = [by_id[item] for item in style.candidate_ids]
        non_centered = [item for item in members if not item.centered]
        if not non_centered:
            continue
        if style.median_height_ratio < config.semantic_h2_min_style_ratio:
            continue
        has_model = any(item.strict_model_support for item in non_centered)
        has_strong_ink = any(
            item.height_ratio >= config.semantic_ink_only_title_ratio
            and item.whitespace_ratio >= config.semantic_ink_only_min_whitespace_ratio
            for item in non_centered
        )
        if has_model or has_strong_ink:
            eligible_h2_styles.append(style)

    selected_style_ids: set[str] = set()
    if eligible_h2_styles:
        best = max(
            eligible_h2_styles,
            key=lambda item: (
                item.median_height_ratio,
                -item.median_left_ratio,
                item.model_supported_count,
            ),
        )
        selected_style_ids = {
            item.id
            for item in eligible_h2_styles
            if (
                abs(item.median_height_ratio - best.median_height_ratio)
                / best.median_height_ratio
                <= config.semantic_h2_cluster_height_tolerance
                and item.median_left_ratio
                <= best.median_left_ratio + config.semantic_style_indent_tolerance
            )
        }
        styles = [
            replace(item, selected_as_h2=item.id in selected_style_ids)
            for item in styles
        ]

    h2_candidates = [
        item
        for item in candidates
        if item.eligible_for_style
        and assignment.get(item.candidate_id) in selected_style_ids
        and not item.centered
    ]
    first_h2_y = min((item.box.y1 for item in h2_candidates), default=10**18)
    subordinate_style_ids = [
        style.id
        for style in styles
        if style.id not in selected_style_ids
        and style.median_height_ratio >= config.semantic_model_title_min_ratio
        and any(
            not by_id[candidate_id].centered
            and by_id[candidate_id].box.y1 >= first_h2_y
            for candidate_id in style.candidate_ids
        )
    ]
    subordinate_level = {
        style_id: min(4, 3 + index)
        for index, style_id in enumerate(subordinate_style_ids)
    }

    headings: list[Heading] = []
    updated: list[TitleEvidence] = []
    current_h2: str | None = None
    current_h3: str | None = None
    counters = {1: 0, 2: 0, 3: 0, 4: 0}
    if toc is not None:
        # 目录标题与正文主标题是特殊锚点：它们用于切开目录，但不加入 H2
        # 样式投票，否则目录中的编号和字号会污染正文层级。
        headings.extend(
            [
                Heading(
                    id="semantic_toc_0000",
                    level=1,
                    role="semantic_toc",
                    box=toc["toc_title_box"],
                    parent_id=None,
                    confidence=1.0,
                    centered=True,
                    member_ids=(toc["toc_title_block_id"],),
                ),
                Heading(
                    id="semantic_h1_0000",
                    level=1,
                    role="semantic_h1",
                    box=toc["document_h1_box"],
                    parent_id=None,
                    confidence=1.0,
                    centered=True,
                    member_ids=(toc["document_h1_block_id"],),
                ),
            ]
        )
        counters[1] = 1
    for item in candidates:
        style_id = assignment.get(item.candidate_id)
        level: int | None = None
        parent_id: str | None = None
        role = "style_candidate"
        if (
            toc is None
            and item.centered
            and item.box.y1 < first_h2_y
            and item.height_ratio >= config.semantic_min_heading_ratio
        ):
            level = 1
            role = "semantic_h1"
        elif style_id in selected_style_ids and not item.centered:
            level = 2
            role = "semantic_h2"
        elif (
            current_h2 is not None
            and item.eligible_for_style
            and style_id in subordinate_level
        ):
            level = subordinate_level[style_id]
            role = f"semantic_h{level}_candidate"

        heading_id: str | None = None
        if level is not None:
            heading_id = f"semantic_h{level}_{counters[level]:04d}"
            counters[level] += 1
            if level == 2:
                current_h2 = heading_id
                current_h3 = None
            elif level == 3:
                parent_id = current_h2
                current_h3 = heading_id
            elif level >= 4:
                parent_id = current_h3 or current_h2
            headings.append(
                Heading(
                    id=heading_id,
                    level=level,
                    role=role,
                    box=item.box,
                    parent_id=parent_id,
                    confidence=(
                        item.model_confidence
                        if item.strict_model_support
                        else min(1.0, item.height_ratio / 2)
                    ),
                    centered=item.centered,
                    member_ids=item.member_ids,
                )
            )
        updated.append(
            replace(
                item,
                style_id=style_id,
                style_rank=next(
                    (style.rank for style in styles if style.id == style_id),
                    None,
                ),
                final_heading_id=heading_id,
                final_level=level,
                used_as_h2_boundary=level == 2,
                rejection_reason=(
                    item.rejection_reason
                    if level is not None or item.rejection_reason is not None
                    else "样式未被选为 H1/H2，且当前没有可归属的上级 H2"
                ),
            )
        )

    debug = {
        "method": "strict-general6+guarded-independent-ink+toc-isolation-v3",
        "body_ink_line_height": body_height,
        "ink_line_count": len(lines),
        "strict_model_title_count": sum(item.source == "model" for item in candidates),
        "ink_only_candidate_count": sum(item.source == "ink" for item in candidates),
        "eligible_candidate_count": sum(item.eligible_for_style for item in candidates),
        "h1_count": sum(item.role == "semantic_h1" for item in headings),
        "toc_count": sum(item.role == "semantic_toc" for item in headings),
        "h2_count": sum(item.level == 2 for item in headings),
        "h3_count": sum(item.level == 3 for item in headings),
        "h4_count": sum(item.level == 4 for item in headings),
        "selected_h2_style_ids": sorted(selected_style_ids),
        "fallback_required": not any(item.level == 2 for item in headings),
        "toc_rejected_candidate_count": toc_rejected_count,
        "overlap_rejected_candidate_count": overlap_rejected_count,
        "oversized_ink_rejected_count": sum(
            item.source == "ink"
            and item.height_ratio > config.semantic_title_max_height_ratio
            for item in candidates
        ),
        "toc_region": (
            {
                **{
                    key: value
                    for key, value in toc.items()
                    if not isinstance(value, Box)
                },
                "toc_box": toc["toc_box"].to_dict(),
                "toc_title_box": toc["toc_title_box"].to_dict(),
                "document_h1_box": toc["document_h1_box"].to_dict(),
            }
            if toc is not None
            else None
        ),
        "parameters": {
            "title_confidence": config.title_confidence,
            "ink_threshold": config.semantic_ink_threshold,
            "full_width_active_ratio": config.semantic_full_width_active_ratio,
            "model_title_min_ratio": config.semantic_model_title_min_ratio,
            "ink_only_title_ratio": config.semantic_ink_only_title_ratio,
            "title_max_height_ratio": config.semantic_title_max_height_ratio,
            "h2_min_style_ratio": config.semantic_h2_min_style_ratio,
            "candidate_overlap_ratio": config.semantic_candidate_overlap_ratio,
            "style_height_tolerance": config.semantic_style_height_tolerance,
            "style_indent_tolerance": config.semantic_style_indent_tolerance,
            "h2_cluster_height_tolerance": config.semantic_h2_cluster_height_tolerance,
        },
        "ink_lines": [item.to_dict() for item in lines],
        "styles": [item.to_dict() for item in styles],
        "candidates": [item.to_dict() for item in updated],
    }
    return sorted(headings, key=lambda item: (item.box.y1, item.level)), updated, debug


def save_semantic_audit_windows(
    window_paths: list[Path],
    windows: list[DetectionWindow],
    evidence: list[TitleEvidence],
    headings: list[Heading],
    debug: dict[str, Any],
    output_dir: Path,
) -> None:
    """分别保存独立墨迹、严格模型候选和最终层级三套窗口图。"""

    ink_lines = [
        InkLine(
            id=str(raw["id"]),
            box=Box.from_dict(raw["box"]),
            ink_density=float(raw["ink_density"]),
            gap_above=int(raw["gap_above"]),
            gap_below=int(raw["gap_below"]),
        )
        for raw in debug.get("ink_lines", [])
    ]
    directories = {
        "ink": output_dir / "004_独立墨迹行窗口图",
        "model": output_dir / "005_严格模型标题窗口图",
        "toc": output_dir / "006A_目录隔离窗口图",
        "rejected": output_dir / "006B_候选拒绝原因窗口图",
        "final": output_dir / "007_最终标题层级窗口图",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    for window_path, window in zip(window_paths, windows):
        with Image.open(window_path) as source:
            base = source.convert("RGB")
        for mode, target in directories.items():
            image = base.copy()
            draw = ImageDraw.Draw(image)
            if mode == "ink":
                items = [
                    (line.box, "#ff8800" if line.height >= debug["body_ink_line_height"] * 1.2 else "#777777", f"ink h={line.height}")
                    for line in ink_lines
                ]
            elif mode == "model":
                items = [
                    (item.box, "#ffaa00", f"{item.candidate_id} conf={item.model_confidence:.2f}")
                    for item in evidence
                    if item.source == "model"
                ]
            elif mode == "toc":
                raw_toc = debug.get("toc_region")
                items = [] if raw_toc is None else [
                    (
                        Box.from_dict(raw_toc["toc_box"]),
                        "#cc00cc",
                        "TOC：不参与正文标题投票",
                    ),
                    (
                        Box.from_dict(raw_toc["document_h1_box"]),
                        "#8a2be2",
                        "正文 H1 起点",
                    ),
                ]
            elif mode == "rejected":
                items = [
                    (
                        item.box,
                        "#cc00cc",
                        f"拒绝 {item.candidate_id}：{item.rejection_reason}",
                    )
                    for item in evidence
                    if not item.eligible_for_style and item.rejection_reason
                ]
            else:
                colors = {1: "#8a2be2", 2: "#e60000", 3: "#0066ff", 4: "#00aa55"}
                items = [
                    (item.box, colors.get(item.level, "#333333"), f"H{item.level} {item.id}")
                    for item in headings
                ]
            visible = [
                (box, color, label)
                for box, color, label in items
                if box.y2 > window.start_y and box.y1 < window.end_y
            ]
            if not visible:
                continue
            for box, color, label in visible:
                local = (
                    box.x1,
                    max(0, box.y1 - window.start_y),
                    box.x2,
                    min(image.height, box.y2 - window.start_y),
                )
                draw.rectangle(local, outline=color, width=3)
                draw.text((local[0] + 3, max(0, local[1] - 13)), label, fill=color)
            image.save(target / window_path.name, format="PNG", compress_level=4)
