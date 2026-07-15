"""融合小模型、墨迹字号和全文顺序，生成保守的 H2/H3 章节锚点。

本模块不把整段 H2 再送入小模型。general6 始终只处理固定检测窗口；这里
读取已经映射回原图的版面框，并在窗口小图中测量标题与正文的相对墨迹行高。
所有阈值都使用当前文档的相对量，不按测试图片文件名或绝对坐标写特殊规则。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import LongConfig
from ..步骤001_数据定义 import DetectionWindow, Heading, LayoutBlock
from .连续标题层级_v0 import (
    group_consecutive_titles,
    infer_heading_hierarchy,
    is_centered,
)


@dataclass(frozen=True)
class HeadingEvidence:
    """一个逻辑标题的可审计证据。

    numbering_hint 暂时为空：准备阶段默认不加载 OCR。后续若接入标题专用 OCR，
    可在不改变层级接口的前提下补入编号证据。
    """

    title_id: str
    member_ids: tuple[str, ...]
    y1: int
    y2: int
    model_confidence: float
    ink_line_height: float
    body_line_height: float
    text_height_ratio: float
    left_prominence: float
    centered: bool
    consecutive_run_start: bool
    baseline_level: int | None
    baseline_role: str | None
    numbering_hint: str | None
    score_model: float
    score_ink: float
    score_left: float
    score_sequence: float
    score_baseline: float
    h2_score: float
    final_level: int
    final_confidence: float
    used_as_h2_boundary: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["member_ids"] = list(self.member_ids)
        return result


def _active_line_heights(
    gray: Image.Image,
    *,
    ink_threshold: int,
    active_row_ratio: float,
) -> list[int]:
    """从局部灰度图中估算每一行真实文字的墨迹高度。"""

    pixels = np.asarray(gray, dtype=np.uint8)
    if pixels.ndim != 2 or pixels.size == 0:
        return []
    active = np.mean(pixels < ink_threshold, axis=1) >= active_row_ratio
    bands: list[tuple[int, int]] = []
    start: int | None = None
    # 允许抗锯齿造成的一像素断口，但不把两行正文合并成一个大黑块。
    last_active = -10
    for index, value in enumerate(active):
        if bool(value):
            if start is None:
                start = index
            last_active = index
        elif start is not None and index - last_active > 1:
            bands.append((start, last_active + 1))
            start = None
    if start is not None:
        bands.append((start, last_active + 1))
    return [end - start for start, end in bands if end > start]


def measure_layout_ink(
    blocks: list[LayoutBlock],
    window_paths: list[Path],
    windows: list[DetectionWindow],
    config: LongConfig,
) -> tuple[dict[str, float], float]:
    """按窗口一次解码，测量 Title/Text 框中的实际墨迹行高。"""

    by_window: dict[int, list[LayoutBlock]] = {}
    for block in blocks:
        if block.label not in {"Title", "Text"}:
            continue
        if block.label == "Title" or block.confidence >= config.text_confidence:
            by_window.setdefault(block.source_window, []).append(block)

    path_by_index = {window.index: path for path, window in zip(window_paths, windows)}
    window_by_index = {window.index: window for window in windows}
    measurements: dict[str, float] = {}
    body_lines: list[int] = []
    for window_index, current_blocks in by_window.items():
        path = path_by_index.get(window_index)
        window = window_by_index.get(window_index)
        if path is None or window is None:
            continue
        with Image.open(path) as source:
            gray = source.convert("L")
            for block in current_blocks:
                x1 = max(0, block.box.x1)
                x2 = min(gray.width, block.box.x2)
                y1 = max(0, block.box.y1 - window.start_y)
                y2 = min(gray.height, block.box.y2 - window.start_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                heights = _active_line_heights(
                    gray.crop((x1, y1, x2, y2)),
                    ink_threshold=config.semantic_ink_threshold,
                    active_row_ratio=config.semantic_active_row_ratio,
                )
                if not heights:
                    continue
                value = float(median(heights))
                measurements[block.id] = value
                if block.label == "Text":
                    body_lines.extend(heights)

    if body_lines:
        body_height = float(median(body_lines))
    else:
        title_values = list(measurements.values())
        body_height = float(median(title_values)) if title_values else 1.0
    return measurements, max(1.0, body_height)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def analyze_semantic_headings(
    blocks: list[LayoutBlock],
    image_width: int,
    window_paths: list[Path],
    windows: list[DetectionWindow],
    config: LongConfig,
) -> tuple[list[Heading], list[HeadingEvidence], dict[str, Any]]:
    """生成高精度 H2 边界，并把其余正文标题保守地归为 H3 候选。"""

    logical_titles, baseline = infer_heading_hierarchy(blocks, image_width, config)
    raw_ink, body_height = measure_layout_ink(blocks, window_paths, windows, config)
    baseline_by_members = {
        tuple(item.member_ids): item for item in baseline if item.member_ids
    }
    runs = group_consecutive_titles(logical_titles, blocks)
    run_starts = {run[0].id for run in runs if len(run) >= 2}
    left_values = [title.box.x1 for title in logical_titles]
    left_min = min(left_values, default=0)
    left_max = max(left_values, default=left_min)
    left_span = max(1, left_max - left_min)

    provisional: list[dict[str, Any]] = []
    for title in logical_titles:
        member_heights = [raw_ink[item] for item in title.member_ids if item in raw_ink]
        ink_height = float(median(member_heights)) if member_heights else float(title.box.height)
        height_ratio = ink_height / body_height
        base = baseline_by_members.get(tuple(title.member_ids))
        score_model = _clamp01(title.confidence)
        # 与本文文字相同大小只提供弱证据；达到约 1.8 倍时封顶。
        score_ink = _clamp01((height_ratio - 0.9) / 0.9)
        score_left = _clamp01(1.0 - (title.box.x1 - left_min) / left_span)
        score_sequence = 1.0 if title.id in run_starts else 0.0
        score_baseline = 1.0 if base is not None and base.level == 2 else 0.0
        h2_score = (
            0.25 * score_model
            + 0.20 * score_ink
            + 0.15 * score_left
            + 0.20 * score_sequence
            + 0.20 * score_baseline
        )
        provisional.append(
            {
                "title": title,
                "base": base,
                "ink_height": ink_height,
                "height_ratio": height_ratio,
                "score_model": score_model,
                "score_ink": score_ink,
                "score_left": score_left,
                "score_sequence": score_sequence,
                "score_baseline": score_baseline,
                "h2_score": h2_score,
            }
        )

    # H1/目录标题沿用已经验证过的居中规则。语义切块只改变正文 H2/H3。
    fixed = [item for item in baseline if item.level == 1]
    body_h1 = next((item for item in fixed if item.role == "body_h1"), None)
    body_start = body_h1.box.y2 if body_h1 is not None else 0
    body_items = [item for item in provisional if item["title"].box.y1 >= body_start]
    selected_h2 = {
        item["title"].id
        for item in body_items
        if item["h2_score"] >= config.semantic_h2_min_score
        and not is_centered(
            item["title"].box, image_width, config.center_tolerance_ratio
        )
    }
    # 完全没有达到阈值时，选择最强的非居中候选作为唯一保守边界；如果连
    # 候选也没有，调用方会自动退回 legacy 自适应安全切割。
    if not selected_h2:
        non_centered = [
            item
            for item in body_items
            if not is_centered(
                item["title"].box, image_width, config.center_tolerance_ratio
            )
        ]
        if non_centered:
            selected_h2.add(max(non_centered, key=lambda item: item["h2_score"])["title"].id)

    headings = list(fixed)
    evidence: list[HeadingEvidence] = []
    current_h2: str | None = None
    h2_index = 0
    h3_index = 0
    for item in body_items:
        title = item["title"]
        base = item["base"]
        centered = is_centered(
            title.box, image_width, config.center_tolerance_ratio
        )
        if title.id in selected_h2:
            level = 2
            role = "semantic_h2"
            heading_id = f"semantic_h2_{h2_index:04d}"
            parent_id = body_h1.id if body_h1 is not None else None
            current_h2 = heading_id
            h2_index += 1
            final_confidence = item["h2_score"]
        else:
            level = 3
            role = "semantic_h3_candidate"
            heading_id = f"semantic_h3_{h3_index:04d}"
            parent_id = current_h2
            h3_index += 1
            final_confidence = 1.0 - item["h2_score"]
        headings.append(
            Heading(
                id=heading_id,
                level=level,
                role=role,
                box=title.box,
                parent_id=parent_id,
                confidence=float(final_confidence),
                centered=centered,
                member_ids=title.member_ids,
            )
        )
        evidence.append(
            HeadingEvidence(
                title_id=heading_id,
                member_ids=title.member_ids,
                y1=title.box.y1,
                y2=title.box.y2,
                model_confidence=title.confidence,
                ink_line_height=item["ink_height"],
                body_line_height=body_height,
                text_height_ratio=item["height_ratio"],
                left_prominence=item["score_left"],
                centered=centered,
                consecutive_run_start=item["score_sequence"] > 0,
                baseline_level=base.level if base is not None else None,
                baseline_role=base.role if base is not None else None,
                numbering_hint=None,
                score_model=item["score_model"],
                score_ink=item["score_ink"],
                score_left=item["score_left"],
                score_sequence=item["score_sequence"],
                score_baseline=item["score_baseline"],
                h2_score=item["h2_score"],
                final_level=level,
                final_confidence=float(final_confidence),
                used_as_h2_boundary=level == 2,
            )
        )

    headings.sort(key=lambda item: (item.box.y1, item.level, item.box.x1))
    debug = {
        "method": "general6+relative-ink+legacy-sequence",
        "body_ink_line_height": body_height,
        "h2_min_score": config.semantic_h2_min_score,
        "logical_title_count": len(logical_titles),
        "h2_count": sum(item.level == 2 for item in headings),
        "h3_candidate_count": sum(item.level == 3 for item in headings),
        "numbering_note": (
            "准备阶段默认不加载 OCR；标题编号由 FinixDoc-VL 在请求块内判断，"
            "本字段为以后标题专用 OCR 预留。"
        ),
        "evidence": [item.to_dict() for item in evidence],
    }
    return headings, evidence, debug


def save_heading_audit_windows(
    window_paths: list[Path],
    windows: list[DetectionWindow],
    headings: list[Heading],
    evidence: list[HeadingEvidence],
    output_dir: Path,
) -> None:
    """把最终层级和分数画回检测窗口，保留可直接肉眼检查的中间产物。"""

    score_by_id = {item.title_id: item.h2_score for item in evidence}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, window in zip(window_paths, windows):
        visible = [
            item
            for item in headings
            if item.box.y2 > window.start_y and item.box.y1 < window.end_y
        ]
        if not visible:
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for heading in visible:
            color = {1: "#8a2be2", 2: "#e60000", 3: "#0066ff"}.get(
                heading.level, "#00aa55"
            )
            box = (
                heading.box.x1,
                max(0, heading.box.y1 - window.start_y),
                heading.box.x2,
                min(image.height, heading.box.y2 - window.start_y),
            )
            draw.rectangle(box, outline=color, width=4)
            score = score_by_id.get(heading.id)
            label = f"H{heading.level} {heading.id}"
            if score is not None:
                label += f" h2={score:.2f}"
            draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color)
        image.save(output_dir / path.name, format="PNG", compress_level=4)
