"""本地视觉模型的极端长图临时切割。

正式预处理仍保留完整请求块；这里只在本地模型即将推理前，把压缩后会过窄
的长图沿横向空白带临时拆开。每个子块都会复制同一标题上下文，最终再去掉
重复标题并恢复成一个 Markdown 结果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SPLIT_VERSION = "local-long-split-v2-columns-headings"


@dataclass(frozen=True)
class LocalLongPart:
    index: int
    file_name: str
    content_start_x: int
    content_end_x: int
    content_start_y: int
    content_end_y: int
    overlap_top: int
    overlap_bottom: int
    cut_method: str
    header_height: int
    output_height: int
    column_index: int = 0
    column_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimated_model_width(width: int, height: int, max_pixels: int) -> float:
    """估算模型按像素上限等比例缩放后的宽度。"""

    pixels = width * height
    if pixels <= max_pixels:
        return float(width)
    return width * math.sqrt(max_pixels / pixels)


def needs_local_long_split(
    width: int,
    height: int,
    max_pixels: int,
    *,
    trigger_height: int,
    minimum_estimated_width: int,
) -> bool:
    """只处理又高、又会在模型内部被压成窄条的图片。"""

    return (
        height > trigger_height
        and estimated_model_width(width, height, max_pixels)
        < minimum_estimated_width
    )


def leading_header_height(
    pack: Any,
    image_manifest: dict[str, Any],
    context_gap: int,
    request_height: int,
) -> tuple[int, tuple[str, ...]]:
    """从请求包和语义标题框精确还原复合图片顶部的标题上下文高度。"""

    context_boxes = tuple(getattr(pack, "context_boxes", ()))
    context_ids = tuple(getattr(pack, "context_heading_ids", ()))
    if context_boxes:
        height = sum(box.height for box in context_boxes)
        # save_recognition_pack_images 会在每条上下文后（包括正文前）放一个间隔。
        height += context_gap * len(context_boxes)
        return min(request_height - 1, max(0, height)), context_ids

    visible_ids = tuple(getattr(pack, "visible_heading_ids", ()))
    source_box = getattr(pack, "source_box", None)
    body_scale = float(getattr(pack, "body_scale", 1.0))
    if source_box is None or not visible_ids:
        return 0, ()

    heading_by_id = {
        str(item.get("id")): item
        for item in image_manifest.get("semantic_headings", [])
        if isinstance(item, dict)
    }
    padding = int(image_manifest.get("config", {}).get("semantic_title_padding", 0))
    # 只复制真正位于请求块开头的标题。正文中途遇到的 H3 不能被误当成公共头。
    tolerance = max(20, padding * 3)
    candidates: list[tuple[int, str]] = []
    for heading_id in visible_ids:
        raw = heading_by_id.get(heading_id)
        if raw is None or not isinstance(raw.get("box"), dict):
            continue
        box = raw["box"]
        start_y = int(box["y1"])
        if abs(start_y - source_box.y1) > tolerance:
            continue
        end_y = min(source_box.y2, int(box["y2"]) + padding)
        scaled_height = round((end_y - source_box.y1) * body_scale)
        if scaled_height > 0:
            candidates.append((scaled_height, heading_id))

    if not candidates:
        return 0, ()
    height = max(item[0] for item in candidates)
    ids = tuple(item[1] for item in sorted(candidates))
    return min(request_height - 1, height), ids


def _row_ink_projection(
    image: Image.Image,
    *,
    sample_width: int,
    white_threshold: int,
) -> np.ndarray:
    gray = image.convert("L")
    width = min(sample_width, gray.width)
    if width != gray.width:
        gray = gray.resize((width, gray.height), Image.Resampling.BOX)
    values = np.asarray(gray, dtype=np.uint8)
    return np.mean(values < white_threshold, axis=1)


def _choose_boundary(
    projection: np.ndarray,
    target: int,
    lower: int,
    upper: int,
    *,
    blank_ratio: float,
    minimum_blank_height: int,
) -> tuple[int, str]:
    """在目标附近优先取完整空白带；找不到时取墨迹最少的一行。"""

    lower = max(1, lower)
    upper = min(len(projection) - 1, upper)
    if lower >= upper:
        return lower, "minimum_ink"

    candidates: list[tuple[int, float, int]] = []
    cursor = lower
    while cursor <= upper:
        if projection[cursor] > blank_ratio:
            cursor += 1
            continue
        start = cursor
        while cursor <= upper and projection[cursor] <= blank_ratio:
            cursor += 1
        end = cursor
        if end - start >= minimum_blank_height:
            middle = (start + end) // 2
            candidates.append((abs(middle - target), float(projection[start:end].mean()), middle))

    if candidates:
        _, _, boundary = min(candidates)
        return boundary, "blank_band"

    window = projection[lower : upper + 1]
    minimum = float(window.min())
    rows = np.flatnonzero(window == minimum) + lower
    boundary = int(min(rows, key=lambda value: abs(int(value) - target)))
    method = "minimum_ink" if minimum <= blank_ratio * 5 else "fallback_overlap"
    return boundary, method


def _detect_column_ranges(
    image: Image.Image,
    header_height: int,
    *,
    white_threshold: int,
    blank_ratio: float,
) -> list[tuple[int, int]]:
    """在正文中央查找贯穿大部分高度的竖向空白槽。"""

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    body = gray[header_height:] if header_height < gray.shape[0] else gray
    projection = np.mean(body < white_threshold, axis=0)
    width = image.width
    search_start = round(width * 0.28)
    search_end = round(width * 0.72)
    minimum_width = max(12, round(width * 0.02))
    candidates: list[tuple[int, float, int, int]] = []
    cursor = search_start
    while cursor < search_end:
        if projection[cursor] > blank_ratio:
            cursor += 1
            continue
        band_start = cursor
        while cursor < search_end and projection[cursor] <= blank_ratio:
            cursor += 1
        band_end = cursor
        if band_end - band_start >= minimum_width:
            middle = (band_start + band_end) // 2
            candidates.append(
                (
                    abs(middle - width // 2),
                    float(projection[band_start:band_end].mean()),
                    band_start,
                    band_end,
                )
            )

    if not candidates:
        return [(0, width)]
    _, _, band_start, band_end = min(candidates)
    boundary = (band_start + band_end) // 2
    return [(0, boundary), (boundary, width)]


def _fit_header_to_width(header: Image.Image, width: int) -> Image.Image:
    """保留标题真实字形，去掉两侧大空白后居中放进列宽。"""

    gray = np.asarray(header.convert("L"), dtype=np.uint8)
    active = np.argwhere(gray < 245)
    if active.size:
        x1 = max(0, int(active[:, 1].min()) - 12)
        x2 = min(header.width, int(active[:, 1].max()) + 13)
        visible = header.crop((x1, 0, x2, header.height))
    else:
        visible = header.copy()
    if visible.width > width:
        scale = width / visible.width
        visible = visible.resize(
            (width, max(1, round(visible.height * scale))),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGB", (width, header.height), "white")
    canvas.paste(
        visible,
        ((width - visible.width) // 2, max(0, (header.height - visible.height) // 2)),
    )
    return canvas


def _plan_vertical_ranges(
    projection: np.ndarray,
    body_start: int,
    height: int,
    *,
    target_height: int,
    maximum_height: int,
    header_height: int,
    minimum_content_height: int,
    search_radius: int,
    fallback_overlap: int,
    blank_ratio: float,
    minimum_blank_height: int,
) -> list[tuple[int, int, int, int, str]]:
    gap = 10 if header_height else 0
    body_height = height - body_start
    capacity = maximum_height - header_height - gap
    if capacity <= minimum_content_height:
        raise RuntimeError(
            f"标题上下文高 {header_height}px，留给正文的本地切块高度不足"
        )

    safe_capacity = max(minimum_content_height, capacity - fallback_overlap)
    target_capacity = min(
        safe_capacity,
        max(minimum_content_height, target_height - header_height - gap),
    )
    part_count = max(1, math.ceil(body_height / target_capacity))
    if part_count == 1:
        return []

    ideal_content = body_height / part_count
    effective_minimum = min(
        minimum_content_height,
        max(128, int(ideal_content * 0.45)),
    )
    boundaries: list[tuple[int, str]] = []
    previous = body_start
    for index in range(1, part_count):
        remaining = part_count - index
        target = body_start + round(body_height * index / part_count)
        lower = max(previous + effective_minimum, target - search_radius)
        upper = min(
            previous + safe_capacity,
            height - remaining * effective_minimum,
            target + search_radius,
        )
        boundary, method = _choose_boundary(
            projection,
            target,
            lower,
            upper,
            blank_ratio=blank_ratio,
            minimum_blank_height=minimum_blank_height,
        )
        boundaries.append((boundary, method))
        previous = boundary

    ranges: list[tuple[int, int, int, int, str]] = []
    previous_boundary = body_start
    for index in range(part_count):
        next_boundary = boundaries[index][0] if index < len(boundaries) else height
        next_method = boundaries[index][1] if index < len(boundaries) else "document_end"
        previous_method = boundaries[index - 1][1] if index > 0 else "document_start"
        top_overlap = fallback_overlap // 2 if previous_method == "fallback_overlap" else 0
        bottom_overlap = (
            fallback_overlap - fallback_overlap // 2
            if next_method == "fallback_overlap"
            else 0
        )
        content_start = max(body_start, previous_boundary - top_overlap)
        content_end = min(height, next_boundary + bottom_overlap)
        ranges.append(
            (
                content_start,
                content_end,
                max(0, previous_boundary - content_start),
                max(0, content_end - next_boundary),
                next_method,
            )
        )
        previous_boundary = next_boundary
    return ranges


def create_local_long_parts(
    image_path: Path,
    output_root: Path,
    *,
    header_height: int,
    target_height: int,
    maximum_height: int,
    minimum_content_height: int,
    search_radius: int,
    fallback_overlap: int,
    sample_width: int,
    white_threshold: int,
    blank_ratio: float,
    minimum_blank_height: int,
    split_columns: bool = False,
) -> list[LocalLongPart]:
    """生成按阅读顺序排列、且带重复标题头的本地临时子块。"""

    output_root.mkdir(parents=True, exist_ok=True)
    parts_dir = output_root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source_image:
        source = source_image.convert("RGB")
        width, height = source.size
        header_height = min(max(0, header_height), height - 1)
        body_start = header_height
        gap = 10 if header_height else 0
        full_header = (
            source.crop((0, 0, width, header_height))
            if header_height
            else None
        )
        column_ranges = (
            _detect_column_ranges(
                source,
                header_height,
                white_threshold=white_threshold,
                blank_ratio=blank_ratio,
            )
            if split_columns
            else [(0, width)]
        )

        parts: list[LocalLongPart] = []
        for column_index, (x1, x2) in enumerate(column_ranges):
            column = source.crop((x1, 0, x2, height))
            projection = _row_ink_projection(
                column,
                sample_width=sample_width,
                white_threshold=white_threshold,
            )
            ranges = _plan_vertical_ranges(
                projection,
                body_start,
                height,
                target_height=target_height,
                maximum_height=maximum_height,
                header_height=header_height,
                minimum_content_height=minimum_content_height,
                search_radius=search_radius,
                fallback_overlap=fallback_overlap,
                blank_ratio=blank_ratio,
                minimum_blank_height=minimum_blank_height,
            )
            if not ranges:
                return []

            fitted_header = None
            if full_header is not None:
                fitted_header = (
                    full_header.copy()
                    if len(column_ranges) == 1 and x1 == 0 and x2 == width
                    else _fit_header_to_width(full_header, x2 - x1)
                )
            for content_start, content_end, overlap_top, overlap_bottom, method in ranges:
                content = source.crop((x1, content_start, x2, content_end))
                output_height = content.height + (
                    header_height + gap if fitted_header is not None else 0
                )
                if output_height > maximum_height:
                    raise RuntimeError(
                        f"本地临时子块 {len(parts)} 高 {output_height}px，"
                        f"超过 {maximum_height}px"
                    )
                canvas = Image.new("RGB", (x2 - x1, output_height), "white")
                cursor = 0
                if fitted_header is not None:
                    canvas.paste(fitted_header, (0, 0))
                    cursor = header_height + gap
                    draw = ImageDraw.Draw(canvas)
                    line_y = header_height + gap // 2
                    draw.line(
                        (0, line_y, canvas.width, line_y),
                        fill=(190, 190, 190),
                        width=1,
                    )
                canvas.paste(content, (0, cursor))
                part_index = len(parts)
                file_name = (
                    f"part_{part_index:03d}_c{column_index:02d}_"
                    f"x{x1:05d}_{x2:05d}_y{content_start:07d}_{content_end:07d}.png"
                )
                canvas.save(parts_dir / file_name, format="PNG", compress_level=4)
                parts.append(
                    LocalLongPart(
                        index=part_index,
                        file_name=file_name,
                        content_start_x=x1,
                        content_end_x=x2,
                        content_start_y=content_start,
                        content_end_y=content_end,
                        overlap_top=overlap_top,
                        overlap_bottom=overlap_bottom,
                        cut_method=method,
                        header_height=header_height,
                        output_height=output_height,
                        column_index=column_index,
                        column_count=len(column_ranges),
                    )
                )

    return parts


def _numbered_level(text: str) -> int | None:
    value = text.strip()
    arabic = re.match(r"^(\d+(?:\.\d+){0,4})(?:\s|[、．]|$)", value)
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
    if re.match(r"^[（(][一二三四五六七八九十百0-9]+[）)]", value):
        return 4
    return None


def restore_local_markdown_headings(markdown: str, pack: Any) -> str:
    """利用小模型已检测层级和编号，把本地 OCR 漏掉的标题井号补回来。"""

    lines = markdown.splitlines()
    role = str(getattr(pack, "semantic_role", ""))
    if role.startswith("table_of_contents"):
        for index, line in enumerate(lines):
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            text = match.group(1) if match else line.strip()
            if text and "目录" in text and len(text) <= 40:
                lines[index] = f"# {text}"
                break
        return "\n".join(lines).strip()

    hints: list[tuple[str, int]] = []
    seen_ids: set[str] = set()
    for raw in getattr(pack, "heading_hints", ()):
        heading_id = str(raw.get("heading_id", ""))
        level = int(raw.get("level", 0))
        if heading_id and 1 <= level <= 5 and heading_id not in seen_ids:
            seen_ids.add(heading_id)
            hints.append((heading_id, level))
    expected = {level: sum(item[1] == level for item in hints) for level in range(1, 6)}
    recognized = {level: 0 for level in range(1, 6)}
    hint_cursor = 0

    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        text = match.group(2)
        hint_level = None
        if hint_cursor < len(hints):
            hint_level = hints[hint_cursor][1]
            hint_cursor += 1
        level = _numbered_level(text)
        if level is None:
            level = hint_level or len(match.group(1))
        recognized[level] = recognized.get(level, 0) + 1
        lines[index] = f"{'#' * level} {text}"

    # 文档主标题通常没有编号。若原图明确检测到 H1，而模型连井号都漏了，
    # 只允许请求开头第一个短行补成 H1。
    if expected.get(1, 0) > recognized.get(1, 0):
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(stripped) <= 80 and stripped[-1:] not in "。；;，,":
                lines[index] = f"# {stripped}"
                recognized[1] += 1
            break

    # 只提升“原图标题预算仍有缺口”的强编号短行；普通（1）枚举不会无限升级。
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or len(stripped) > 80:
            continue
        if stripped[-1:] in "。；;，,":
            continue
        level = _numbered_level(stripped)
        if level is None or recognized.get(level, 0) >= expected.get(level, 0):
            continue
        lines[index] = f"{'#' * level} {stripped}"
        recognized[level] = recognized.get(level, 0) + 1
    return "\n".join(lines).strip()


def _normalized_heading(text: str) -> str:
    return re.sub(r"[\s#：:，,。．.、（）()\-—_]", "", text).lower()


def _leading_heading_signatures(markdown: str, maximum: int) -> list[str]:
    signatures: list[str] = []
    for line in markdown.splitlines()[:12]:
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if match:
            signatures.append(_normalized_heading(match.group(1)))
            if len(signatures) >= maximum:
                break
        elif line.strip() and signatures:
            break
    return signatures


def _strip_repeated_leading_headings(markdown: str, signatures: list[str]) -> str:
    if not signatures:
        return markdown.strip()
    lines = markdown.strip().splitlines()
    signature_index = 0
    output: list[str] = []
    scanning_header = True
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
        candidate = match.group(1) if match else stripped
        if (
            scanning_header
            and candidate
            and signature_index < len(signatures)
            and SequenceMatcher(
                None,
                _normalized_heading(candidate),
                signatures[signature_index],
            ).ratio()
            >= 0.62
        ):
            signature_index += 1
            continue
        if stripped:
            scanning_header = False
        output.append(line)
    return "\n".join(output).strip()


def _merge_overlap(left: str, right: str, max_chars: int = 1600) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    for length in range(min(len(left), len(right), max_chars), 19, -1):
        if left[-length:] == right[:length]:
            return left + right[length:]

    left_lines = left.splitlines()
    right_lines = right.splitlines()
    normalize = lambda value: re.sub(r"\s+", "", value)
    for count in range(min(len(left_lines), len(right_lines), 20), 0, -1):
        if [normalize(line) for line in left_lines[-count:]] == [
            normalize(line) for line in right_lines[:count]
        ]:
            return "\n".join(left_lines + right_lines[count:]).strip()
    return left + "\n" + right


def merge_local_part_markdowns(
    markdowns: list[str],
    parts: list[LocalLongPart],
    *,
    repeated_heading_count: int,
) -> str:
    """保留第一份标题头，删除续块重复头，并处理物理重叠接缝。"""

    if len(markdowns) != len(parts):
        raise ValueError("本地子块数量与 Markdown 数量不一致")
    if not markdowns:
        return ""

    signatures = _leading_heading_signatures(
        markdowns[0],
        max(1, repeated_heading_count),
    )
    current = markdowns[0].strip()
    for markdown, part in zip(markdowns[1:], parts[1:]):
        cleaned = _strip_repeated_leading_headings(markdown, signatures)
        if not current:
            current = cleaned
        elif part.overlap_top > 0:
            current = _merge_overlap(current, cleaned)
        elif cleaned:
            current = current.rstrip() + "\n\n" + cleaned.lstrip()
    return current.strip()
