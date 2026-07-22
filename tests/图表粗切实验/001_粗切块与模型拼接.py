"""图表粗切实验：横向分表、像素粗切、API识别、模型矩阵对齐。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from html import escape
import json
import math
from pathlib import Path
import re
import sys
from threading import Lock
from typing import Any

from PIL import Image, ImageDraw

项目根目录 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(项目根目录))

from afac_pipeline.common.cache import ResultCache  # noqa: E402
from afac_pipeline.common.hashing import discover_images  # noqa: E402
from afac_pipeline.common.models import Box  # noqa: E402
from afac_pipeline.common.vlm_client import FinixDocClient  # noqa: E402
from afac_pipeline.table import TableConfig, TablePipeline  # noqa: E402
from afac_pipeline.table.步骤009_HTML表格软对齐 import parse_table_response_checked  # noqa: E402


默认样本 = ("d8b59365", "5b93ec6f", "1829aea8", "185a2337", "58b3cb9e")


def 写JSON(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def 找图片(input_dir: Path, prefixes: tuple[str, ...]) -> list[Path]:
    images = discover_images(input_dir)
    result = []
    for prefix in prefixes:
        matches = [path for path in images if path.name.startswith(prefix)]
        if not matches:
            raise FileNotFoundError(f"找不到图片前缀：{prefix}")
        result.append(matches[0])
    return result


def 责任段(length: int, target: int) -> list[tuple[int, int]]:
    count = max(1, math.ceil(length / target))
    points = [round(index * length / count) for index in range(count + 1)]
    return list(zip(points, points[1:]))


def 生成切块(
    image_path: Path,
    work_dir: Path,
    config: TableConfig,
    target: int,
    overlap: int,
) -> dict[str, Any]:
    sample_dir = work_dir / image_path.stem
    split_dir = sample_dir / "00_横向分表中间产物"
    tile_dir = sample_dir / "02_API实际输入切块"
    tile_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TablePipeline(config, sample_dir / "内部临时")
    meta = pipeline.backend.read_meta(image_path)
    preview = pipeline._make_detection_preview(image_path, meta, split_dir)
    pipeline._detect_regions(preview, meta)
    pipeline._save_v6_detection_debug(preview, split_dir)
    result = pipeline._last_v6_regions
    if result is None or not result.split_boxes:
        raise RuntimeError(f"{image_path.name} 没有得到横向分表框")
    tables = [
        pipeline._map_preview_box(box, preview.width, preview.height, meta)
        for box in result.split_boxes
        if box.width > 0 and box.height > 0
    ]

    tiles = []
    for table_index, table in enumerate(tables):
        rows = 责任段(table.height, target)
        columns = 责任段(table.width, target)
        for row_index, (ly1, ly2) in enumerate(rows):
            for column_index, (lx1, lx2) in enumerate(columns):
                owner = Box(
                    table.x1 + lx1, table.y1 + ly1,
                    table.x1 + lx2, table.y1 + ly2,
                )
                crop = Box(
                    max(table.x1, owner.x1 - overlap),
                    max(table.y1, owner.y1 - overlap),
                    min(table.x2, owner.x2 + overlap),
                    min(table.y2, owner.y2 + overlap),
                )
                name = (
                    f"第{table_index+1:03d}表_第{row_index+1:03d}行带_"
                    f"第{column_index+1:03d}列块.png"
                )
                output = tile_dir / name
                pipeline.backend.save_crop(image_path, crop, output)
                tiles.append(
                    {
                        "图片名": image_path.name,
                        "表序号": table_index,
                        "行带序号": row_index,
                        "列块序号": column_index,
                        "行带总数": len(rows),
                        "列块总数": len(columns),
                        "责任框": owner.to_dict(),
                        "实际裁切框": crop.to_dict(),
                        "文件路径": str(output.resolve()),
                    }
                )

    overlay = preview.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    sx, sy = preview.width / meta.width, preview.height / meta.height
    for index, table in enumerate(tables):
        coords = tuple(
            round(value)
            for value in (table.x1*sx, table.y1*sy, table.x2*sx, table.y2*sy)
        )
        draw.rectangle(coords, outline=(0, 80, 255, 255), width=5)
        draw.text((coords[0]+4, coords[1]+4), f"表{index+1}", fill=(0, 80, 255))
    for tile in tiles:
        owner = Box.from_dict(tile["责任框"])
        crop = Box.from_dict(tile["实际裁切框"])
        crop_coords = tuple(
            round(value)
            for value in (crop.x1*sx, crop.y1*sy, crop.x2*sx, crop.y2*sy)
        )
        owner_coords = tuple(
            round(value)
            for value in (owner.x1*sx, owner.y1*sy, owner.x2*sx, owner.y2*sy)
        )
        draw.rectangle(crop_coords, outline=(255, 0, 0, 220), width=2)
        draw.rectangle(owner_coords, outline=(0, 170, 0, 255), width=3)
        draw.text(
            (owner_coords[0]+3, owner_coords[1]+3),
            f"{tile['表序号']+1}-{tile['行带序号']+1}-{tile['列块序号']+1}",
            fill=(0, 120, 0),
        )
    overlay.save(sample_dir / "01_粗切块位置总览.png")
    manifest = {
        "图片": str(image_path.resolve()),
        "原图尺寸": [meta.width, meta.height],
        "分表数量": len(tables),
        "分表框": [box.to_dict() for box in tables],
        "目标责任区边长": target,
        "重叠像素": overlap,
        "API切块数量": len(tiles),
        "切块": tiles,
    }
    写JSON(sample_dir / "粗切块清单.json", manifest)
    return manifest


def 解析空格分隔纯文本(response: str) -> dict[str, Any]:
    """把模型偶尔返回的稳定空格列恢复成矩阵，不猜测具体业务字段。"""

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    token_rows = [re.split(r"\s+", line) for line in lines]
    candidates = [
        index for index, tokens in enumerate(token_rows)
        if len(tokens) >= 2
    ]
    if not candidates:
        return {"单元格": [[line] for line in lines], "前置文字": "", "后置文字": "", "解析方式": "单列纯文字"}

    # 选择最长的连续多列区段，避免把标题中的偶然空格误当成表格。
    runs: list[list[int]] = []
    current: list[int] = []
    for index in candidates:
        if current and index != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(index)
    if current:
        runs.append(current)
    selected = max(runs, key=len)
    if len(selected) < 1:
        raise ValueError("纯文字中没有稳定的多列数据")

    counts: dict[int, int] = {}
    for index in selected:
        width = len(token_rows[index])
        counts[width] = counts.get(width, 0) + 1
    target_width = max(counts, key=lambda width: (counts[width], width))
    rows = []
    accepted_indices = []
    for index in selected:
        tokens = token_rows[index]
        if len(tokens) < max(2, target_width - 1):
            continue
        if len(tokens) > target_width:
            # 多出的词通常来自带空格的行名，合并进第一格。
            extra = len(tokens) - target_width + 1
            tokens = [" ".join(tokens[:extra]), *tokens[extra:]]
        tokens = [*tokens, *([""] * (target_width - len(tokens)))]
        rows.append(tokens)
        accepted_indices.append(index)
    if len(rows) < 1:
        raise ValueError("纯文字多列区段规整后为空")

    first = accepted_indices[0]
    last = accepted_indices[-1]
    return {
        "单元格": rows,
        "前置文字": "\n".join(lines[:first]),
        "后置文字": "\n".join(lines[last + 1:]),
        "解析方式": "稳定空格列纯文字",
    }


def 解析矩阵(response: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", response).strip().lower()
    if (
        "markdown parsing task" in normalized
        and len(normalized) < 300
    ):
        return {
            "单元格": [],
            "前置文字": "",
            "后置文字": "",
            "解析方式": "空白块导致后台提示词回显",
        }
    try:
        parsed = parse_table_response_checked(response)
    except Exception as table_error:
        if "<" in normalized:
            raise table_error
        try:
            return 解析空格分隔纯文本(response)
        except Exception:
            raise table_error
    cells = [["" for _ in range(parsed.column_count)] for _ in range(parsed.row_count)]
    for placement in parsed.placements:
        cells[placement.row][placement.column] = placement.cell.text.strip()
    return {
        "单元格": cells,
        "前置文字": parsed.prefix,
        "后置文字": parsed.suffix,
        "解析方式": "HTML或Markdown表格",
    }


def 尺寸(matrix: dict[str, Any]) -> tuple[int, int]:
    rows = matrix["单元格"]
    return len(rows), max((len(row) for row in rows), default=0)


def 规整(matrix: dict[str, Any]) -> dict[str, Any]:
    rows, columns = 尺寸(matrix)
    return {
        **matrix,
        "单元格": [
            [*row, *([""] * (columns-len(row)))]
            for row in matrix["单元格"]
        ],
    }


def 标准文字(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def 行相似(left: list[str], right: list[str]) -> float:
    """只比较物理接缝附近：左块尾部与右块头部。"""

    a = [标准文字(value) for value in left if 标准文字(value)][-16:]
    b = [标准文字(value) for value in right if 标准文字(value)][:16]
    if not a or not b:
        return 0.0
    exact = len(set(a) & set(b)) / max(1, min(len(set(a)), len(set(b))))
    if exact:
        # 一个精确共同值已是很强的行锚点；其余OCR误差不能把整行拉低。
        return min(1.0, 0.52 + 0.48 * exact)
    forward = [
        max(SequenceMatcher(None, value, other).ratio() for other in b)
        for value in a
    ]
    backward = [
        max(SequenceMatcher(None, value, other).ratio() for other in a)
        for value in b
    ]
    return (
        (sum(forward) / len(forward) + sum(backward) / len(backward))
        / 2
        * 0.62
    )


def 对齐(left: list[list[str]], right: list[list[str]]) -> tuple[list[tuple[int|None, int|None]], dict[str, Any]]:
    if not left:
        return [(None, j) for j in range(len(right))], {"方式": "左侧为空"}
    if not right:
        return [(i, None) for i in range(len(left))], {"方式": "右侧为空"}
    similarities = [[行相似(a, b) for b in right] for a in left]
    maximum = max(max(row) for row in similarities)
    if maximum < 0.22:
        length = max(len(left), len(right))
        return [
            (i if i < len(left) else None, i if i < len(right) else None)
            for i in range(length)
        ], {"方式": "无可靠内容锚点，按位置对齐", "最大相似度": maximum}

    m, n, gap = len(left), len(right), -0.55
    score = [[0.0]*(n+1) for _ in range(m+1)]
    trace = [[""]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        score[i][0], trace[i][0] = i*gap, "上"
    for j in range(1, n+1):
        score[0][j], trace[0][j] = j*gap, "左"
    for i in range(1, m+1):
        for j in range(1, n+1):
            similarity = similarities[i-1][j-1]
            match = -2.0 if similarity < 0.16 else 3*similarity-0.65
            score[i][j], trace[i][j] = max(
                (score[i-1][j-1]+match, "斜"),
                (score[i-1][j]+gap, "上"),
                (score[i][j-1]+gap, "左"),
                key=lambda item: item[0],
            )
    alignment = []
    i, j = m, n
    while i or j:
        direction = trace[i][j]
        if direction == "斜":
            alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif direction == "上":
            alignment.append((i-1, None))
            i -= 1
        else:
            alignment.append((None, j-1))
            j -= 1
    alignment.reverse()
    return alignment, {
        "方式": "内容动态序列对齐",
        "最大相似度": maximum,
        "得分": score[m][n],
    }


def 重叠列数(
    left: list[list[str]], right: list[list[str]],
    alignment: list[tuple[int|None, int|None]],
) -> tuple[int, dict[str, Any]]:
    lw = max((len(row) for row in left), default=0)
    rw = max((len(row) for row in right), default=0)
    best = (0.0, 0, 0, 0)
    for count in range(1, min(10, lw, rw)+1):
        matches = comparisons = 0
        for li, ri in alignment:
            if li is None or ri is None:
                continue
            for a, b in zip(left[li][-count:], right[ri][:count]):
                a, b = 标准文字(a), 标准文字(b)
                if not a or not b:
                    continue
                comparisons += 1
                if SequenceMatcher(None, a, b).ratio() >= 0.82:
                    matches += 1
        ratio = matches/comparisons if comparisons else 0.0
        if matches >= 2 and ratio >= 0.55 and (ratio, matches) > best[:2]:
            best = (ratio, matches, count, comparisons)
    return best[2], {
        "重复列数": best[2], "匹配比例": best[0],
        "匹配格数": best[1], "比较格数": best[3],
    }



def 列指纹相似度(
    left: list[list[str]],
    right: list[list[str]],
    row_alignment: list[tuple[int | None, int | None]],
    left_column: int,
    right_column: int,
) -> float:
    """按已经对齐的行比较两列；少量OCR误字不会让整列失配。"""

    matched_rows = [
        (left_row, right_row)
        for left_row, right_row in row_alignment
        if left_row is not None and right_row is not None
    ]
    if len(matched_rows) > 12:
        # 均匀抽样首尾和中间行；表格列通常纵向稳定，无需比较全部行。
        positions = {
            round(index * (len(matched_rows) - 1) / 11)
            for index in range(12)
        }
        matched_rows = [
            matched_rows[index] for index in sorted(positions)
        ]
    ratios = []
    for left_row, right_row in matched_rows:
        a = 标准文字(left[left_row][left_column])
        b = 标准文字(right[right_row][right_column])
        if not a or not b:
            continue
        ratios.append(SequenceMatcher(None, a, b).ratio())
    if len(ratios) < 2:
        return 0.0
    # 中位数抵抗个别严重误识别；均值保留整体差异信息。
    ordered = sorted(ratios)
    median = ordered[len(ordered) // 2]
    average = sum(ratios) / len(ratios)
    return median * 0.65 + average * 0.35


def 列序列局部对齐(
    left: list[list[str]],
    right: list[list[str]],
    row_alignment: list[tuple[int | None, int | None]],
) -> tuple[list[tuple[int | None, int | None]] | None, dict[str, Any]]:
    """在左矩阵尾部和右矩阵头部做局部动态对齐。

    返回的是重叠区列映射。右侧匹配前的少量列视为重叠区OCR插列，不作为
    新列重复追加；左侧匹配后的尾列则保留，避免模型漏掉右侧重叠内容。
    """

    left_width = max((len(row) for row in left), default=0)
    right_width = max((len(row) for row in right), default=0)
    if not left_width or not right_width:
        return None, {"方式": "一侧矩阵为空"}

    # 256像素上下文在现有样本中远小于40个模型列；只查相邻端部，
    # 避免累计矩阵变宽后出现平方级无效比较。
    left_start_limit = max(0, left_width - 40)
    right_end_limit = min(40, right_width)
    left_columns = list(range(left_start_limit, left_width))
    right_columns = list(range(right_end_limit))
    m, n = len(left_columns), len(right_columns)
    similarities = [
        [
            列指纹相似度(
                left,
                right,
                row_alignment,
                left_column,
                right_column,
            )
            for right_column in right_columns
        ]
        for left_column in left_columns
    ]

    gap = -0.70
    scores = [[0.0] * (n + 1) for _ in range(m + 1)]
    traces = [[""] * (n + 1) for _ in range(m + 1)]
    best_score = 0.0
    best_position = (0, 0)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            similarity = similarities[i - 1][j - 1]
            if similarity >= 0.88:
                match = 2.7
            elif similarity >= 0.72:
                match = 1.8
            elif similarity >= 0.55:
                match = 0.7
            else:
                match = -1.6
            value, direction = max(
                (0.0, ""),
                (scores[i - 1][j - 1] + match, "斜"),
                (scores[i - 1][j] + gap, "上"),
                (scores[i][j - 1] + gap, "左"),
                key=lambda item: item[0],
            )
            scores[i][j] = value
            traces[i][j] = direction
            if value > best_score:
                best_score = value
                best_position = (i, j)

    i, j = best_position
    mapping: list[tuple[int | None, int | None]] = []
    diagonal_similarities = []
    while i and j and scores[i][j] > 0:
        direction = traces[i][j]
        if direction == "斜":
            left_column = left_columns[i - 1]
            right_column = right_columns[j - 1]
            mapping.append((left_column, right_column))
            diagonal_similarities.append(similarities[i - 1][j - 1])
            i -= 1
            j -= 1
        elif direction == "上":
            mapping.append((left_columns[i - 1], None))
            i -= 1
        elif direction == "左":
            mapping.append((None, right_columns[j - 1]))
            j -= 1
        else:
            break
    mapping.reverse()
    diagonals = [
        (left_column, right_column)
        for left_column, right_column in mapping
        if left_column is not None and right_column is not None
    ]
    strong = sum(value >= 0.72 for value in diagonal_similarities)
    average = (
        sum(diagonal_similarities) / len(diagonal_similarities)
        if diagonal_similarities
        else 0.0
    )
    if not diagonals:
        return None, {
            "方式": "未找到列序列重叠",
            "最高局部得分": best_score,
        }
    first_left = min(left_column for left_column, _ in diagonals)
    first_right = min(right_column for _, right_column in diagonals)
    last_left = max(left_column for left_column, _ in diagonals)
    last_right = max(right_column for _, right_column in diagonals)
    valid = (
        len(diagonals) >= 2
        and strong >= 2
        and average >= 0.68
        and (
            first_right <= 4
            or (
                first_right <= 12
                and strong >= 8
                and average >= 0.88
            )
        )
        and (
            left_width - 1 - last_left <= 3
            or (
                left_width - 1 - last_left <= 12
                and strong >= 8
                and average >= 0.88
            )
        )
    )
    audit = {
        "方式": "列指纹局部动态对齐" if valid else "列序列证据不足",
        "最高局部得分": best_score,
        "对角匹配列数": len(diagonals),
        "强匹配列数": strong,
        "平均列相似度": average,
        "左侧重叠起点": first_left,
        "右侧匹配前疑似OCR插列数": first_right,
        "左侧末端未匹配列数": left_width - 1 - last_left,
        "右侧重叠结束位置": last_right,
    }
    if not valid:
        return None, audit

    # 左侧末端即使没有在右块找到对应，也必须保留在重叠映射内。
    used_left = [
        left_column for left_column, _ in mapping
        if left_column is not None
    ]
    for left_column in range(max(used_left) + 1, left_width):
        mapping.append((left_column, None))
    return mapping, audit


def 左右拼接(left_matrix: dict[str, Any], right_matrix: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left, right = 规整(left_matrix), 规整(right_matrix)
    a, b = left["单元格"], right["单元格"]
    alignment, align_audit = 对齐(a, b)
    column_mapping, column_audit = 列序列局部对齐(
        a,
        b,
        alignment,
    )
    left_width, right_width = 尺寸(left)[1], 尺寸(right)[1]
    rows, conflicts = [], 0

    if column_mapping is not None:
        overlap_left_start = min(
            left_column for left_column, _ in column_mapping
            if left_column is not None
        )
        overlap_right_end = max(
            right_column for _, right_column in column_mapping
            if right_column is not None
        ) + 1
    else:
        fixed_overlap, fixed_audit = 重叠列数(a, b, alignment)
        column_audit["固定重叠回退"] = fixed_audit
        overlap_left_start = left_width - fixed_overlap
        overlap_right_end = fixed_overlap
        column_mapping = [
            (overlap_left_start + offset, offset)
            for offset in range(fixed_overlap)
        ]

    for left_row_index, right_row_index in alignment:
        left_row = (
            a[left_row_index] if left_row_index is not None
            else [""] * left_width
        )
        right_row = (
            b[right_row_index] if right_row_index is not None
            else [""] * right_width
        )
        merged_overlap = []
        for left_column, right_column in column_mapping:
            left_value = (
                left_row[left_column] if left_column is not None else ""
            )
            right_value = (
                right_row[right_column] if right_column is not None else ""
            )
            if (
                left_value.strip()
                and right_value.strip()
                and SequenceMatcher(
                    None,
                    标准文字(left_value),
                    标准文字(right_value),
                ).ratio() < 0.72
            ):
                conflicts += 1
            merged_overlap.append(
                left_value if left_value.strip() else right_value
            )
        rows.append(
            [
                *left_row[:overlap_left_start],
                *merged_overlap,
                *right_row[overlap_right_end:],
            ]
        )
    matrix = {
        "单元格": rows,
        "前置文字": left.get("前置文字", ""),
        "后置文字": right.get("后置文字", ""),
    }
    audit = {
        "方向": "左右",
        "左尺寸": 尺寸(left),
        "右尺寸": 尺寸(right),
        "输出尺寸": 尺寸(matrix),
        "行对齐": align_audit,
        "列序列对齐": column_audit,
        "重叠冲突格": conflicts,
        "补空行数": sum(
            left_row is None or right_row is None
            for left_row, right_row in alignment
        ),
    }
    return matrix, audit


def 转置(matrix: dict[str, Any]) -> dict[str, Any]:
    matrix = 规整(matrix)
    rows, columns = 尺寸(matrix)
    return {
        "单元格": [
            [matrix["单元格"][row][column] for row in range(rows)]
            for column in range(columns)
        ],
        "前置文字": matrix.get("前置文字", ""),
        "后置文字": matrix.get("后置文字", ""),
    }


def 上下拼接(top: dict[str, Any], bottom: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    transposed, audit = 左右拼接(转置(top), 转置(bottom))
    result = 转置(transposed)
    audit["方向"] = "上下"
    audit["列对齐"] = audit.pop("行对齐")
    audit["行序列对齐"] = audit.pop("列序列对齐")
    audit["补空列数"] = audit.pop("补空行数")
    return result, audit


def 渲染(matrix: dict[str, Any]) -> str:
    matrix = 规整(matrix)
    lines = ["<table>"]
    for row in matrix["单元格"]:
        lines.append("  <tr>")
        for cell in row:
            lines.append(f"    <td>{escape(cell).replace(chr(10), '<br>')}</td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n\n".join(
        part for part in (
            matrix.get("前置文字", "").strip(),
            "\n".join(lines),
            matrix.get("后置文字", "").strip(),
        ) if part
    )


def 保存矩阵(base: Path, matrix: dict[str, Any], audit: Any = None) -> None:
    rows, columns = 尺寸(matrix)
    写JSON(base.with_suffix(".json"), {
        "行数": rows, "列数": columns,
        "前置文字": matrix.get("前置文字", ""),
        "后置文字": matrix.get("后置文字", ""),
        "解析方式": matrix.get("解析方式", ""),
        "单元格": matrix["单元格"], "拼接审计": audit,
    })
    base.with_suffix(".html").write_text(渲染(matrix), encoding="utf-8")


def API识别并拼接(
    work_dir: Path, manifests: list[dict[str, Any]],
    workers: int, timeout: int, retries: int,
) -> dict[str, Any]:
    cache = ResultCache(work_dir / "粗切实验API缓存.sqlite3")
    client = FinixDocClient.from_official_doc(
        项目根目录 / "FinixDoc_VL调用.txt",
        user_id="finixB2002", timeout=timeout, max_retries=retries,
    )
    all_tiles = [tile for manifest in manifests for tile in manifest["切块"]]

    statistics = {
        "API新请求": 0,
        "缓存命中": 0,
        "本地原始响应复用": 0,
        "自动复切父块": 0,
    }
    statistics_lock = Lock()

    def count(name: str) -> None:
        with statistics_lock:
            statistics[name] += 1

    def request_matrix(
        path: Path,
        tile: dict[str, Any],
        label: str,
        raw_path: Path,
        *,
        depth: int = 0,
    ) -> tuple[dict[str, Any], str]:
        key = cache.tile_key(path.read_bytes(), "", client.model+"@coarse-v1")
        response = cache.get_tile(key)
        source = "缓存"
        if response is not None:
            count("缓存命中")
        elif raw_path.is_file():
            response = raw_path.read_text(encoding="utf-8")
            source = "本地原始响应"
            count("本地原始响应复用")
        else:
            source = "API"
            print(f"[粗切API] {label}", flush=True)
            response = client.recognize(path, "", request_label=label)
            count("API新请求")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(response, encoding="utf-8")
        try:
            matrix = 解析矩阵(response)
        except Exception as error:
            # 只对模型输出截断/无法形成表格的块做递归复切；原块和坏响应保留。
            if depth >= 7:
                raise RuntimeError(path) from error
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
            width, height = image.size
            split_x = width >= height and width >= 300
            split_y = not split_x and height >= 300
            if not split_x and not split_y:
                raise RuntimeError(path) from error

            repair_image_dir = path.parent / "失败块自动复切"
            repair_raw_dir = raw_path.parent / "失败块自动复切"
            repair_matrix_dir = (
                work_dir
                / Path(tile["图片名"]).stem
                / "04_模型二维矩阵"
                / "失败块自动复切"
            )
            repair_image_dir.mkdir(parents=True, exist_ok=True)
            repair_matrix_dir.mkdir(parents=True, exist_ok=True)
            context = 96
            if split_x:
                middle = width // 2
                boxes = (
                    (0, 0, min(width, middle + context), height),
                    (max(0, middle - context), 0, width, height),
                )
                direction = "左右"
                names = ("左半", "右半")
            else:
                middle = height // 2
                boxes = (
                    (0, 0, width, min(height, middle + context)),
                    (0, max(0, middle - context), width, height),
                )
                direction = "上下"
                names = ("上半", "下半")

            child_matrices = []
            child_sources = []
            child_records = []
            for child_index, (box, name) in enumerate(zip(boxes, names)):
                child_path = (
                    repair_image_dir
                    / f"{path.stem}_第{depth+1}层_{name}.png"
                )
                image.crop(box).save(child_path)
                child_raw = (
                    repair_raw_dir
                    / f"{raw_path.stem}_第{depth+1}层_{name}.md"
                )
                child_matrix, child_source = request_matrix(
                    child_path,
                    tile,
                    f"{label} / 自动复切第{depth+1}层{name}",
                    child_raw,
                    depth=depth + 1,
                )
                保存矩阵(
                    repair_matrix_dir / child_path.stem,
                    child_matrix,
                )
                child_matrices.append(child_matrix)
                child_sources.append(child_source)
                child_records.append(
                    {
                        "名称": name,
                        "图片": str(child_path.resolve()),
                        "裁切框": list(box),
                        "矩阵尺寸": 尺寸(child_matrix),
                        "来源": child_source,
                    }
                )
            if direction == "左右":
                matrix, merge_audit = 左右拼接(
                    child_matrices[0],
                    child_matrices[1],
                )
            else:
                matrix, merge_audit = 上下拼接(
                    child_matrices[0],
                    child_matrices[1],
                )
            repair_audit = {
                "父块": str(path.resolve()),
                "触发错误": f"{type(error).__name__}：{error}",
                "复切层级": depth + 1,
                "复切方向": direction,
                "子块": child_records,
                "子块拼接": merge_audit,
                "修复矩阵尺寸": 尺寸(matrix),
            }
            写JSON(
                repair_matrix_dir / f"{path.stem}_第{depth+1}层_修复审计.json",
                repair_audit,
            )
            保存矩阵(
                repair_matrix_dir / f"{path.stem}_第{depth+1}层_修复结果",
                matrix,
                repair_audit,
            )
            # 父块缓存保存的是修复后的规整HTML，下次无需再次进入复切。
            cache.put_tile(
                key,
                渲染(matrix),
                {
                    "实验": "粗切V1自动复切",
                    "切块": tile,
                    "父块": str(path.resolve()),
                    "修复审计": repair_audit,
                },
            )
            count("自动复切父块")
            return matrix, "+".join(child_sources) + "+自动复切"

        if source != "缓存":
            cache.put_tile(
                key,
                response,
                {
                    "实验": "粗切V1",
                    "切块": tile,
                    "尺寸": 尺寸(matrix),
                    "来源": source,
                },
            )
        return matrix, source

    def recognize(tile: dict[str, Any]) -> dict[str, Any]:
        path = Path(tile["文件路径"])
        sample = work_dir / Path(tile["图片名"]).stem
        stem = path.stem
        raw_dir = sample / "03_API原始响应"
        matrix_dir = sample / "04_模型二维矩阵"
        raw_dir.mkdir(parents=True, exist_ok=True)
        matrix_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{stem}.md"
        label = (
            f"原图 {tile['图片名']} / 表{tile['表序号']+1} / "
            f"行带{tile['行带序号']+1} / 列块{tile['列块序号']+1}"
        )
        try:
            matrix, source = request_matrix(
                path,
                tile,
                label,
                raw_path,
            )
        except Exception as error:
            (raw_dir / f"{stem}_解析失败.txt").write_text(
                f"{type(error).__name__}：{error}" + chr(10) * 2
                + f"原始响应已保存在：{raw_path}",
                encoding="utf-8",
            )
            raise
        保存矩阵(matrix_dir/stem, matrix)
        return {"切块": tile, "矩阵": matrix, "来源": source}

    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(recognize, tile): tile for tile in all_tiles}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append({"切块": tile, "错误类型": type(error).__name__, "错误": str(error)})
                print(
                    f"[粗切失败] {tile['图片名']} 表{tile['表序号']+1}-"
                    f"行{tile['行带序号']+1}-列{tile['列块序号']+1}：{error}",
                    flush=True,
                )

    image_results = []
    for manifest in manifests:
        image_name = Path(manifest["图片"]).name
        expected = manifest["切块"]
        items = [item for item in successes if item["切块"]["图片名"] == image_name]
        if len(items) != len(expected):
            image_results.append({
                "图片": image_name, "状态": "切块不完整，未拼接",
                "成功切块": len(items), "应有切块": len(expected),
            })
            continue
        lookup = {
            (item["切块"]["表序号"], item["切块"]["行带序号"], item["切块"]["列块序号"]): item["矩阵"]
            for item in items
        }
        sample = work_dir / Path(image_name).stem
        process_dir, final_dir = sample/"05_逐步拼接审计", sample/"06_最终拼接结果"
        process_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        tables, audits = [], []
        for table_index in range(manifest["分表数量"]):
            table_tiles = [tile for tile in expected if tile["表序号"] == table_index]
            row_count = max(tile["行带总数"] for tile in table_tiles)
            column_count = max(tile["列块总数"] for tile in table_tiles)
            row_bands = []
            for row_index in range(row_count):
                current = lookup[(table_index, row_index, 0)]
                for column_index in range(1, column_count):
                    current, audit = 左右拼接(current, lookup[(table_index, row_index, column_index)])
                    audit.update({"表": table_index+1, "行带": row_index+1, "接入列块": column_index+1})
                    audits.append(audit)
                    保存矩阵(
                        process_dir/f"第{table_index+1:03d}表_第{row_index+1:03d}行带_拼至第{column_index+1:03d}列块",
                        current, audit,
                    )
                row_bands.append(current)
            table = row_bands[0]
            for row_index in range(1, len(row_bands)):
                table, audit = 上下拼接(table, row_bands[row_index])
                audit.update({"表": table_index+1, "接入行带": row_index+1})
                audits.append(audit)
                保存矩阵(
                    process_dir/f"第{table_index+1:03d}表_拼至第{row_index+1:03d}行带",
                    table, audit,
                )
            保存矩阵(final_dir/f"第{table_index+1:03d}表_最终结果", table, audits)
            tables.append(table)
        final_path = final_dir/"整张图片最终结果.html"
        final_path.write_text("\n\n".join(渲染(table) for table in tables), encoding="utf-8")
        image_results.append({
            "图片": image_name, "状态": "完成", "表数": len(tables),
            "最终矩阵": [尺寸(table) for table in tables],
            "低置信拼接数": sum("无可靠内容锚点" in str(audit) for audit in audits),
            "冲突拼接数": sum(bool(audit.get("重叠冲突格")) for audit in audits),
            "结果": str(final_path.resolve()),
        })

    summary = {
        "总切块": len(all_tiles), "成功切块": len(successes), "失败切块": len(failures),
        **statistics,
        "图片结果": image_results, "失败详情": failures,
    }
    写JSON(work_dir/"API识别与拼接汇总.json", summary)
    return summary


def 写总览(work_dir: Path, manifests: list[dict[str, Any]], summary: dict[str, Any] | None) -> None:
    results = {item["图片"]: item for item in (summary or {}).get("图片结果", [])}
    rows = []
    for manifest in manifests:
        name = Path(manifest["图片"]).name
        stem = Path(name).stem
        result = results.get(name, {})
        final = result.get("结果")
        final_link = (
            f'<a href="{Path(final).relative_to(work_dir).as_posix()}">查看最终拼接</a>'
            if final else "尚未完成"
        )
        rows.append(
            "<tr><td>"+"</td><td>".join([
                escape(name), str(manifest["分表数量"]), str(manifest["API切块数量"]),
                f'<a href="{stem}/01_粗切块位置总览.png">查看切块位置</a>',
                final_link, escape(str(result.get("最终矩阵", ""))),
                escape(str(result.get("状态", ""))),
            ])+"</td></tr>"
        )
    (work_dir/"粗切实验总览.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>粗切实验总览</title>
<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:7px}th{background:#eee}</style>
<h1>粗切块相信模型V1</h1><p>不使用黑线、白缝、R×C或墨迹矩阵。</p>
<table><tr><th>图片</th><th>分表数</th><th>API切块数</th><th>切块位置</th>
<th>结果</th><th>最终矩阵</th><th>状态</th></tr>"""
        +"".join(rows)+"</table>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--只跑", default=",".join(默认样本))
    parser.add_argument("--仅预处理", action="store_true")
    parser.add_argument("--责任区边长", type=int, default=3200)
    parser.add_argument("--重叠像素", type=int, default=256)
    parser.add_argument("--并行数", type=int, default=2)
    parser.add_argument("--超时秒数", type=int, default=600)
    parser.add_argument("--重试次数", type=int, default=3)
    args = parser.parse_args()

    input_dir = 项目根目录/"raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images"
    work_dir = 项目根目录/"work/验证/粗切块相信模型V1"
    work_dir.mkdir(parents=True, exist_ok=True)
    config = TableConfig.from_json(项目根目录/"afac_pipeline/table/config.example.json")
    prefixes = tuple(item.strip() for item in args.只跑.split(",") if item.strip())
    manifests = [
        生成切块(path, work_dir, config, args.责任区边长, args.重叠像素)
        for path in 找图片(input_dir, prefixes)
    ]
    写JSON(work_dir/"全部粗切块清单.json", manifests)
    if args.仅预处理:
        写总览(work_dir, manifests, None)
        print(f"[粗切预处理完成] {work_dir/'粗切实验总览.html'}")
        return 0
    summary = API识别并拼接(work_dir, manifests, args.并行数, args.超时秒数, args.重试次数)
    写总览(work_dir, manifests, summary)
    print(
        f"[粗切实验完成] 成功切块 {summary['成功切块']}/{summary['总切块']}\n"
        f"总览：{work_dir/'粗切实验总览.html'}"
    )
    return 0 if not summary["失败切块"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
