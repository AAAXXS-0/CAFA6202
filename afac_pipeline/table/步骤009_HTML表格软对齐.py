"""按逻辑行列坐标解析、校验并合并 FinixDoc-VL 的 HTML 表格。"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
import re

from ..common.models import TilePlan
from .步骤008_Markdown表格合并 import MarkdownMergeError, parse_first_table


class HtmlTableMergeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Cell:
    text: str
    tag: str = "td"
    rowspan: int = 1
    colspan: int = 1


@dataclass(frozen=True)
class Placement:
    cell: Cell
    row: int
    column: int


@dataclass
class ParsedResponse:
    prefix: str
    suffix: str
    placements: list[Placement]
    row_count: int
    column_count: int
    # 只记录不会改变任何可见文字的保守格式修复，最后随质量报告输出。
    format_warnings: list[str] = field(default_factory=list)


class _FirstTableParser(HTMLParser):
    """只解析响应中的第一张表，保留 th/td 及 rowspan/colspan。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_depth = 0
        self.finished = False
        self.rows: list[list[Cell]] = []
        self.current_row: list[Cell] | None = None
        self.current_tag: str | None = None
        self.current_attrs: dict[str, str] = {}
        self.current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table" and not self.finished:
            self.table_depth += 1
            return
        if self.table_depth != 1:
            return
        if tag == "tr":
            self.current_row = []
        elif tag in {"th", "td"} and self.current_row is not None:
            self.current_tag = tag
            self.current_attrs = {key.lower(): value or "" for key, value in attrs}
            self.current_text = []
        elif tag == "br" and self.current_tag is not None:
            self.current_text.append("\n")

    def handle_data(self, data: str) -> None:
        if self.table_depth == 1 and self.current_tag is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.table_depth != 1:
            if tag == "table" and self.table_depth > 0:
                self.table_depth -= 1
            return
        if (
            tag in {"th", "td"}
            and self.current_tag == tag
            and self.current_row is not None
        ):
            try:
                rowspan = max(1, int(self.current_attrs.get("rowspan", "1")))
                colspan = max(1, int(self.current_attrs.get("colspan", "1")))
            except ValueError as error:
                raise HtmlTableMergeError(
                    "HTML 单元格的 rowspan/colspan 不是整数"
                ) from error
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.current_text)).strip()
            self.current_row.append(Cell(text, tag, rowspan, colspan))
            self.current_tag = None
            self.current_attrs = {}
            self.current_text = []
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table":
            self.table_depth = 0
            self.finished = True


def _expand_rows(rows: list[list[Cell]]) -> tuple[list[Placement], int, int]:
    placements: list[Placement] = []
    occupied: set[tuple[int, int]] = set()
    row_count = len(rows)
    column_count = 0
    for row_index, cells in enumerate(rows):
        column = 0
        for cell in cells:
            while (row_index, column) in occupied:
                column += 1
            placement = Placement(cell, row_index, column)
            placements.append(placement)
            for row in range(row_index, row_index + cell.rowspan):
                for col in range(column, column + cell.colspan):
                    if (row, col) in occupied:
                        raise HtmlTableMergeError("HTML 表格的合并单元格互相覆盖")
                    occupied.add((row, col))
            row_count = max(row_count, row_index + cell.rowspan)
            column_count = max(column_count, column + cell.colspan)
            column += cell.colspan
    return placements, row_count, column_count


def _parse_html(response: str) -> ParsedResponse | None:
    match = re.search(r"<table\b[\s\S]*?</table\s*>", response, re.IGNORECASE)
    if match is None:
        return None
    parser = _FirstTableParser()
    parser.feed(match.group(0))
    if not parser.rows:
        raise HtmlTableMergeError("HTML 中存在 table 标签，但没有可解析的表格行")
    placements, rows, columns = _expand_rows(parser.rows)
    return ParsedResponse(
        response[: match.start()].strip(),
        response[match.end() :].strip(),
        placements,
        rows,
        columns,
    )


def _parse_markdown(response: str) -> ParsedResponse:
    """兼容模型偶尔无视 HTML 要求而返回标准 Markdown 表格。"""

    try:
        table = parse_first_table(response)
    except MarkdownMergeError as error:
        raise HtmlTableMergeError("响应中没有找到 HTML 或 Markdown 表格") from error
    rows = [
        [Cell(value, "th") for value in table.header],
        *[[Cell(value, "td") for value in row] for row in table.rows],
    ]
    placements, row_count, column_count = _expand_rows(rows)
    return ParsedResponse(
        table.prefix,
        table.suffix,
        placements,
        row_count,
        column_count,
    )


def parse_table_response(response: str) -> ParsedResponse:
    return _parse_html(response) or _parse_markdown(response)


def _visible_text_signature(response: str) -> str:
    """提取标签外的可见文字，用来证明自动修复没有改动识别内容。"""

    without_tags = re.sub(r"<[^>]*>", "", response)
    return re.sub(r"\s+", " ", without_tags).strip()


def _repair_missing_final_table_end(response: str) -> str | None:
    """只修复位于响应末尾、且仅缺 ``</table>`` 的单一 HTML 表。

    模型真正被 token 截断时，常常还会缺 ``</td>`` 或 ``</tr>``。那种情况
    无法证明最后一个单元格是否完整，必须继续判为损坏并重试，不能猜着补。
    """

    if len(re.findall(r"<table\b", response, re.IGNORECASE)) != 1:
        return None
    if re.search(r"</table\s*>", response, re.IGNORECASE):
        return None
    if len(re.findall(r"<tr\b", response, re.IGNORECASE)) != len(
        re.findall(r"</tr\s*>", response, re.IGNORECASE)
    ):
        return None
    for tag in ("td", "th"):
        if len(re.findall(rf"<{tag}\b", response, re.IGNORECASE)) != len(
            re.findall(rf"</{tag}\s*>", response, re.IGNORECASE)
        ):
            return None
    last_row_end = list(re.finditer(r"</tr\s*>", response, re.IGNORECASE))
    if not last_row_end or response[last_row_end[-1].end() :].strip():
        return None

    repaired = response.rstrip() + "\n</table>"
    if _visible_text_signature(repaired) != _visible_text_signature(response):
        return None
    return repaired


def _response_format_guard(response: str) -> None:
    """在宽松解析前拦截截断、多表重复和围栏循环。

    HTMLParser 会容忍不少残缺标签，这对浏览器是优点，但对切片缓存很危险：
    半截结果也可能被当成成功。因此先检查正式运行中已经出现过的损坏形态。
    """

    table_starts = len(re.findall(r"<table\b", response, re.IGNORECASE))
    table_ends = len(re.findall(r"</table\s*>", response, re.IGNORECASE))
    if table_starts != table_ends:
        raise HtmlTableMergeError(
            f"HTML 表格标签不闭合：开始 {table_starts} 个，结束 {table_ends} 个"
        )
    if table_starts > 1:
        raise HtmlTableMergeError(
            f"单个切片返回了 {table_starts} 张 HTML 表，疑似重复生成"
        )
    fence_count = response.count("```")
    if fence_count > 4:
        raise HtmlTableMergeError(
            f"响应包含 {fence_count} 个代码围栏，疑似 Markdown 围栏循环"
        )


def parse_table_response_checked(response: str) -> ParsedResponse:
    """解析单个切片响应，并拒绝已知的损坏输出。"""

    try:
        _response_format_guard(response)
    except HtmlTableMergeError:
        repaired = _repair_missing_final_table_end(response)
        if repaired is None:
            raise
        _response_format_guard(repaired)
        parsed = parse_table_response(repaired)
        parsed.format_warnings.append(
            "模型只遗漏了响应末尾的 </table>；已在不改变可见文字的前提下补全"
        )
        return parsed
    return parse_table_response(response)


def _render(
    placements: list[Placement],
    row_count: int,
    column_count: int,
    prefix: str = "",
    suffix: str = "",
) -> str:
    anchors = {(item.row, item.column): item for item in placements}
    covered: set[tuple[int, int]] = set()
    lines = ["<table>"]
    for row in range(row_count):
        lines.append("  <tr>")
        for column in range(column_count):
            if (row, column) in covered:
                continue
            item = anchors.get((row, column))
            if item is None:
                lines.append("    <td></td>")
                continue
            cell = item.cell
            attrs = ""
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            value = escape(cell.text).replace("\n", "<br>")
            lines.append(f"    <{cell.tag}{attrs}>{value}</{cell.tag}>")
            for covered_row in range(row, row + cell.rowspan):
                for covered_column in range(column, column + cell.colspan):
                    if (covered_row, covered_column) != (row, column):
                        covered.add((covered_row, covered_column))
        lines.append("  </tr>")
    lines.append("</table>")
    parts = [
        part for part in (prefix.strip(), "\n".join(lines), suffix.strip()) if part
    ]
    return "\n".join(parts)


def render_empty_table(row_count: int, column_count: int) -> str:
    """按预处理识别到的行列数生成一张全空 HTML 表。

    这不是让模型猜测内容，而是保留预处理已经确定的表格形状。
    它主要用于“只有网格线、每个格子都没有文字”的切片。
    """
    if row_count <= 0 or column_count <= 0:
        raise ValueError("空表的行数和列数必须大于 0")
    return _render([], row_count, column_count)


def normalize_table_response(response: str) -> tuple[str, dict[str, int]]:
    parsed = parse_table_response_checked(response)
    return (
        _render(
            parsed.placements,
            parsed.row_count,
            parsed.column_count,
            parsed.prefix,
            parsed.suffix,
        ),
        {"rows": parsed.row_count, "columns": parsed.column_count},
    )


def _simple_rows(parsed: ParsedResponse) -> list[list[Cell]] | None:
    """把没有合并单元格的响应还原成逐行序列，供软对齐使用。"""

    if any(
        item.cell.rowspan != 1 or item.cell.colspan != 1 for item in parsed.placements
    ):
        return None
    rows: list[list[Cell]] = [[] for _ in range(parsed.row_count)]
    for item in sorted(parsed.placements, key=lambda value: (value.row, value.column)):
        if item.column != len(rows[item.row]):
            return None
        rows[item.row].append(item.cell)
    return rows


def _cell_signature(cell: Cell) -> str:
    # 重复表头有时会在 th/td 间漂移，判重只比较真实文字。
    return cell.text.strip()


def _remove_extra_simple_structure(
    rows: list[list[Cell]],
    expected_rows: int,
    expected_columns: int,
) -> tuple[list[list[Cell]], list[str]]:
    """删除模型偶尔附加的全空边缘和重复表头/首列。"""

    warnings = []
    width = max((len(row) for row in rows), default=0)
    matrix = [[*row, *[Cell("") for _ in range(width - len(row))]] for row in rows]

    # 常见两层表头会共用同一组列值，例如“保单年度 70…80”下一行是
    # “年龄 70…80”。若预处理只得到一条表头行，不裁掉任何独有文字，
    # 而是把两个左上角标签合进同一个物理格，完全相同的列值只保留一次。
    if (
        len(matrix) == expected_rows + 1
        and width == expected_columns
        and width >= 2
        and matrix[0][0].text.strip()
        and matrix[1][0].text.strip()
        and matrix[0][0].text.strip() != matrix[1][0].text.strip()
        and tuple(_cell_signature(cell) for cell in matrix[0][1:])
        == tuple(_cell_signature(cell) for cell in matrix[1][1:])
        and any(cell.text.strip() for cell in matrix[0][1:])
    ):
        first_label = matrix[0][0].text.strip()
        second = matrix[1][0]
        matrix[1][0] = Cell(
            f"{first_label}\n{second.text.strip()}",
            second.tag,
        )
        matrix.pop(0)
        warnings.append(
            "模型识别出共享列值的两层表头；已合并两个左上角标签，"
            "没有丢弃任何不同文字"
        )

    removed_empty_rows = 0
    while len(matrix) > expected_rows:
        index = next(
            (
                i
                for i, row in enumerate(matrix)
                if not any(cell.text.strip() for cell in row)
            ),
            None,
        )
        if index is None:
            break
        matrix.pop(index)
        removed_empty_rows += 1

    removed_repeated_rows = 0
    while len(matrix) > expected_rows and matrix:
        first = tuple(_cell_signature(cell) for cell in matrix[0])
        index = next(
            (
                i
                for i, row in enumerate(matrix[1:], start=1)
                if tuple(_cell_signature(cell) for cell in row) == first
            ),
            None,
        )
        if index is None:
            break
        matrix.pop(index)
        removed_repeated_rows += 1

    removed_empty_columns = 0
    while matrix and len(matrix[0]) > expected_columns:
        index = next(
            (
                column
                for column in range(len(matrix[0]))
                if not any(row[column].text.strip() for row in matrix)
            ),
            None,
        )
        if index is None:
            break
        for row in matrix:
            row.pop(index)
        removed_empty_columns += 1

    removed_repeated_columns = 0
    while matrix and len(matrix[0]) > expected_columns:
        first = tuple(_cell_signature(row[0]) for row in matrix)
        index = next(
            (
                column
                for column in range(1, len(matrix[0]))
                if tuple(_cell_signature(row[column]) for row in matrix) == first
            ),
            None,
        )
        if index is None:
            break
        for row in matrix:
            row.pop(index)
        removed_repeated_columns += 1

    if (
        removed_empty_rows
        + removed_repeated_rows
        + removed_empty_columns
        + removed_repeated_columns
    ):
        warnings.append(
            "已删除模型额外生成的无信息结构："
            f"空行 {removed_empty_rows}、重复表头行 {removed_repeated_rows}、"
            f"空列 {removed_empty_columns}、重复首列 {removed_repeated_columns}"
        )
    return matrix, warnings


def _sequence_mapping(
    expected_count: int,
    actual_count: int,
    pair_cost,
    skip_expected_cost,
) -> list[int] | None:
    """保持顺序地把模型序列放回更长的预处理槽位。"""

    if actual_count > expected_count:
        return None
    infinity = float("inf")
    costs = [[infinity] * (actual_count + 1) for _ in range(expected_count + 1)]
    previous: list[list[tuple[int, int, bool] | None]] = [
        [None] * (actual_count + 1) for _ in range(expected_count + 1)
    ]
    costs[0][0] = 0.0
    for expected in range(expected_count):
        for actual in range(actual_count + 1):
            current = costs[expected][actual]
            if current == infinity:
                continue
            skipped = current + float(skip_expected_cost(expected))
            if skipped < costs[expected + 1][actual]:
                costs[expected + 1][actual] = skipped
                previous[expected + 1][actual] = (expected, actual, False)
            if actual < actual_count:
                paired = current + float(pair_cost(expected, actual))
                if paired < costs[expected + 1][actual + 1]:
                    costs[expected + 1][actual + 1] = paired
                    previous[expected + 1][actual + 1] = (
                        expected,
                        actual,
                        True,
                    )
    if costs[expected_count][actual_count] == infinity:
        return None
    mapping = [-1] * actual_count
    state = (expected_count, actual_count)
    while state != (0, 0):
        item = previous[state[0]][state[1]]
        if item is None:
            return None
        old_expected, old_actual, paired = item
        if paired:
            mapping[old_actual] = old_expected
        state = old_expected, old_actual
    return mapping


def _soft_align_parsed(
    parsed: ParsedResponse,
    expected_rows: int,
    expected_columns: int,
    ink_mask: list[list[bool]] | None,
) -> tuple[ParsedResponse, list[str]]:
    """把模型内容放回预处理确定的固定物理矩阵。"""

    if parsed.row_count == expected_rows and parsed.column_count == expected_columns:
        return parsed, list(parsed.format_warnings)

    warning = (
        f"模型返回 {parsed.row_count}×{parsed.column_count}，"
        f"预处理物理结构为 {expected_rows}×{expected_columns}"
    )
    rows = _simple_rows(parsed)
    if rows is None:
        clipped = []
        clipped_count = 0
        for item in parsed.placements:
            if item.row >= expected_rows or item.column >= expected_columns:
                if item.cell.text.strip():
                    raise HtmlTableMergeError(
                        warning + "；合并单元格在预处理网格之外仍含非空内容"
                    )
                clipped_count += 1
                continue
            rowspan = min(item.cell.rowspan, expected_rows - item.row)
            colspan = min(item.cell.colspan, expected_columns - item.column)
            if rowspan != item.cell.rowspan or colspan != item.cell.colspan:
                clipped_count += 1
            clipped.append(
                Placement(
                    Cell(item.cell.text, item.cell.tag, rowspan, colspan),
                    item.row,
                    item.column,
                )
            )
        detail = "；已限制在预处理物理边界内"
        if clipped_count:
            detail += f"，裁掉 {clipped_count} 个越界空位或跨度"
        return (
            ParsedResponse(
                parsed.prefix, parsed.suffix, clipped, expected_rows, expected_columns
            ),
            [*parsed.format_warnings, warning + detail],
        )

    rows, cleanup_warnings = _remove_extra_simple_structure(
        rows, expected_rows, expected_columns
    )
    if len(rows) > expected_rows or any(len(row) > expected_columns for row in rows):
        raise HtmlTableMergeError(
            warning + "；删除空行空列及重复表头后仍有非空结构超出预处理网格"
        )

    normalized_mask = [
        [
            bool(
                ink_mask
                and row < len(ink_mask)
                and column < len(ink_mask[row])
                and ink_mask[row][column]
            )
            for column in range(expected_columns)
        ]
        for row in range(expected_rows)
    ]
    if not ink_mask:
        normalized_mask = [[True] * expected_columns for _ in range(expected_rows)]
    expected_row_ink = [sum(row) for row in normalized_mask]
    actual_row_ink = [sum(bool(cell.text.strip()) for cell in row) for row in rows]
    row_mapping = _sequence_mapping(
        expected_rows,
        len(rows),
        lambda expected, actual: (
            abs(expected_row_ink[expected] - actual_row_ink[actual])
            / max(1, expected_row_ink[expected])
        ),
        lambda expected: 0.0 if expected_row_ink[expected] == 0 else 2.0,
    )
    if row_mapping is None:
        raise HtmlTableMergeError(warning + "；无法可靠映射到预处理行坐标")

    aligned = []
    for actual_row, cells in enumerate(rows):
        expected_row = row_mapping[actual_row]
        flags = normalized_mask[expected_row]
        column_mapping = _sequence_mapping(
            expected_columns,
            len(cells),
            lambda expected, actual: (
                0.0
                if flags[expected] == bool(cells[actual].text.strip())
                else (2.5 if cells[actual].text.strip() else 1.0)
            ),
            lambda expected: 0.0 if not flags[expected] else 1.5,
        )
        if column_mapping is None:
            raise HtmlTableMergeError(
                warning + f"；第 {actual_row + 1} 行无法可靠映射到预处理列坐标"
            )
        for actual_column, cell in enumerate(cells):
            aligned.append(Placement(cell, expected_row, column_mapping[actual_column]))
    return (
        ParsedResponse(
            parsed.prefix, parsed.suffix, aligned, expected_rows, expected_columns
        ),
        [
            *parsed.format_warnings,
            *cleanup_warnings,
            warning + "；已按单元格墨迹补齐空位",
        ],
    )


def normalize_table_response_soft(
    response: str,
    expected_rows: int,
    expected_columns: int,
    ink_mask: list[list[bool]] | None = None,
) -> tuple[str, dict[str, object]]:
    """固定预处理行列数，并把模型内容映射到对应物理格。"""

    if expected_rows <= 0 or expected_columns <= 0:
        raise ValueError("预处理物理行列数必须大于 0")
    parsed = parse_table_response_checked(response)
    actual = {"rows": parsed.row_count, "columns": parsed.column_count}
    aligned, warnings = _soft_align_parsed(
        parsed,
        expected_rows,
        expected_columns,
        ink_mask,
    )
    return (
        _render(
            aligned.placements,
            aligned.row_count,
            aligned.column_count,
            aligned.prefix,
            aligned.suffix,
        ),
        {
            **actual,
            "nonempty_cells": sum(
                bool(item.cell.text.strip()) for item in parsed.placements
            ),
            "warnings": warnings,
            "physical_rows": expected_rows,
            "physical_columns": expected_columns,
        },
    )


def merge_logical_tiles(
    contents: dict[tuple[int, int], str],
    plans: list[TilePlan],
    logical_row_count: int,
    logical_column_count: int,
    tile_ink_masks: dict[tuple[int, int], list[list[bool]]] | None = None,
) -> tuple[str, dict[str, object]]:
    """删除重复上下文后，把各块放回全局逻辑坐标并保留合并单元格。"""

    plan_map = {(plan.row_index, plan.column_index): plan for plan in plans}
    if set(contents) != set(plan_map):
        missing = sorted(set(plan_map) - set(contents))
        raise HtmlTableMergeError(f"缺少表格切片响应：{missing}")

    global_placements: list[Placement] = []
    occupied: set[tuple[int, int]] = set()
    tile_reports: list[dict[str, object]] = []
    parsed_by_key: dict[tuple[int, int], ParsedResponse] = {}
    all_warnings: list[str] = []
    for key in sorted(contents):
        plan = plan_map[key]
        parsed = parse_table_response_checked(contents[key])
        expected_rows = plan.header_context_rows + (
            plan.logical_row_end - plan.logical_row_start
        )
        expected_columns = plan.stub_context_columns + (
            plan.logical_column_end - plan.logical_column_start
        )
        actual_rows = parsed.row_count
        actual_columns = parsed.column_count
        parsed, warnings = _soft_align_parsed(
            parsed,
            expected_rows,
            expected_columns,
            tile_ink_masks.get(key) if tile_ink_masks else None,
        )
        parsed_by_key[key] = parsed
        all_warnings.extend(f"{plan.file_name}：{warning}" for warning in warnings)
        tile_reports.append(
            {
                "tile": plan.file_name,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "warnings": warnings,
            }
        )

        body_top = plan.header_context_rows
        body_left = plan.stub_context_columns
        for item in parsed.placements:
            row1 = max(item.row, body_top)
            row2 = min(item.row + item.cell.rowspan, parsed.row_count)
            col1 = max(item.column, body_left)
            col2 = min(item.column + item.cell.colspan, parsed.column_count)
            if row1 >= row2 or col1 >= col2:
                continue
            global_row = plan.logical_row_start + row1 - body_top
            global_column = plan.logical_column_start + col1 - body_left
            cell = Cell(
                item.cell.text,
                item.cell.tag,
                rowspan=row2 - row1,
                colspan=col2 - col1,
            )
            for row in range(global_row, global_row + cell.rowspan):
                for column in range(global_column, global_column + cell.colspan):
                    if (row, column) in occupied:
                        raise HtmlTableMergeError(
                            f"逻辑坐标 ({row}, {column}) 被多个切片重复占用"
                        )
                    occupied.add((row, column))
            global_placements.append(Placement(cell, global_row, global_column))

    first = parsed_by_key[min(parsed_by_key)]
    last = parsed_by_key[max(parsed_by_key)]
    result = _render(
        global_placements,
        logical_row_count,
        logical_column_count,
        first.prefix,
        last.suffix,
    )
    report: dict[str, object] = {
        "logical_rows": logical_row_count,
        "logical_columns": logical_column_count,
        "covered_cells": len(occupied),
        "total_cells": logical_row_count * logical_column_count,
        "tiles": tile_reports,
        "warnings": all_warnings,
        "status": "warning" if all_warnings else "ok",
    }
    return result, report
