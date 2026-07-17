"""按逻辑行列坐标解析、校验并合并 FinixDoc-VL 的 HTML 表格。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
import re

from ..common.models import TilePlan
from .markdown_merge import MarkdownMergeError, parse_first_table


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
        if tag in {"th", "td"} and self.current_tag == tag and self.current_row is not None:
            try:
                rowspan = max(1, int(self.current_attrs.get("rowspan", "1")))
                colspan = max(1, int(self.current_attrs.get("colspan", "1")))
            except ValueError as error:
                raise HtmlTableMergeError("HTML 单元格的 rowspan/colspan 不是整数") from error
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
    parts = [part for part in (prefix.strip(), "\n".join(lines), suffix.strip()) if part]
    return "\n".join(parts)


def normalize_table_response(response: str) -> tuple[str, dict[str, int]]:
    parsed = parse_table_response(response)
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
        item.cell.rowspan != 1 or item.cell.colspan != 1
        for item in parsed.placements
    ):
        return None
    rows: list[list[Cell]] = [[] for _ in range(parsed.row_count)]
    for item in sorted(parsed.placements, key=lambda value: (value.row, value.column)):
        if item.column != len(rows[item.row]):
            return None
        rows[item.row].append(item.cell)
    return rows


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
    costs = [
        [infinity] * (actual_count + 1)
        for _ in range(expected_count + 1)
    ]
    previous: list[list[tuple[int, int, bool] | None]] = [
        [None] * (actual_count + 1)
        for _ in range(expected_count + 1)
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
    """按墨迹槽位补回模型省略的空行空列，模型文字始终优先保留。"""

    if (
        parsed.row_count == expected_rows
        and parsed.column_count == expected_columns
    ):
        return parsed, []
    warning = (
        f"模型返回 {parsed.row_count}×{parsed.column_count}，"
        f"预处理参考为 {expected_rows}×{expected_columns}"
    )
    rows = _simple_rows(parsed)
    if (
        rows is None
        or parsed.row_count > expected_rows
        or any(len(row) > expected_columns for row in rows)
    ):
        # 合并单元格或模型给出更多结构时不强行改写，只在全局边界内保留
        # 模型原始坐标，并把差异交给quality warning。
        return parsed, [warning + "；保留模型原始结构"]

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
        normalized_mask = [
            [True] * expected_columns
            for _ in range(expected_rows)
        ]
    expected_row_ink = [sum(row) for row in normalized_mask]
    actual_row_ink = [
        sum(bool(cell.text.strip()) for cell in row)
        for row in rows
    ]
    row_mapping = _sequence_mapping(
        expected_rows,
        len(rows),
        lambda expected, actual: abs(
            expected_row_ink[expected] - actual_row_ink[actual]
        ) / max(1, expected_row_ink[expected]),
        lambda expected: 0.0 if expected_row_ink[expected] == 0 else 2.0,
    )
    if row_mapping is None:
        return parsed, [warning + "；无法可靠对齐行，保留模型原始结构"]

    aligned: list[Placement] = []
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
            column_mapping = list(range(len(cells)))
        for actual_column, cell in enumerate(cells):
            aligned.append(
                Placement(cell, expected_row, column_mapping[actual_column])
            )
    return (
        ParsedResponse(
            parsed.prefix,
            parsed.suffix,

            aligned,
            expected_rows,
            expected_columns,
        ),
        [warning + "；已按单元格墨迹补齐空位"],
    )


def normalize_table_response_soft(
    response: str,
    expected_rows: int,
    expected_columns: int,
    ink_mask: list[list[bool]] | None = None,
) -> tuple[str, dict[str, object]]:
    """以模型内容为主，利用预处理墨迹参考补回被省略的空位。"""

    parsed = parse_table_response(response)
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
        {**actual, "warnings": warnings},
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
        parsed = parse_table_response(contents[key])
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
        all_warnings.extend(
            f"{plan.file_name}：{warning}"
            for warning in warnings
        )
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
