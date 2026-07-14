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


def merge_logical_tiles(
    contents: dict[tuple[int, int], str],
    plans: list[TilePlan],
    logical_row_count: int,
    logical_column_count: int,
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
    for key in sorted(contents):
        plan = plan_map[key]
        parsed = parse_table_response(contents[key])
        parsed_by_key[key] = parsed
        expected_rows = plan.header_context_rows + (
            plan.logical_row_end - plan.logical_row_start
        )
        expected_columns = plan.stub_context_columns + (
            plan.logical_column_end - plan.logical_column_start
        )
        tile_reports.append(
            {
                "tile": plan.file_name,
                "expected_rows": expected_rows,
                "actual_rows": parsed.row_count,
                "expected_columns": expected_columns,
                "actual_columns": parsed.column_count,
            }
        )
        if parsed.row_count != expected_rows or parsed.column_count != expected_columns:
            raise HtmlTableMergeError(
                f"{plan.file_name} 行列数不符：期望 {expected_rows}×{expected_columns}，"
                f"实际 {parsed.row_count}×{parsed.column_count}"
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
    }
    return result, report
