"""表格切片 Markdown 的确定性合并。

合并只依赖相邻切片和表格矩阵，不调用其他大模型。若视觉模型输出的行列数
无法对应，代码会显式报错并保留原始响应，而不是静默拼出错误表格。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


class MarkdownMergeError(RuntimeError):
    pass


@dataclass
class ParsedTable:
    prefix: str
    header: list[str]
    rows: list[list[str]]
    suffix: str


def _cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def parse_first_table(markdown: str) -> ParsedTable:
    lines = markdown.strip().splitlines()
    start = None
    end = None
    for index in range(len(lines) - 1):
        if lines[index].lstrip().startswith("|") and _is_separator(lines[index + 1]):
            start = index
            end = index + 2
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            break
    if start is None or end is None:
        raise MarkdownMergeError("切片响应中没有找到标准 Markdown 表格")

    header = _cells(lines[start])
    rows = [_cells(line) for line in lines[start + 2 : end]]
    if any(len(row) != len(header) for row in rows):
        raise MarkdownMergeError("同一切片的表格列数不一致")
    return ParsedTable(
        prefix="\n".join(lines[:start]).strip(),
        header=header,
        rows=rows,
        suffix="\n".join(lines[end:]).strip(),
    )


def render_table(table: ParsedTable) -> str:
    lines: list[str] = []
    if table.prefix:
        lines.extend([table.prefix, ""])
    lines.append("| " + " | ".join(table.header) + " |")
    lines.append("| " + " | ".join("---" for _ in table.header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in table.rows)
    if table.suffix:
        lines.extend(["", table.suffix])
    return "\n".join(lines).strip()


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("|")


def _common_column_overlap(left: ParsedTable, right: ParsedTable) -> int:
    """查找左右切片因重叠像素而重复输出的末列/首列。"""

    if len(left.rows) != len(right.rows):
        return 0
    max_overlap = min(len(left.header), len(right.header))
    left_matrix = [left.header, *left.rows]
    right_matrix = [right.header, *right.rows]
    for overlap in range(max_overlap, 0, -1):
        if all(
            [_normalized(cell) for cell in left_row[-overlap:]]
            == [_normalized(cell) for cell in right_row[:overlap]]
            for left_row, right_row in zip(left_matrix, right_matrix)
        ):
            return overlap
    return 0


def merge_horizontal(left_markdown: str, right_markdown: str) -> str:
    left = parse_first_table(left_markdown)
    right = parse_first_table(right_markdown)
    if len(left.rows) != len(right.rows):
        raise MarkdownMergeError(
            f"横向切片行数不同：左侧 {len(left.rows)} 行，右侧 {len(right.rows)} 行"
        )
    overlap = _common_column_overlap(left, right)
    merged = ParsedTable(
        prefix=left.prefix or right.prefix,
        header=left.header + right.header[overlap:],
        rows=[
            left_row + right_row[overlap:]
            for left_row, right_row in zip(left.rows, right.rows)
        ],
        suffix=right.suffix or left.suffix,
    )
    return render_table(merged)


def _common_row_overlap(top: ParsedTable, bottom: ParsedTable) -> int:
    max_overlap = min(len(top.rows), len(bottom.rows), 10)
    for overlap in range(max_overlap, 0, -1):
        top_rows = [[_normalized(cell) for cell in row] for row in top.rows[-overlap:]]
        bottom_rows = [[_normalized(cell) for cell in row] for row in bottom.rows[:overlap]]
        if top_rows == bottom_rows:
            return overlap
    return 0


def merge_vertical(top_markdown: str, bottom_markdown: str) -> str:
    top = parse_first_table(top_markdown)
    bottom = parse_first_table(bottom_markdown)
    if len(top.header) != len(bottom.header):
        raise MarkdownMergeError(
            f"纵向切片列数不同：上块 {len(top.header)} 列，下块 {len(bottom.header)} 列"
        )
    overlap = _common_row_overlap(top, bottom)
    merged = ParsedTable(
        prefix=top.prefix or bottom.prefix,
        header=top.header,
        rows=top.rows + bottom.rows[overlap:],
        suffix=bottom.suffix or top.suffix,
    )
    return render_table(merged)


def merge_markdown_grid(contents: dict[tuple[int, int], str]) -> str:
    """先横向合并同一行切片，再按纵向合并切片行。"""

    if not contents:
        return ""
    row_indices = sorted({row for row, _ in contents})
    merged_rows: list[str] = []
    for row_index in row_indices:
        columns = sorted(column for row, column in contents if row == row_index)
        if columns != list(range(max(columns) + 1)):
            raise MarkdownMergeError(f"第 {row_index} 个切片行缺少列")
        current = contents[(row_index, columns[0])]
        for column_index in columns[1:]:
            current = merge_horizontal(current, contents[(row_index, column_index)])
        merged_rows.append(current)

    result = merged_rows[0]
    for current in merged_rows[1:]:
        result = merge_vertical(result, current)
    return result
