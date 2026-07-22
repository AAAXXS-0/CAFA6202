"""半窗重叠专用拼接：允许重叠锚点后仍保留左块的少量语义列。"""

from __future__ import annotations

import argparse
from html import escape
import importlib.util
import json
from pathlib import Path
from typing import Any


def 加载粗切模块() -> Any:
    path = Path(__file__).with_name("001_粗切块与模型拼接.py")
    spec = importlib.util.spec_from_file_location("粗切拼接实现", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载粗切拼接实现")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def 尺寸(matrix: dict[str, Any]) -> tuple[int, int]:
    rows = matrix["单元格"]
    return len(rows), max((len(row) for row in rows), default=0)


def 补宽(row: list[str], width: int) -> list[str]:
    return [*row, *([""] * (width - len(row)))]


def 半窗横拼(
    module: Any,
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """先用通用拼接；仅在证据明确时接受“左块内部的一列锚点”。"""

    ordinary, audit = module.左右拼接(left, right)
    column_audit = audit.get("列序列对齐", {})
    average = float(column_audit.get("平均列相似度", 0.0))
    strong = int(column_audit.get("强匹配列数", 0))
    unmatched_left = int(column_audit.get("左侧末端未匹配列数", 999))
    right_end = int(column_audit.get("右侧重叠结束位置", -1))
    if not (
        column_audit.get("方式") == "列序列证据不足"
        and strong >= 1
        and average >= 0.90
        and unmatched_left <= 2
        and right_end == 0
    ):
        return ordinary, audit

    left_rows = left["单元格"]
    right_rows = right["单元格"]
    alignment, row_audit = module.对齐(left_rows, right_rows)
    left_width = max((len(row) for row in left_rows), default=0)
    right_width = max((len(row) for row in right_rows), default=0)
    drop_right_columns = right_end + 1
    merged_rows = []
    for left_index, right_index in alignment:
        left_row = (
            补宽(left_rows[left_index], left_width)
            if left_index is not None
            else [""] * left_width
        )
        right_row = (
            补宽(right_rows[right_index], right_width)
            if right_index is not None
            else [""] * right_width
        )
        merged_rows.append([*left_row, *right_row[drop_right_columns:]])
    result = {
        "单元格": merged_rows,
        "前置文字": left.get("前置文字", "") or right.get("前置文字", ""),
        "后置文字": right.get("后置文字", "") or left.get("后置文字", ""),
    }
    improved_audit = {
        **audit,
        "输出尺寸": 尺寸(result),
        "行对齐": row_audit,
        "列序列对齐": {
            **column_audit,
            "方式": "半窗局部内锚点",
            "删除右块重复前缀列数": drop_right_columns,
            "保留左块锚点后语义列数": unmatched_left,
        },
        "补空行数": sum(
            1 for left_index, right_index in alignment
            if left_index is None or right_index is None
        ),
    }
    return result, improved_audit


def 规整(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    return [补宽(row, width) for row in rows]


def 渲染(rows: list[list[str]]) -> str:
    lines = ["<table>"]
    for row in 规整(rows):
        lines.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("实验目录", type=Path)
    args = parser.parse_args()
    module = 加载粗切模块()
    matrix_dir = args.实验目录 / "05_API抽查矩阵"
    output_dir = args.实验目录 / "07_半窗重叠拼接"
    output_dir.mkdir(parents=True, exist_ok=True)

    row_bands = []
    audits = []
    for row_index in (1, 2):
        first = json.loads(
            (matrix_dir / f"第001表_第{row_index:03d}行带_首块.json").read_text(
                encoding="utf-8"
            )
        )
        neighbor = json.loads(
            (matrix_dir / f"第001表_第{row_index:03d}行带_相邻块.json").read_text(
                encoding="utf-8"
            )
        )
        merged, audit = 半窗横拼(
            module,
            {"单元格": first["单元格"]},
            {"单元格": neighbor["单元格"]},
        )
        audit["行带"] = row_index
        audits.append(audit)
        row_bands.append(merged["单元格"])
        (output_dir / f"第{row_index:03d}行带_横拼.html").write_text(
            渲染(merged["单元格"]), encoding="utf-8"
        )

    final_rows = 规整([*row_bands[0], *row_bands[1]])
    summary = {
        "行带尺寸": [
            [len(rows), max((len(row) for row in rows), default=0)]
            for rows in row_bands
        ],
        "最终尺寸": [len(final_rows), max((len(row) for row in final_rows), default=0)],
        "横向拼接审计": audits,
    }
    (output_dir / "拼接汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "最终抽查结果.html").write_text(渲染(final_rows), encoding="utf-8")
    print(f"[完成] 半窗重叠最终尺寸 {summary['最终尺寸']}：{output_dir}")


if __name__ == "__main__":
    main()
