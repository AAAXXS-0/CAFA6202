"""离线拼接局部重叠抽查结果：横向找重叠列，上下按顺序补空追加。"""

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


def 规整(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    return [[*row, *([""] * (width - len(row)))] for row in rows]


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
    output_dir = args.实验目录 / "06_局部重叠拼接"
    output_dir.mkdir(parents=True, exist_ok=True)

    horizontal_results = []
    audits = []
    for row_index in (1, 2):
        first_path = matrix_dir / f"第001表_第{row_index:03d}行带_首块.json"
        neighbor_path = matrix_dir / f"第001表_第{row_index:03d}行带_相邻块.json"
        first_data = json.loads(first_path.read_text(encoding="utf-8"))
        neighbor_data = json.loads(neighbor_path.read_text(encoding="utf-8"))
        first = {"单元格": first_data["单元格"], "前置文字": "", "后置文字": ""}
        neighbor = {"单元格": neighbor_data["单元格"], "前置文字": "", "后置文字": ""}
        merged, audit = module.左右拼接(first, neighbor)
        audit["行带"] = row_index
        audits.append(audit)
        horizontal_results.append(merged["单元格"])
        (output_dir / f"第{row_index:03d}行带_横拼.json").write_text(
            json.dumps(
                {"单元格": merged["单元格"], "拼接审计": audit},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / f"第{row_index:03d}行带_横拼.html").write_text(
            渲染(merged["单元格"]), encoding="utf-8"
        )

    # 两个行带在原图中上下相邻、没有重复责任区，因此不做重叠删除。
    # 先各自补到共同列数，再顺序追加；模型漏列只表现为空白，不挪动数据。
    final_rows = 规整([*horizontal_results[0], *horizontal_results[1]])
    summary = {
        "行带尺寸": [
            [len(rows), max((len(row) for row in rows), default=0)]
            for rows in horizontal_results
        ],
        "最终尺寸": [
            len(final_rows),
            max((len(row) for row in final_rows), default=0),
        ],
        "横向拼接审计": audits,
    }
    (output_dir / "拼接汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "最终抽查结果.html").write_text(渲染(final_rows), encoding="utf-8")
    print(f"[完成] 最终尺寸 {summary['最终尺寸']}：{output_dir}")


if __name__ == "__main__":
    main()
