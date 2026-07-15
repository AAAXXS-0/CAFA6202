"""赛事 CSV 输出与长图/图表结果合并。"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Iterable


FIELD_NAMES = ["file_name", "ground_truth"]


def write_submission(
    results: dict[str, str],
    output_path: str | Path,
    file_order: Iterable[str] | None = None,
) -> None:
    """严格输出且仅输出 file_name、ground_truth 两列。"""

    names = list(file_order) if file_order is not None else sorted(results)
    if len(names) != len(set(names)):
        raise ValueError("CSV 输出顺序中存在重复 file_name")
    missing = set(names) - set(results)
    extra = set(results) - set(names)
    if missing or extra:
        raise ValueError(
            f"CSV 输出文件名不一致：缺少 {sorted(missing)}，多出 {sorted(extra)}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for file_name in names:
            writer.writerow({"file_name": file_name, "ground_truth": results[file_name]})


def _read_submission(path: str | Path) -> tuple[list[str], dict[str, str]]:
    """读取两列提交 CSV，并在文件内部拒绝重复图片名。"""

    # 超大表格的 HTML 可能超过 csv 模块默认 128KiB 单字段限制。赛题允许
    # ground_truth 保存完整文档，因此读取合并时必须接受进程可寻址的字段。
    csv.field_size_limit(sys.maxsize)
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELD_NAMES:
            raise ValueError(f"{source} 表头必须严格为 {FIELD_NAMES}，实际为 {reader.fieldnames}")
        order: list[str] = []
        results: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            file_name = row["file_name"]
            if not file_name:
                raise ValueError(f"{source}:{row_number} 的 file_name 为空")
            if file_name in results:
                raise ValueError(f"{source} 中 file_name 重复：{file_name}")
            order.append(file_name)
            results[file_name] = row["ground_truth"]
    return order, results


def combine_submissions(
    input_paths: Iterable[str | Path],
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """按官方模板顺序合并多个分支 CSV，并严格校验 100 张图片集合。"""

    template_order, _ = _read_submission(template_path)
    combined: dict[str, str] = {}
    for input_path in input_paths:
        _, branch_results = _read_submission(input_path)
        duplicates = set(combined) & set(branch_results)
        if duplicates:
            raise ValueError(f"分支 CSV 之间存在重复 file_name：{sorted(duplicates)}")
        combined.update(branch_results)

    template_names = set(template_order)
    missing = template_names - set(combined)
    extra = set(combined) - template_names
    if missing or extra:
        raise ValueError(
            f"合并结果与官方模板不一致：缺少 {sorted(missing)}，多出 {sorted(extra)}"
        )

    output = Path(output_path)
    write_submission(combined, output, template_order)
    return output
