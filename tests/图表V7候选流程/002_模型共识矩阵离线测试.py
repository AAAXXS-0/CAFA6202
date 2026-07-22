"""使用已经落盘的模型原始响应，离线测试“模型优先R×C”拼接。

本脚本不会调用API或本地模型。它把正式预处理R×C降级为切块位置索引，
实际行列数由同一行带、同一列带的模型输出共同决定：

- 同一逻辑行带的左右切块采用最大的模型行数，少行的块在尾部补空；
- 同一逻辑列带的上下切块采用最大的模型列数，少列的块在右侧补空；
- 预处理R×C和墨迹bool只写入对照报告，不参与强制映射；
- 无法解析的响应单独报告，不让一块坏响应中断整张图的离线检查。

“尾部补空”只是第一版可视化基线，不声称已经找准缺失发生的位置。后续可
根据本轮中间产物再加入重叠行名/列名锚点。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import re
import sys
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from afac_pipeline.table.步骤009_HTML表格软对齐 import (
    ParsedResponse,
    parse_table_response_checked,
)


默认准备目录 = 项目根目录 / "work/正式运行/图表_c811361c0b5b"
默认输出目录 = 项目根目录 / "work/验证/图表V7模型共识矩阵"


def 写JSON(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def 模型矩阵(parsed: ParsedResponse) -> list[list[str]]:
    """展开成普通二维文字矩阵；合并格只在左上角保留一次文字。"""

    matrix = [[""] * parsed.column_count for _ in range(parsed.row_count)]
    for placement in parsed.placements:
        if 0 <= placement.row < parsed.row_count and 0 <= placement.column < parsed.column_count:
            current = matrix[placement.row][placement.column]
            text = placement.cell.text.strip()
            if current and text and text != current:
                matrix[placement.row][placement.column] = f"{current} | {text}"
            elif text:
                matrix[placement.row][placement.column] = text
    return matrix


def 矩阵HTML(matrix: list[list[str]], title: str) -> str:
    body = []
    for row in matrix:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body.append(f"<tr>{cells}</tr>")
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}"
        "td{border:1px solid #777;padding:3px;white-space:pre-wrap}"
        "th{position:sticky;top:0;background:#eee}</style>"
        f"<h1>{html.escape(title)}</h1><p>{len(matrix)}行×"
        f"{max((len(row) for row in matrix), default=0)}列</p>"
        f"<table>{''.join(body)}</table>"
    )


def 响应序号(path: Path) -> int:
    match = re.search(r"_attempt_(\d+)\.md$", path.name)
    return int(match.group(1)) if match else 0


def 选择最新可解析响应(raw_dir: Path, tile_name: str) -> tuple[Path | None, ParsedResponse | None, list[str]]:
    stem = Path(tile_name).stem
    candidates = sorted(
        raw_dir.glob(f"{stem}_attempt_*.md"),
        key=响应序号,
        reverse=True,
    )
    errors: list[str] = []
    for path in candidates:
        raw = path.read_text(encoding="utf-8")
        try:
            return path, parse_table_response_checked(raw), errors
        except Exception as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    return None, None, errors


def 预处理切块形状(plan: dict[str, Any]) -> tuple[int, int]:
    rows = int(plan.get("header_context_rows", 0)) + (
        int(plan["logical_row_end"]) - int(plan["logical_row_start"])
    )
    columns = int(plan.get("stub_context_columns", 0)) + (
        int(plan["logical_column_end"]) - int(plan["logical_column_start"])
    )
    return rows, columns


def 切块墨迹比例(mask: list[list[bool]], plan: dict[str, Any]) -> float | None:
    if not mask:
        return None
    row_indices = [
        *range(int(plan.get("header_context_rows", 0))),
        *range(int(plan["logical_row_start"]), int(plan["logical_row_end"])),
    ]
    column_indices = [
        *range(int(plan.get("stub_context_columns", 0))),
        *range(int(plan["logical_column_start"]), int(plan["logical_column_end"])),
    ]
    values = [
        bool(mask[row][column])
        for row in row_indices
        if 0 <= row < len(mask)
        for column in column_indices
        if 0 <= column < len(mask[row])
    ]
    return None if not values else sum(values) / len(values)


def 共识统计(values: list[int]) -> dict[str, object]:
    counter = Counter(values)
    if not counter:
        return {"values": [], "mode": 0, "maximum": 0, "selected": 0}
    best_frequency = max(counter.values())
    modes = sorted(value for value, count in counter.items() if count == best_frequency)
    mode = modes[-1]
    maximum = max(values)
    # 实验阶段不裁掉任何模型非空结构，因此采用最大值；众数同时落盘，
    # 方便判断最大值是否只是单块离群。
    return {
        "values": values,
        "counts": {str(key): value for key, value in sorted(counter.items())},
        "mode": mode,
        "mode_frequency": best_frequency,
        "maximum": maximum,
        "selected": maximum,
        "reason": "为避免丢失模型非空内容，第一版选择最大值；众数仅供审计",
    }


def 补齐矩阵(matrix: list[list[str]], rows: int, columns: int) -> list[list[str]]:
    result = []
    for row in matrix[:rows]:
        result.append([*row[:columns], *([""] * max(0, columns - len(row)))])
    while len(result) < rows:
        result.append([""] * columns)
    return result


def 放置矩阵(
    target: list[list[str]],
    source: list[list[str]],
    row_offset: int,
    column_offset: int,
    warnings: list[str],
    tile_name: str,
) -> None:
    for row, values in enumerate(source):
        for column, value in enumerate(values):
            if not value:
                continue
            y = row_offset + row
            x = column_offset + column
            current = target[y][x]
            if current and current != value:
                warnings.append(
                    f"{tile_name} 在全局({y},{x})与已有内容冲突：{current!r} / {value!r}"
                )
                target[y][x] = f"{current} | {value}"
            else:
                target[y][x] = value


def 处理区域(
    image_name: str,
    image_dir: Path,
    region: dict[str, Any],
    output_dir: Path,
) -> dict[str, object]:
    region_index = int(region["index"])
    region_dir = output_dir / f"第{region_index + 1:03d}表"
    raw_dir = image_dir / "responses" / "模型原始"
    region_dir.mkdir(parents=True, exist_ok=True)
    mask = region.get("cell_ink_mask") or []
    tiles: list[dict[str, Any]] = []
    for plan in region.get("tiles", []):
        if str(plan.get("tiling_mode", "")) != "logical_grid":
            continue
        response_path, parsed, parse_errors = 选择最新可解析响应(
            raw_dir,
            str(plan["file_name"]),
        )
        expected_rows, expected_columns = 预处理切块形状(plan)
        item: dict[str, Any] = {
            "tile": str(plan["file_name"]),
            "plan": plan,
            "row_band": [int(plan["logical_row_start"]), int(plan["logical_row_end"])],
            "column_band": [
                int(plan["logical_column_start"]),
                int(plan["logical_column_end"]),
            ],
            "preprocess_shape": [expected_rows, expected_columns],
            "ink_bool_true_ratio": 切块墨迹比例(mask, plan),
            "response": None if response_path is None else str(response_path.resolve()),
            "parse_errors": parse_errors,
        }
        if parsed is not None:
            matrix = 模型矩阵(parsed)
            item["model_shape"] = [parsed.row_count, parsed.column_count]
            item["matrix"] = matrix
            tile_stem = Path(str(plan["file_name"])).stem
            写JSON(region_dir / "001_模型原始矩阵" / f"{tile_stem}.json", item)
            html_path = region_dir / "001_模型原始矩阵" / f"{tile_stem}.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(
                矩阵HTML(matrix, f"{image_name} / {tile_stem} / 模型原始矩阵"),
                encoding="utf-8",
            )
        else:
            item["model_shape"] = None
            item["matrix"] = None
        tiles.append(item)

    parsed_tiles = [item for item in tiles if item["matrix"] is not None]
    row_groups: dict[tuple[int, int], list[int]] = {}
    column_groups: dict[tuple[int, int], list[int]] = {}
    for item in parsed_tiles:
        row_key = tuple(item["row_band"])
        column_key = tuple(item["column_band"])
        row_groups.setdefault(row_key, []).append(int(item["model_shape"][0]))
        column_groups.setdefault(column_key, []).append(int(item["model_shape"][1]))
    row_consensus = {key: 共识统计(values) for key, values in row_groups.items()}
    column_consensus = {key: 共识统计(values) for key, values in column_groups.items()}
    row_keys = sorted(row_consensus)
    column_keys = sorted(column_consensus)
    row_offsets: dict[tuple[int, int], int] = {}
    column_offsets: dict[tuple[int, int], int] = {}
    total_rows = 0
    for key in row_keys:
        row_offsets[key] = total_rows
        total_rows += int(row_consensus[key]["selected"])
    total_columns = 0
    for key in column_keys:
        column_offsets[key] = total_columns
        total_columns += int(column_consensus[key]["selected"])
    merged = [[""] * total_columns for _ in range(total_rows)]
    warnings: list[str] = []
    for item in parsed_tiles:
        row_key = tuple(item["row_band"])
        column_key = tuple(item["column_band"])
        target_rows = int(row_consensus[row_key]["selected"])
        target_columns = int(column_consensus[column_key]["selected"])
        padded = 补齐矩阵(item["matrix"], target_rows, target_columns)
        item["consensus_shape"] = [target_rows, target_columns]
        item["padded_matrix"] = padded
        tile_stem = Path(str(item["tile"])).stem
        写JSON(
            region_dir / "003_共识补齐矩阵" / f"{tile_stem}.json",
            {
                key: value
                for key, value in item.items()
                if key not in {"matrix", "padded_matrix"}
            }
            | {"padded_matrix": padded},
        )
        html_path = region_dir / "003_共识补齐矩阵" / f"{tile_stem}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            矩阵HTML(padded, f"{image_name} / {tile_stem} / 共识补齐矩阵"),
            encoding="utf-8",
        )
        放置矩阵(
            merged,
            padded,
            row_offsets[row_key],
            column_offsets[column_key],
            warnings,
            str(item["tile"]),
        )

    preprocess_rows = max(0, len(region.get("row_boundaries", [])) - 1)
    preprocess_columns = max(0, len(region.get("column_boundaries", [])) - 1)
    shape_rows = []
    for item in tiles:
        model = item.get("model_shape")
        shape_rows.append(
            {
                "切块": item["tile"],
                "预处理行": item["preprocess_shape"][0],
                "预处理列": item["preprocess_shape"][1],
                "模型行": "" if model is None else model[0],
                "模型列": "" if model is None else model[1],
                "墨迹bool真值比例": item["ink_bool_true_ratio"],
                "响应文件": item["response"] or "",
                "解析错误": "；".join(item["parse_errors"]),
            }
        )
    csv_path = region_dir / "002_逐块尺寸对比.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(shape_rows[0]) if shape_rows else ["切块"])
        writer.writeheader()
        writer.writerows(shape_rows)
    consensus_report = {
        "image": image_name,
        "region_index": region_index,
        "preprocess_region_shape": [preprocess_rows, preprocess_columns],
        "model_consensus_shape": [total_rows, total_columns],
        "row_band_consensus": {
            f"{key[0]}:{key[1]}": value for key, value in row_consensus.items()
        },
        "column_band_consensus": {
            f"{key[0]}:{key[1]}": value for key, value in column_consensus.items()
        },
        "parsed_tile_count": len(parsed_tiles),
        "missing_or_bad_tile_count": len(tiles) - len(parsed_tiles),
        "warnings": warnings,
        "note": "预处理R×C和墨迹bool仅作对照；共识矩阵不按它们强制裁剪",
    }
    写JSON(region_dir / "004_模型共识判定.json", consensus_report)
    写JSON(region_dir / "005_模型优先拼接矩阵.json", merged)
    (region_dir / "005_模型优先拼接矩阵.html").write_text(
        矩阵HTML(merged, f"{image_name} / 第{region_index + 1}表 / 模型优先拼接"),
        encoding="utf-8",
    )
    return consensus_report


def 处理单图(item: dict[str, Any], output_root: Path) -> dict[str, object]:
    manifest_path = Path(str(item["image_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_name = str(item["file_name"])
    image_dir = manifest_path.parent
    output_dir = output_root / Path(image_name).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    region_reports = [
        处理区域(image_name, image_dir, region, output_dir)
        for region in manifest.get("regions", [])
    ]
    report = {
        "image": image_name,
        "manifest": str(manifest_path.resolve()),
        "regions": region_reports,
    }
    写JSON(output_dir / "000_单图汇总.json", report)
    return report


def 生成总览(output_root: Path, reports: list[dict[str, object]]) -> None:
    rows = []
    for report in reports:
        image_name = str(report["image"])
        folder = Path(image_name).stem
        for region in report["regions"]:
            index = int(region["region_index"])
            before = region["preprocess_region_shape"]
            after = region["model_consensus_shape"]
            rows.append(
                "<tr>"
                f"<td>{html.escape(image_name)}</td><td>{index + 1}</td>"
                f"<td>{before[0]}×{before[1]}</td><td>{after[0]}×{after[1]}</td>"
                f"<td>{region['parsed_tile_count']}</td>"
                f"<td>{region['missing_or_bad_tile_count']}</td>"
                f"<td><a href='{html.escape(folder)}/第{index + 1:03d}表/"
                "005_模型优先拼接矩阵.html'>打开</a></td></tr>"
            )
    page = """<!doctype html><meta charset='utf-8'><title>模型共识R×C离线测试</title>
<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{border:1px solid #777;padding:6px}</style>
<h1>模型共识R×C离线测试</h1><p>本页不代表最终算法，只用于比较模型矩阵与预处理R×C。</p>
<table><tr><th>图片</th><th>表</th><th>预处理R×C</th><th>模型共识</th><th>可解析块</th><th>坏块</th><th>结果</th></tr>"""
    page += "".join(rows) + "</table>"
    (output_root / "000_总览.html").write_text(page, encoding="utf-8")


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线使用已有模型原始响应，生成模型优先R×C共识拼接中间产物。"
    )
    parser.add_argument("--prepare-work", type=Path, default=默认准备目录)
    parser.add_argument("--output-dir", type=Path, default=默认输出目录)
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="只处理文件名包含这些片段的图片，可重复填写",
    )
    return parser.parse_args()


def main() -> int:
    args = 解析参数()
    dataset_path = args.prepare_work / "dataset_manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = list(dataset.get("items", []))
    if args.name_contains:
        items = [
            item
            for item in items
            if any(value in str(item.get("file_name", "")) for value in args.name_contains)
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, item in enumerate(items, start=1):
        print(
            f"[模型共识离线测试 {index}/{len(items)}] {item.get('file_name')}",
            flush=True,
        )
        try:
            reports.append(处理单图(item, args.output_dir))
        except Exception as error:
            failure = {
                "image": item.get("file_name"),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            写JSON(
                args.output_dir / Path(str(item.get("file_name", "未知"))).stem / "999_测试失败.json",
                failure,
            )
    写JSON(args.output_dir / "000_汇总.json", reports)
    生成总览(args.output_dir, reports)
    print(f"[完成] 离线中间产物：{args.output_dir.resolve()}")
    print(f"[总览] {(args.output_dir / '000_总览.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
