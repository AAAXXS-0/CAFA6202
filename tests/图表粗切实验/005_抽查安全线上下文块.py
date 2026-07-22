"""抽查四种安全线上下文块，严格空提示词调用官方 API。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
import json
from pathlib import Path
import sys
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(项目根目录))

from afac_pipeline.common.cache import ResultCache  # noqa: E402
from afac_pipeline.common.vlm_client import FinixDocClient  # noqa: E402
from afac_pipeline.table.步骤009_HTML表格软对齐 import (  # noqa: E402
    parse_table_response_checked,
)


抽查坐标 = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)}


def 渲染(cells: list[list[str]]) -> str:
    lines = ["<table>"]
    for row in cells:
        lines.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("安全线清单", type=Path)
    parser.add_argument("--并行数", type=int, default=2)
    parser.add_argument("--超时秒数", type=int, default=180)
    parser.add_argument("--重试次数", type=int, default=1)
    parser.add_argument("--全部", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.安全线清单.read_text(encoding="utf-8"))
    output_dir = args.安全线清单.parent
    raw_dir = output_dir / "04_API抽查原始响应"
    matrix_dir = output_dir / "05_API抽查矩阵"
    raw_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        tile for tile in manifest["切块"]
        if args.全部 or (tile["行带序号"], tile["列块序号"]) in 抽查坐标
    ]
    cache = ResultCache(output_dir / "API抽查缓存.sqlite3")
    client = FinixDocClient.from_official_doc(
        项目根目录 / "FinixDoc_VL调用.txt",
        user_id="finixB2002",
        timeout=args.超时秒数,
        max_retries=args.重试次数,
    )

    def recognize(tile: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(tile["文件路径"])
        key = cache.tile_key(image_path.read_bytes(), "", client.model + "@safe-context-v2")
        response = cache.get_tile(key)
        source = "缓存"
        if response is None:
            source = "API"
            label = (
                f"原图 {tile['图片名']} / 表{tile['表序号'] + 1} / "
                f"行带{tile['行带序号'] + 1} / 列块{tile['列块序号'] + 1}"
            )
            print(f"[安全线API抽查] {label}", flush=True)
            response = client.recognize(image_path, "", request_label=label)
            cache.put_tile(key, response, {"实验": "安全线上下文V2", "切块": tile})
        stem = image_path.stem
        (raw_dir / f"{stem}.md").write_text(response, encoding="utf-8")
        parsed = parse_table_response_checked(response)
        cells = [["" for _ in range(parsed.column_count)] for _ in range(parsed.row_count)]
        for placement in parsed.placements:
            cells[placement.row][placement.column] = placement.cell.text.strip()
        result = {
            "切块": tile,
            "来源": source,
            "响应字符数": len(response),
            "模型矩阵尺寸": [parsed.row_count, parsed.column_count],
            "期望主体尺寸": [tile["主体逻辑行数"], tile["主体逻辑列数"]],
            "重复上下文尺寸": [tile["重复表头行数"], tile["重复行名列数"]],
            "单元格": cells,
        }
        (matrix_dir / f"{stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (matrix_dir / f"{stem}.html").write_text(渲染(cells), encoding="utf-8")
        return result

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.并行数) as executor:
        futures = {executor.submit(recognize, tile): tile for tile in selected}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "切块": Path(tile["文件路径"]).name,
                        "错误类型": type(error).__name__,
                        "错误": str(error),
                    }
                )
    summary = {
        "抽查数": len(selected),
        "成功数": len(results),
        "失败数": len(failures),
        "结果": [
            {
                "切块": Path(item["切块"]["文件路径"]).name,
                "来源": item["来源"],
                "响应字符数": item["响应字符数"],
                "模型矩阵尺寸": item["模型矩阵尺寸"],
                "期望主体尺寸": item["期望主体尺寸"],
                "重复上下文尺寸": item["重复上下文尺寸"],
            }
            for item in sorted(
                results,
                key=lambda item: (
                    item["切块"]["行带序号"],
                    item["切块"]["列块序号"],
                ),
            )
        ],
        "失败": failures,
    }
    (output_dir / "API抽查汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[抽查完成] 成功 {len(results)}/{len(selected)}；汇总：{output_dir / 'API抽查汇总.json'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
