"""把失败的四列主体沿安全列线复切成两列主体，并始终保留五列行键。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


项目根目录 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(项目根目录))

from afac_pipeline.common.models import Box  # noqa: E402
from afac_pipeline.table import TableConfig, TablePipeline  # noqa: E402
from afac_pipeline.table.步骤006_逻辑网格切块 import plan_grid_tiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("正式清单", type=Path)
    parser.add_argument("输出目录", type=Path)
    args = parser.parse_args()

    source_manifest = json.loads(args.正式清单.read_text(encoding="utf-8"))
    image_path = Path(source_manifest["image"]["path"])
    args.输出目录.mkdir(parents=True, exist_ok=True)
    tile_dir = args.输出目录 / "03_API实际输入切块"
    tile_dir.mkdir(parents=True, exist_ok=True)
    config = TableConfig.from_json(项目根目录 / "afac_pipeline/table/config.example.json")
    pipeline = TablePipeline(config, args.输出目录 / "内部临时")

    records = []
    for region in source_manifest["regions"]:
        region_index = int(region["index"])
        rows = tuple(int(value) for value in region["row_boundaries"])
        columns = tuple(int(value) for value in region["column_boundaries"])
        safe_box = Box(columns[0], rows[0], columns[-1], rows[-1])
        plans = plan_grid_tiles(
            safe_box,
            region_index,
            rows,
            columns,
            config.max_vlm_side,
            1,
            0,
            140,
            80,
            config.max_tile_aspect_ratio,
        )
        for parent in plans:
            if parent.row_index not in {0, 1} or parent.column_index != 3:
                continue
            middle = (parent.logical_column_start + parent.logical_column_end) // 2
            logical_parts = (
                (parent.logical_column_start, middle, "左半"),
                (middle, parent.logical_column_end, "右半"),
            )
            for part_index, (column_start, column_end, part_name) in enumerate(logical_parts):
                body = Box(
                    columns[column_start],
                    parent.source_box.y1,
                    columns[column_end],
                    parent.source_box.y2,
                )
                context_width = columns[5] - columns[0]
                plan = replace(
                    parent,
                    column_index=part_index,
                    column_count=2,
                    source_box=body,
                    output_width=body.width + context_width,
                    logical_column_start=column_start,
                    logical_column_end=column_end,
                    stub_context_columns=5,
                )
                name = (
                    f"第{region_index + 1:03d}表_"
                    f"第{plan.row_index + 1:03d}行带_"
                    f"原第004列块_{part_name}.png"
                )
                output = tile_dir / name
                pipeline._save_tile(image_path, output, plan, rows, columns)
                records.append(
                    {
                        "图片名": image_path.name,
                        "表序号": region_index,
                        "行带序号": plan.row_index,
                        "列块序号": plan.column_index,
                        "行带总数": 2,
                        "列块总数": 2,
                        "主体责任框": plan.source_box.to_dict(),
                        "主体逻辑行数": plan.logical_row_end - plan.logical_row_start,
                        "主体逻辑列数": plan.logical_column_end - plan.logical_column_start,
                        "重复表头行数": plan.header_context_rows,
                        "重复行名列数": plan.stub_context_columns,
                        "API图片尺寸": [plan.output_width, plan.output_height],
                        "文件路径": str(output.resolve()),
                    }
                )

    result = {
        "路线": "失败主体沿安全列线二分；每个子块保留五列行键",
        "图片": str(image_path.resolve()),
        "总切块数": len(records),
        "切块": records,
    }
    (args.输出目录 / "安全线粗切清单.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[完成] 生成 {len(records)} 个安全线复切子块：{args.输出目录}")


if __name__ == "__main__":
    main()
