"""生成连续矩形粗块：首块含左侧分类列，后块只重复一列局部数值。"""

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
        base_plans = plan_grid_tiles(
            safe_box,
            region_index,
            rows,
            columns,
            config.max_vlm_side,
            0,
            0,
            140,
            80,
            config.max_tile_aspect_ratio,
        )
        row_plans = {
            plan.row_index: plan
            for plan in base_plans
            if plan.column_index == 0 and plan.row_index in {0, 1}
        }
        # 第一块连续包含左分类区和第一组数值；第二块向左重叠一列数值。
        column_parts = ((0, 8, "首块"), (4, 12, "相邻块"))
        for row_index, row_plan in row_plans.items():
            for column_index, (column_start, column_end, part_name) in enumerate(column_parts):
                body = Box(
                    columns[column_start],
                    row_plan.source_box.y1,
                    columns[column_end],
                    row_plan.source_box.y2,
                )
                plan = replace(
                    row_plan,
                    column_index=column_index,
                    column_count=len(column_parts),
                    source_box=body,
                    output_width=body.width,
                    logical_column_start=column_start,
                    logical_column_end=column_end,
                )
                name = (
                    f"第{region_index + 1:03d}表_"
                    f"第{row_index + 1:03d}行带_{part_name}.png"
                )
                output = tile_dir / name
                pipeline._save_tile(image_path, output, plan, rows, columns)
                records.append(
                    {
                        "图片名": image_path.name,
                        "表序号": region_index,
                        "行带序号": row_index,
                        "列块序号": column_index,
                        "行带总数": 2,
                        "列块总数": 2,
                        "主体责任框": body.to_dict(),
                        "主体逻辑行数": plan.logical_row_end - plan.logical_row_start,
                        "主体逻辑列数": column_end - column_start,
                        "重复表头行数": 0,
                        "重复行名列数": 0,
                        "局部重叠逻辑列数": 0 if column_index == 0 else 4,
                        "API图片尺寸": [body.width, body.height],
                        "文件路径": str(output.resolve()),
                    }
                )

    result = {
        "路线": "连续矩形 + 相邻块一列局部重叠",
        "图片": str(image_path.resolve()),
        "总切块数": len(records),
        "切块": records,
    }
    (args.输出目录 / "安全线粗切清单.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[完成] 生成 {len(records)} 个局部重叠对照块：{args.输出目录}")


if __name__ == "__main__":
    main()
