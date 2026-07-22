"""生成带五列行键的相邻数值块，验证年龄列能否稳定对齐。"""

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


抽查坐标 = {(0, 2), (0, 3), (1, 2), (1, 3)}


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
        for original_plan in plans:
            if (original_plan.row_index, original_plan.column_index) not in 抽查坐标:
                continue
            context_width = columns[5] - columns[0]
            plan = replace(
                original_plan,
                stub_context_columns=5,
                output_width=original_plan.source_box.width + context_width,
            )
            name = (
                f"第{region_index + 1:03d}表_"
                f"第{plan.row_index + 1:03d}行带_"
                f"第{plan.column_index + 1:03d}列块.png"
            )
            output = tile_dir / name
            pipeline._save_tile(image_path, output, plan, rows, columns)
            records.append(
                {
                    "图片名": image_path.name,
                    "表序号": region_index,
                    "行带序号": plan.row_index,
                    "列块序号": plan.column_index,
                    "行带总数": plan.row_count,
                    "列块总数": plan.column_count,
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
        "路线": "左四列分类字段 + 年龄行键 + 相邻数值主体",
        "图片": str(image_path.resolve()),
        "总切块数": len(records),
        "切块": records,
    }
    (args.输出目录 / "安全线粗切清单.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[完成] 生成 {len(records)} 个五列行键相邻块：{args.输出目录}")


if __name__ == "__main__":
    main()
