"""根据已有安全线，重新生成带首行、首列上下文的粗切块。

本实验不重新判断单元格有没有墨迹，也不生成 R×C 墨迹 bool 矩阵。
行列边界只承担两个职责：避免从文字中间下刀，以及粗略控制每块内容量。
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw


项目根目录 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(项目根目录))

from afac_pipeline.common.models import Box  # noqa: E402
from afac_pipeline.table import TableConfig, TablePipeline  # noqa: E402
from afac_pipeline.table.步骤006_逻辑网格切块 import plan_grid_tiles  # noqa: E402


def 写JSON(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def 生成总览(
    image_path: Path,
    output_dir: Path,
    records: list[dict[str, object]],
) -> None:
    """保存总览叠加图和可逐块点开的 HTML；总览缩放不影响 API 图片。"""

    with Image.open(image_path) as source:
        preview = source.convert("RGB")
        original_width, original_height = source.size
    preview.thumbnail((1800, 1800))
    scale_x = preview.width / original_width
    scale_y = preview.height / original_height
    draw = ImageDraw.Draw(preview, "RGBA")
    for record in records:
        box = Box.from_dict(record["主体责任框"])
        coordinates = (
            round(box.x1 * scale_x),
            round(box.y1 * scale_y),
            round(box.x2 * scale_x),
            round(box.y2 * scale_y),
        )
        draw.rectangle(coordinates, outline=(0, 150, 0, 220), width=2)
    preview.save(output_dir / "01_安全线粗切总览.png")

    cards = []
    for record in records:
        relative = Path(record["文件路径"]).relative_to(output_dir.resolve()).as_posix()
        cards.append(
            "<figure>"
            f'<a href="{escape(relative)}"><img src="{escape(relative)}"></a>'
            f"<figcaption>{escape(Path(relative).stem)}<br>"
            f"主体逻辑格：{record['主体逻辑行数']}×{record['主体逻辑列数']}；"
            f"重复表头：{record['重复表头行数']}；"
            f"重复行名列：{record['重复行名列数']}</figcaption></figure>"
        )
    (output_dir / "02_逐块检查.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>安全线粗块检查</title>
<style>body{font-family:sans-serif;margin:20px}.grid{display:grid;
grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
figure{margin:0;border:1px solid #bbb;padding:8px}img{width:100%;height:220px;
object-fit:contain;background:#eee}figcaption{font-size:13px;line-height:1.6}</style>
<h1>带首行、首列上下文的安全线粗切块</h1>
<p>绿框只表示每块真正负责的主体；API 图片保持原始像素，不缩放、不拉伸。</p>
<div class="grid">"""
        + "".join(cards)
        + "</div>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("正式清单", type=Path, help="已有正式预处理的 manifest.json")
    parser.add_argument("输出目录", type=Path)
    parser.add_argument("--每块最多逻辑格", type=int, default=140)
    parser.add_argument("--重复表头行数", type=int, default=1)
    parser.add_argument("--重复行名列数", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.正式清单.read_text(encoding="utf-8"))
    image_path = Path(manifest["image"]["path"])
    args.输出目录.mkdir(parents=True, exist_ok=True)
    tile_dir = args.输出目录 / "03_API实际输入切块"
    tile_dir.mkdir(parents=True, exist_ok=True)

    config = TableConfig.from_json(项目根目录 / "afac_pipeline/table/config.example.json")
    pipeline = TablePipeline(config, args.输出目录 / "内部临时")
    records: list[dict[str, object]] = []
    region_summaries = []
    for region in manifest["regions"]:
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
            args.重复表头行数,
            args.重复行名列数,
            args.每块最多逻辑格,
            min(80, args.每块最多逻辑格),
            config.max_tile_aspect_ratio,
        )
        if not plans:
            raise RuntimeError(f"第{region_index + 1}张表无法沿安全线规划粗切块")

        for plan in plans:
            name = (
                f"第{region_index + 1:03d}表_"
                f"第{plan.row_index + 1:03d}行带_"
                f"第{plan.column_index + 1:03d}列块.png"
            )
            output = tile_dir / name
            if not output.is_file():
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
                    "主体逻辑行起点": plan.logical_row_start,
                    "主体逻辑行终点": plan.logical_row_end,
                    "主体逻辑列起点": plan.logical_column_start,
                    "主体逻辑列终点": plan.logical_column_end,
                    "主体逻辑行数": plan.logical_row_end - plan.logical_row_start,
                    "主体逻辑列数": plan.logical_column_end - plan.logical_column_start,
                    "重复表头行数": plan.header_context_rows,
                    "重复行名列数": plan.stub_context_columns,
                    "API图片尺寸": [plan.output_width, plan.output_height],
                    "文件路径": str(output.resolve()),
                }
            )
        region_summaries.append(
            {
                "表序号": region_index,
                "安全线来源": region.get("grid_source", "unknown"),
                "检测行段数": len(rows) - 1,
                "检测列段数": len(columns) - 1,
                "粗切块数": len(plans),
            }
        )

    result = {
        "路线": "安全线只负责下刀；模型矩阵负责最终内容",
        "来源正式清单": str(args.正式清单.resolve()),
        "图片": str(image_path.resolve()),
        "每块最多逻辑格": args.每块最多逻辑格,
        "重复表头行数": args.重复表头行数,
        "重复行名列数": args.重复行名列数,
        "总切块数": len(records),
        "分表汇总": region_summaries,
        "切块": records,
    }
    写JSON(args.输出目录 / "安全线粗切清单.json", result)
    生成总览(image_path, args.输出目录, records)
    print(f"[完成] 共生成 {len(records)} 个安全线粗块：{args.输出目录}")


if __name__ == "__main__":
    main()
