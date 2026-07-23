"""预处理逐图断点、中文中间产物索引和批量汇总。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


def _写入_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _建立硬链接(source: Path, destination: Path) -> None:
    """优先用硬链接整理中文视图，不重复占用大图片空间。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if source.samefile(destination):
                return
        except OSError:
            pass
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        # 少数跨文件系统环境不支持硬链接；退回普通复制保证可查看。
        shutil.copy2(source, destination)


def _图表中文名(relative: Path) -> Path | None:
    text = relative.as_posix()
    fixed = {
        "preview.png": "001_整图分析缩略图.png",
        "density_detection/density.png": "002_横向分表密度图.png",
        "density_detection/split_and_analysis_boxes.png": "003_分表及分析框.png",
        "preview_detected.png": "004_分表结果总览.png",
        "tile_overlay.png": "900_识别切块位置总览.png",
        "tile_contact_sheet.jpg": "901_识别切块缩略汇总.jpg",
        "manifest.json": "990_单图预处理清单.json",
        "预处理致命错误.json": "999_预处理致命错误.json",
    }
    if text in fixed:
        return Path(fixed[text])

    match = re.fullmatch(r"density_detection/第(\d+)表_分析框/(.*)", text)
    if match:
        region = int(match.group(1))
        return Path("分表分析框") / f"第{region:03d}表_{match.group(2)}"

    match = re.fullmatch(r"grid_analysis/region_(\d+)(.*)", text)
    if match:
        region = int(match.group(1)) + 1
        suffix = match.group(2)
        labels = {
            ".png": "01_统一50分析图.png",
            "_body_windows.png": "02_表体滑窗选择.png",
            "_body_selected.png": "03_选中的表格体.png",
            "_black_candidates.png": "04_黑线候选与采用结果.png",
            "_black_cleanup.png": "05_黑线清理结果.png",
            "_white_column_cleanup.png": "06_列白缝候选与采用结果.png",
            "_boundaries.png": "07_最终行列边界.png",
            "_diagnostics.json": "08_结构检测诊断.json",
            "_cell_ink_mask.json": "09_单元格墨迹矩阵.json",
            "_01_灰度二值图.png": "10_灰度二值图.png",
            "_02_行白缝_擦除竖黑线.png": "11_行白缝_擦除竖黑线.png",
            "_03_行白缝_左右晕染.png": "12_行白缝_左右晕染.png",
            "_04_列白缝_擦除全部黑线.png": "13_列白缝_擦除全部黑线.png",
            "_05_列白缝_二维自适应晕染.png": "14_列白缝_二维自适应晕染.png",
        }
        label = labels.get(suffix)
        if label:
            return Path("各表结构分析") / f"第{region:03d}表_{label}"

    match = re.fullmatch(r"tiles/region_(\d+)_r(\d+)_c(\d+)\.(\w+)", text)
    if match:
        region, row, column = (int(match.group(i)) + 1 for i in range(1, 4))
        extension = match.group(4)
        return (
            Path("识别切块")
            / f"第{region:03d}表_第{row:03d}行块_第{column:03d}列块.{extension}"
        )

    match = re.fullmatch(r"top_context/region_(\d+)(_content)?\.(\w+)", text)
    if match:
        region = int(match.group(1)) + 1
        label = "标题有效区域" if match.group(2) else "表顶标题候选"
        return Path("表顶标题") / f"第{region:03d}表_{label}.{match.group(3)}"
    return None


def _长图中文名(relative: Path, sequence: int) -> Path | None:
    text = relative.as_posix()
    fixed = {
        "manifest.json": "990_单图预处理清单.json",
        "预处理致命错误.json": "999_预处理致命错误.json",
    }
    if text in fixed:
        return Path(fixed[text])

    match = re.fullmatch(r"detection_windows/window_(\d+).*\.(\w+)", text)
    if match:
        return Path("滑窗切片") / f"第{int(match.group(1)) + 1:03d}窗.{match.group(2)}"

    match = re.fullmatch(r"vlm_requests/request_(\d+).*\.(\w+)", text)
    if match:
        return Path("大模型输入切块") / (
            f"第{int(match.group(1)) + 1:03d}块.{match.group(2)}"
        )

    match = re.fullmatch(r"vlm_request_parts/request_(\d+)_body\.(\w+)", text)
    if match:
        return Path("切块拼接零件") / (
            f"第{int(match.group(1)) + 1:03d}块_正文主体.{match.group(2)}"
        )

    match = re.fullmatch(r"vlm_request_parts/context_y(\d+)_(\d+)\.(\w+)", text)
    if match:
        return Path("切块拼接零件") / (
            f"目录或标题头_纵坐标{match.group(1)}至{match.group(2)}.{match.group(3)}"
        )

    if text.startswith("semantic_audit/"):
        return Path("标题语义分析") / f"第{sequence:04d}项{relative.suffix}"

    match = re.fullmatch(r"yolo_raw/批次(\d+)/image(\d+)\.(\w+)", text)
    if match:
        return Path("YOLO原始标框") / (
            f"第{int(match.group(1)) + 1:03d}批_第{int(match.group(2)) + 1:03d}张."
            f"{match.group(3)}"
        )
    if text == "yolo_raw/predictions.json":
        return Path("YOLO原始标框") / "检测框原始数据.json"
    return None


def 建立中文中间产物(
    image_dir: str | Path,
    branch: str,
    *,
    original_image: str | Path | None = None,
) -> Path:
    """把内部英文文件整理成中文硬链接视图，原流程引用不受影响。"""

    root = Path(image_dir)
    output = root / "中文中间产物"
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    if original_image is not None:
        source = Path(original_image)
        if source.is_file():
            destination = output / f"000_原始图片{source.suffix.lower()}"
            _建立硬链接(source, destination)
            records.append(
                {
                    "中文文件": str(destination.relative_to(output)),
                    "内部来源": str(source.resolve()),
                }
            )

    sources = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and output not in path.parents
        and "responses" not in path.parts
    )
    sequence = 0
    for source in sources:
        relative = source.relative_to(root)
        sequence += 1
        chinese = (
            _图表中文名(relative)
            if branch == "图表"
            else _长图中文名(relative, sequence)
        )
        if chinese is None:
            continue
        destination = output / chinese
        _建立硬链接(source, destination)
        records.append(
            {
                "中文文件": chinese.as_posix(),
                "内部来源": relative.as_posix(),
            }
        )

    index_path = output / "000_中间产物说明.json"
    _写入_json(
        index_path,
        {
            "说明": (
                "这里是便于人工检查的中文硬链接视图；"
                "删除本目录不会删除内部正式产物。"
            ),
            "分支": branch,
            "产物数量": len(records),
            "文件映射": records,
        },
    )
    return output


def 写入预处理检查点(
    *,
    work_dir: str | Path,
    branch: str,
    input_dir: str | Path,
    config: dict[str, Any],
    config_digest: str,
    image_count: int,
    unique_image_count: int,
    duplicate_reuse_count: int,
    processed_unique_count: int,
    items: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    complete: bool,
) -> Path:
    """每完成一张就原子写清单，进程中断后仍可恢复和追查。"""

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    if complete:
        status = "完成但有错误" if failures else "全部完成"
    else:
        status = "运行中"

    error_report = work / "预处理错误汇总.json"
    _写入_json(
        error_report,
        {
            "状态": status,
            "更新时间": now,
            "分支": branch,
            "失败图片数": sum(len(item.get("file_names", [])) for item in failures),
            "失败组数": len(failures),
            "失败详情": failures,
            # 保留旧测试和外部脚本使用的字段名。
            "failure_count": len(failures),
            "failed_image_count": sum(
                len(item.get("file_names", [])) for item in failures
            ),
            "failures": failures,
        },
    )

    summary_rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value["file_name"]):
        manifest_path = Path(item["image_manifest"])
        summary_rows.append(
            {
                "图片名": item["file_name"],
                "状态": "成功",
                "处理来源": (
                    "复用逐图缓存"
                    if item.get("preprocessing_cache") == "reused"
                    else "本轮新生成"
                ),
                "单图清单": str(manifest_path.resolve()),
                "中文中间产物": str(
                    (manifest_path.parent / "中文中间产物").resolve()
                ),
                "错误类型": "",
                "错误原因": "",
            }
        )
    for failure in failures:
        file_names = failure.get("file_names") or [
            failure.get("canonical_file_name", "未知图片")
        ]
        for file_name in file_names:
            summary_rows.append(
                {
                    "图片名": file_name,
                    "状态": "fatal",
                    "处理来源": "本轮失败",
                    "单图清单": "",
                    "中文中间产物": failure.get("chinese_artifacts", ""),
                    "错误类型": failure.get("error_type", "Error"),
                    "错误原因": failure.get("error", "未知错误"),
                }
            )
    summary_rows.sort(key=lambda row: row["图片名"])

    summary_csv = work / "预处理进度与错误汇总.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "图片名",
                "状态",
                "处理来源",
                "单图清单",
                "中文中间产物",
                "错误类型",
                "错误原因",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = work / "预处理进度与错误汇总.json"
    reused_count = sum(
        1 for item in items if item.get("preprocessing_cache") == "reused"
    )
    failed_image_count = sum(
        len(failure.get("file_names", [])) for failure in failures
    )
    processed_image_count = len(items) + failed_image_count
    pending_image_count = max(0, image_count - processed_image_count)
    _写入_json(
        summary_json,
        {
            "状态": status,
            "更新时间": now,
            "分支": branch,
            "总图片数": image_count,
            "已检查唯一图片数": processed_unique_count,
            "唯一图片总数": unique_image_count,
            "成功图片数": len(items),
            "失败图片数": failed_image_count,
            "尚未检查图片数": pending_image_count,
            "本轮复用缓存图片数": reused_count,
            "本轮新生成图片数": len(items) - reused_count,
            "明细": summary_rows,
        },
    )

    summary_html = work / "预处理进度与错误汇总.html"
    html_rows: list[str] = []
    for row in summary_rows:
        color = "#e8f5e9" if row["状态"] == "成功" else "#ffebee"
        artifact = row["中文中间产物"]
        artifact_cell = html.escape(artifact)
        if artifact:
            try:
                relative = Path(artifact).resolve().relative_to(work.resolve())
                artifact_cell = (
                    f'<a href="{html.escape(relative.as_posix())}">打开中文中间产物</a>'
                )
            except ValueError:
                pass
        html_rows.append(
            "<tr style=\"background:"
            + color
            + "\"><td>"
            + "</td><td>".join(
                [
                    html.escape(str(row["图片名"])),
                    html.escape(str(row["状态"])),
                    html.escape(str(row["处理来源"])),
                    artifact_cell,
                    html.escape(str(row["错误类型"])),
                    html.escape(str(row["错误原因"])),
                ]
            )
            + "</td></tr>"
        )
    summary_html.write_text(
        """<!doctype html><meta charset="utf-8"><title>预处理汇总</title>
<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:6px;vertical-align:top}th{background:#eee}</style>"""
        + f"<h1>{html.escape(branch)}预处理汇总</h1>"
        + f"<p>状态：{html.escape(status)}；成功 {len(items)}/{image_count}；"
        + f"fatal {failed_image_count}；待检查 {pending_image_count}；"
        + f"逐图缓存复用 {reused_count}</p>"
        + "<table><tr><th>图片名</th><th>状态</th><th>来源</th>"
        + "<th>中间产物</th><th>错误类型</th><th>错误原因</th></tr>"
        + "".join(html_rows)
        + "</table>",
        encoding="utf-8",
    )

    dataset_manifest = {
        "schema_version": 3,
        "created_at": now,
        "preprocessing_status": status,
        "input_dir": str(Path(input_dir).resolve()),
        "config": config,
        "config_digest": config_digest,
        "image_count": image_count,
        "prepared_image_count": len(items),
        "failed_image_count": failed_image_count,
        "pending_image_count": pending_image_count,
        "unique_image_count": unique_image_count,
        "processed_unique_image_count": processed_unique_count,
        "prepared_unique_image_count": processed_unique_count - len(failures),
        "duplicate_reuse_count": duplicate_reuse_count,
        "preprocessing_failures": failures,
        "preprocessing_failure_report": str(error_report.resolve()),
        "preprocessing_summary_json": str(summary_json.resolve()),
        "preprocessing_summary_csv": str(summary_csv.resolve()),
        "preprocessing_summary_html": str(summary_html.resolve()),
        "items": sorted(items, key=lambda item: item["file_name"]),
    }
    output = work / "dataset_manifest.json"
    _写入_json(output, dataset_manifest)
    return output
