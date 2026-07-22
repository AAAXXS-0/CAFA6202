"""图表V7候选预处理：只生成中间产物，不修改正式流程。

本脚本集中验证三个预处理改造：

1. 行、列两个方向分别在黑线、白缝、黑白混合三套候选中选择；
2. 稀疏表在寻找列白缝时，沿行方向使用更强的自适应墨迹晕染；
3. 横向分表后先在每个分表块内做一次二维墨迹晕染，再取得实际分析框。

第四项“模型优先R×C共识拼接”由同目录002脚本离线验证。本文件不调用API，
也不会覆盖work/正式运行中的任何清单或缓存。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from afac_pipeline.common.models import Box
from afac_pipeline.table import TableConfig
from afac_pipeline.table.步骤005_黑线白带结构检测 import (
    LineSegment,
    WhiteColumnBand,
    _choose_body_window_range,
    _erase_confirmed_grid_lines,
    _first_stable_column_bands,
    adaptive_line_segments,
    clean_suspicious_column_lines,
    content_envelope_mask,
    detect_v6_grid,
    detect_v6_regions,
    map_box,
)


默认输入目录 = (
    项目根目录
    / "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images"
)
默认输出目录 = 项目根目录 / "work/验证/图表V7四项改造"
默认配置 = 项目根目录 / "afac_pipeline/table/config.example.json"
图片后缀 = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class 轴候选:
    """某个方向的一套候选边界及其可解释评分。"""

    名称: str
    中心线: tuple[int, ...]
    边界: tuple[int, ...]
    有效: bool
    最大格占比: float
    最小格宽: int
    间距稳定度: float
    说明: str


def 写JSON(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def 保存布尔图(path: Path, mask: np.ndarray) -> None:
    """true保存为黑色，false保存为白色。"""

    image = np.where(mask, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def 缩放图片(image: Image.Image, scale: float) -> Image.Image:
    return image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


def 合并近邻位置(values: Iterable[int], tolerance: int = 2) -> list[int]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    groups: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def 中心转边界(centers: Iterable[int], length: int) -> tuple[int, ...]:
    """实验版始终保留分析框两端，黑线只负责增加内部硬边界。"""

    values = 合并近邻位置(centers)
    edge = max(3, round(length * 0.006))
    interior = [value for value in values if edge < value < length - edge]
    return tuple([0, *interior, length])


def 间距稳定度(boundaries: tuple[int, ...]) -> float:
    gaps = np.diff(np.asarray(boundaries, dtype=np.int32))
    if gaps.size < 2:
        return 0.0
    typical = float(np.median(gaps))
    tolerance = max(2.0, typical * 0.22)
    return float(np.mean(np.abs(gaps - typical) <= tolerance))


def 建立轴候选(name: str, centers: Iterable[int], length: int) -> 轴候选:
    boundaries = 中心转边界(centers, length)
    if len(boundaries) < 2:
        return 轴候选(name, (), (), False, 1.0, 0, 0.0, "没有形成边界")
    gaps = np.diff(np.asarray(boundaries, dtype=np.int32))
    maximum_ratio = float(gaps.max() / max(1, length))
    minimum = int(gaps.min())
    stable = 间距稳定度(boundaries)
    valid = minimum >= 2 and maximum_ratio <= 0.95
    return 轴候选(
        name,
        tuple(合并近邻位置(centers)),
        boundaries,
        valid,
        maximum_ratio,
        minimum,
        stable,
        (
            f"{len(boundaries) - 1}格，最大格占{maximum_ratio:.1%}，"
            f"最小格{minimum}px，间距稳定度{stable:.1%}"
        ),
    )


def 选择逐轴候选(
    black: 轴候选,
    white: 轴候选,
    hybrid: 轴候选,
    *,
    formal_black: bool,
) -> tuple[轴候选, str]:
    """逐轴选择；少量黑线与白缝共存时优先黑白混合。

    这不是最终定稿规则。三套候选都会保存，便于人工判断选择是否合理。
    """

    black_count = len(black.中心线)
    if white.有效 and hybrid.有效 and 1 <= black_count <= 4:
        return hybrid, "少量黑线更像表头/分段线，与完整白缝合并"
    if formal_black and black.有效:
        return black, "该方向黑线数量和分布已通过正式可靠性检查"
    if white.有效:
        return white, "该方向采用完整白缝"
    if hybrid.有效:
        return hybrid, "纯白缝不可用，采用黑白混合补足边界"
    if black.有效:
        return black, "白缝不可用，仅保留黑线候选"
    return white, "三套候选均不可信，保留白缝结果供人工检查"


def 分表后晕染分析框(
    crop: Image.Image,
    config: TableConfig,
) -> tuple[Box, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """在单个分表块中连接表体墨迹，再取最大主体的外接框。"""

    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    ink = gray < config.ink_threshold
    density = float(ink.mean())
    # 稀疏表增加横向和纵向连接距离；这里只用于找外框，不参与内部划线。
    sparse = float(np.clip((0.10 - density) / 0.09, 0.0, 1.0))
    kernel_x = max(5, round(crop.width * (0.012 + 0.020 * sparse)))
    kernel_y = max(3, round(crop.height * (0.004 + 0.012 * sparse)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_x, kernel_y))
    smeared = cv2.dilate(ink.astype(np.uint8), kernel, iterations=1)
    close_x = max(3, round(kernel_x * 0.65)) | 1
    close_y = max(3, round(kernel_y * 0.65)) | 1
    connected = cv2.morphologyEx(
        smeared,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_x, close_y)),
        iterations=2,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected.astype(np.uint8),
        connectivity=8,
    )
    component_view = np.zeros_like(connected, dtype=np.uint8)
    selected_labels: list[int] = []
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_label = int(np.argmax(areas)) + 1
        main = stats[main_label]
        main_area = int(main[cv2.CC_STAT_AREA])
        mx1 = int(main[cv2.CC_STAT_LEFT])
        my1 = int(main[cv2.CC_STAT_TOP])
        mx2 = mx1 + int(main[cv2.CC_STAT_WIDTH])
        my2 = my1 + int(main[cv2.CC_STAT_HEIGHT])
        selected_labels.append(main_label)
        # 与主体横向明显重叠、且距离很近的小块一并保留。这样顶部表头不会
        # 因尚未和正文完全连通而被直接丢掉，远处页眉页脚仍会被排除。
        for label in range(1, count):
            if label == main_label:
                continue
            item = stats[label]
            area = int(item[cv2.CC_STAT_AREA])
            x1 = int(item[cv2.CC_STAT_LEFT])
            y1 = int(item[cv2.CC_STAT_TOP])
            x2 = x1 + int(item[cv2.CC_STAT_WIDTH])
            y2 = y1 + int(item[cv2.CC_STAT_HEIGHT])
            overlap = max(0, min(mx2, x2) - max(mx1, x1))
            overlap_ratio = overlap / max(1, min(mx2 - mx1, x2 - x1))
            vertical_gap = max(0, max(my1, y1) - min(my2, y2))
            if (
                area >= max(8, round(main_area * 0.02))
                and overlap_ratio >= 0.45
                and vertical_gap <= crop.height * 0.06
            ):
                selected_labels.append(label)
        for label in selected_labels:
            component_view[labels == label] = 1
    if not component_view.any():
        component_view = connected.astype(np.uint8)

    ys, xs = np.nonzero(component_view)
    if xs.size == 0 or ys.size == 0:
        box = Box(0, 0, crop.width, crop.height)
    else:
        padding = max(4, round(min(crop.size) * 0.012))
        box = Box(
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(crop.width, int(xs.max()) + 1 + padding),
            min(crop.height, int(ys.max()) + 1 + padding),
        )
    info = {
        "ink_density": density,
        "sparse_strength": sparse,
        "smear_kernel": [kernel_x, kernel_y],
        "close_kernel": [close_x, close_y],
        "component_count": max(0, count - 1),
        "selected_components": selected_labels,
        "local_analysis_box": box.to_dict(),
    }
    return box, ink, smeared.astype(bool), component_view.astype(bool), info


def V7稀疏列晕染(
    ink: np.ndarray,
    black_rows: list[LineSegment],
    black_columns: list[LineSegment],
    config: TableConfig,
) -> dict[str, object]:
    """稀疏图沿行方向加强扩张，并执行原有滑窗首稳检测。"""

    height, width = ink.shape
    erased = _erase_confirmed_grid_lines(ink, black_rows, black_columns)
    density = max(0.002, float(erased.mean()))
    old_ratio = float(
        np.clip(
            0.01 * 0.15 / max(0.01, density),
            config.body_column_dilate_min_ratio,
            config.body_column_dilate_max_ratio,
        )
    )
    sparse = float(np.clip((0.08 - density) / 0.07, 0.0, 1.0))
    # 密集图保持原强度；越稀疏越接近6%，用于抹掉同一列文字内部的白缝。
    ratio = old_ratio + (0.06 - old_ratio) * sparse
    kernel = max(3, round(height * ratio))
    mask = cv2.dilate(
        erased.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel)),
    ).astype(bool)
    window_height = min(
        height,
        max(config.body_window_min_height, round(height * config.body_window_height_ratio)),
    )
    step = max(20, round(height * config.body_window_step_ratio))
    starts = list(range(0, max(1, height - window_height + 1), step))
    if starts and starts[-1] != height - window_height:
        starts.append(height - window_height)
    windows: list[dict[str, object]] = []
    for start in starts:
        end = start + window_height
        raw, selected, threshold, counts = _first_stable_column_bands(
            mask[start:end].mean(axis=0),
            config,
        )
        windows.append(
            {
                "start": start,
                "end": end,
                "stable": threshold is not None,
                "threshold": threshold,
                "band_count": len(selected),
                "bands": [[band.start, band.end] for band in selected],
                "counts_until_stop": counts,
                "raw_band_count": len(raw),
            }
        )
    selected_range = _choose_body_window_range(windows, config)
    if selected_range is None:
        return {
            "mask": mask,
            "erased": erased,
            "old_ratio": old_ratio,
            "dilate_ratio": ratio,
            "kernel": kernel,
            "density": density,
            "sparse_strength": sparse,
            "windows": windows,
            "selected": None,
            "body_box": None,
            "raw": [],
            "used": [],
            "threshold": None,
            "message": "没有连续稳定的V7表体窗口",
        }
    y1 = int(windows[selected_range[0]]["start"])
    y2 = int(windows[selected_range[1] - 1]["end"])
    raw, used, threshold, counts = _first_stable_column_bands(
        mask[y1:y2].mean(axis=0),
        config,
    )
    return {
        "mask": mask,
        "erased": erased,
        "old_ratio": old_ratio,
        "dilate_ratio": ratio,
        "kernel": kernel,
        "density": density,
        "sparse_strength": sparse,
        "windows": windows,
        "selected": selected_range,
        "body_box": [0, y1, width, y2],
        "raw": raw,
        "used": used,
        "threshold": threshold,
        "counts_until_stop": counts,
        "message": (
            f"V7表体窗口{selected_range[0]}～{selected_range[1] - 1}，"
            f"y={y1}:{y2}，晕染{ratio:.2%}，稳定阈值{threshold}px，"
            f"保留{len(used)}根列白缝"
        ),
    }


def 绘制线候选(
    image: Image.Image,
    *,
    axis: str,
    black: Iterable[int],
    white: Iterable[int],
    selected: Iterable[int],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    for value in black:
        if axis == "row":
            draw.line((0, value, width, value), fill=(255, 0, 0), width=2)
        else:
            draw.line((value, 0, value, height), fill=(255, 0, 0), width=2)
    for value in white:
        if axis == "row":
            draw.line((0, value, width, value), fill=(0, 80, 255), width=2)
        else:
            draw.line((value, 0, value, height), fill=(0, 80, 255), width=2)
    for value in selected:
        if axis == "row":
            draw.line((0, value, width, value), fill=(0, 220, 70), width=3)
        else:
            draw.line((value, 0, value, height), fill=(0, 220, 70), width=3)
    return overlay


def 绘制窗口(
    image: Image.Image,
    windows: list[dict[str, object]],
    selected: tuple[int, int] | None,
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for index, item in enumerate(windows):
        color = (0, 200, 60) if selected and selected[0] <= index < selected[1] else (230, 50, 50)
        draw.rectangle(
            (0, int(item["start"]), overlay.width - 1, int(item["end"]) - 1),
            outline=color,
            width=2,
        )
    return overlay


def 绘制最终网格(
    image: Image.Image,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for y in rows:
        draw.line((0, y, overlay.width, y), fill=(0, 210, 70), width=2)
    for x in columns:
        draw.line((x, 0, x, overlay.height), fill=(0, 210, 70), width=2)
    return overlay


def 处理单图(image_path: Path, output_root: Path, config: TableConfig) -> dict[str, object]:
    image_dir = output_root / image_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    preview = 缩放图片(source, config.table_analysis_scale)
    preview.save(image_dir / "001_原图20%预览.png")
    regions = detect_v6_regions(preview, config)

    split_overlay = preview.copy()
    split_draw = ImageDraw.Draw(split_overlay)
    for index, box in enumerate(regions.split_boxes):
        split_draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(255, 0, 0), width=3)
        split_draw.text((box.x1 + 4, box.y1 + 4), str(index + 1), fill=(255, 0, 0))
    split_overlay.save(image_dir / "002_横向分表框.png")

    old_overlay = preview.copy()
    old_draw = ImageDraw.Draw(old_overlay)
    for box in regions.analysis_boxes:
        old_draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(0, 80, 255), width=3)
    old_overlay.save(image_dir / "003_旧实际分析框.png")

    new_boxes: list[Box] = []
    table_reports: list[dict[str, object]] = []
    for index, split_box in enumerate(regions.split_boxes):
        table_dir = image_dir / f"第{index + 1:03d}表"
        table_dir.mkdir(parents=True, exist_ok=True)
        split_crop = preview.crop((split_box.x1, split_box.y1, split_box.x2, split_box.y2))
        split_crop.save(table_dir / "010_分表块20%.png")
        local_box, ink, smeared, components, box_info = 分表后晕染分析框(split_crop, config)
        保存布尔图(table_dir / "011_灰度二值墨迹.png", ink)
        保存布尔图(table_dir / "012_分表后二维晕染.png", smeared)
        保存布尔图(table_dir / "013_保留的主体连通块.png", components)
        local_overlay = split_crop.copy()
        ImageDraw.Draw(local_overlay).rectangle(
            (local_box.x1, local_box.y1, local_box.x2, local_box.y2),
            outline=(0, 220, 70),
            width=3,
        )
        local_overlay.save(table_dir / "014_新分析框位置.png")
        preview_box = Box(
            split_box.x1 + local_box.x1,
            split_box.y1 + local_box.y1,
            split_box.x1 + local_box.x2,
            split_box.y1 + local_box.y2,
        ).clamp(preview.width, preview.height)
        new_boxes.append(preview_box)
        source_box = map_box(preview_box, preview.size, source.size)
        source_crop = source.crop((source_box.x1, source_box.y1, source_box.x2, source_box.y2))
        analysis = 缩放图片(source_crop, config.table_black_line_scale)
        analysis.save(table_dir / "015_统一50%分析图.png")

        formal_grid, formal = detect_v6_grid(
            analysis,
            source_box,
            config,
            black_analysis_image=analysis,
        )
        gray = np.asarray(analysis.convert("L"), dtype=np.uint8)
        ink50 = gray < config.grid_white_threshold
        envelope = content_envelope_mask(ink50)
        black_rows = adaptive_line_segments(
            ink50,
            envelope,
            0,
            config.grid_black_line_ratio,
        )
        raw_black_columns = adaptive_line_segments(
            ink50,
            envelope,
            1,
            config.grid_black_column_line_ratio,
            grayscale=gray,
            endpoint_trim_ratio=config.grid_black_column_endpoint_trim_ratio,
            minimum_contrast=config.grid_black_column_min_contrast,
            contrast_bypass_ratio=config.grid_black_column_contrast_bypass_ratio,
        )
        black_columns, rejected_columns, cleanup = clean_suspicious_column_lines(
            raw_black_columns,
            config,
        )
        sparse_columns = V7稀疏列晕染(ink50, black_rows, black_columns, config)
        保存布尔图(table_dir / "016_擦除黑线后的墨迹.png", sparse_columns["erased"])
        保存布尔图(table_dir / "017_V7稀疏列晕染.png", sparse_columns["mask"])
        绘制窗口(
            analysis,
            sparse_columns["windows"],
            sparse_columns["selected"],
        ).save(table_dir / "018_V7表体滑窗.png")

        white_rows = list(formal.white_rows)
        white_column_bands: list[WhiteColumnBand] = list(sparse_columns["used"])
        white_columns = [band.position for band in white_column_bands]
        black_row_positions = [line.position for line in black_rows]
        black_column_positions = [line.position for line in black_columns]

        row_black = 建立轴候选("行_纯黑线", black_row_positions, analysis.height)
        row_white = 建立轴候选("行_纯白缝", white_rows, analysis.height)
        row_hybrid = 建立轴候选(
            "行_黑白混合",
            [*black_row_positions, *white_rows],
            analysis.height,
        )
        column_black = 建立轴候选("列_纯黑线", black_column_positions, analysis.width)
        column_white = 建立轴候选("列_纯白缝", white_columns, analysis.width)
        column_hybrid = 建立轴候选(
            "列_黑白混合",
            [*black_column_positions, *white_columns],
            analysis.width,
        )
        selected_rows, row_reason = 选择逐轴候选(
            row_black,
            row_white,
            row_hybrid,
            formal_black=formal.row_source.startswith("black-line"),
        )
        selected_columns, column_reason = 选择逐轴候选(
            column_black,
            column_white,
            column_hybrid,
            formal_black=(
                formal.column_source.startswith("black-line")
                or formal.column_source.startswith("sparse-black")
            ),
        )
        绘制线候选(
            analysis,
            axis="row",
            black=black_row_positions,
            white=white_rows,
            selected=selected_rows.中心线,
        ).save(table_dir / "019_行轴黑红白蓝采用绿.png")
        绘制线候选(
            analysis,
            axis="column",
            black=black_column_positions,
            white=white_columns,
            selected=selected_columns.中心线,
        ).save(table_dir / "020_列轴黑红白蓝采用绿.png")
        绘制最终网格(
            analysis,
            selected_rows.边界,
            selected_columns.边界,
        ).save(table_dir / "021_V7最终候选网格.png")

        report = {
            "table_index": index,
            "split_box_20_percent": split_box.to_dict(),
            "old_analysis_box_20_percent": (
                regions.analysis_boxes[index].to_dict()
                if index < len(regions.analysis_boxes)
                else None
            ),
            "new_analysis_box_20_percent": preview_box.to_dict(),
            "new_analysis_box_source": source_box.to_dict(),
            "post_split_smear": box_info,
            "formal_grid": formal_grid.to_dict(),
            "formal_diagnostics": formal.to_dict(),
            "v7_sparse_column_smear": {
                key: value
                for key, value in sparse_columns.items()
                if key not in {"mask", "erased", "raw", "used"}
            },
            "black_column_cleanup": cleanup,
            "rejected_black_columns": [item.to_dict() for item in rejected_columns],
            "row_candidates": [
                asdict(row_black),
                asdict(row_white),
                asdict(row_hybrid),
            ],
            "column_candidates": [
                asdict(column_black),
                asdict(column_white),
                asdict(column_hybrid),
            ],
            "selected_row_source": selected_rows.名称,
            "selected_row_reason": row_reason,
            "selected_column_source": selected_columns.名称,
            "selected_column_reason": column_reason,
            "v7_candidate_shape": [
                max(0, len(selected_rows.边界) - 1),
                max(0, len(selected_columns.边界) - 1),
            ],
        }
        写JSON(table_dir / "022_全部判定数据.json", report)
        table_reports.append(report)

    new_overlay = preview.copy()
    new_draw = ImageDraw.Draw(new_overlay)
    for index, box in enumerate(new_boxes):
        new_draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(0, 220, 70), width=3)
        new_draw.text((box.x1 + 4, box.y1 + 4), str(index + 1), fill=(0, 150, 40))
    new_overlay.save(image_dir / "004_分表后晕染分析框.png")
    summary = {
        "image": str(image_path.resolve()),
        "source_size": list(source.size),
        "preview_size": list(preview.size),
        "split_count": len(regions.split_boxes),
        "tables": table_reports,
    }
    写JSON(image_dir / "005_单图汇总.json", summary)
    return summary


def 生成总览(output_root: Path, rows: list[dict[str, object]]) -> None:
    cards = []
    for item in rows:
        image_name = Path(str(item["image"])).name
        folder = Path(str(item["image"])).stem
        table_lines = []
        for table in item["tables"]:
            shape = table["v7_candidate_shape"]
            table_lines.append(
                f"第{int(table['table_index']) + 1}表：{shape[0]}×{shape[1]}，"
                f"行={html.escape(str(table['selected_row_source']))}，"
                f"列={html.escape(str(table['selected_column_source']))}"
            )
        cards.append(
            "<section>"
            f"<h2>{html.escape(image_name)}</h2>"
            f"<img src='{html.escape(folder)}/004_分表后晕染分析框.png'>"
            + "<br>".join(html.escape(line) for line in table_lines)
            + "</section>"
        )
    page = """<!doctype html><meta charset='utf-8'><title>图表V7预处理总览</title>
<style>body{font-family:sans-serif;background:#eee}section{background:white;margin:16px;padding:14px}
img{max-width:100%;max-height:720px;border:1px solid #888}</style>""" + "".join(cards)
    (output_root / "000_总览.html").write_text(page, encoding="utf-8")


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成图表V7候选预处理的全套中文中间产物；不调用API。"
    )
    parser.add_argument("--input-dir", type=Path, default=默认输入目录)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="只处理文件名包含这些片段的图片，可重复填写",
    )
    parser.add_argument("--output-dir", type=Path, default=默认输出目录)
    parser.add_argument("--config", type=Path, default=默认配置)
    return parser.parse_args()


def main() -> int:
    args = 解析参数()
    config = TableConfig.from_json(args.config)
    paths = [Path(path) for path in args.image]
    if not paths:
        paths = sorted(
            path
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in 图片后缀
        )
    if args.name_contains:
        paths = [
            path
            for path in paths
            if any(value in path.name for value in args.name_contains)
        ]
    if not paths:
        raise FileNotFoundError("没有找到需要测试的图表图片")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, start=1):
        print(f"[V7预处理 {index}/{len(paths)}] {path.name}", flush=True)
        try:
            rows.append(处理单图(path, args.output_dir, config))
        except Exception as error:
            failure = {
                "image": str(path.resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            写JSON(args.output_dir / path.stem / "999_测试失败.json", failure)
            rows.append({"image": str(path.resolve()), "tables": [], "failure": failure})
    写JSON(args.output_dir / "000_汇总.json", rows)
    生成总览(args.output_dir, rows)
    print(f"[完成] 中间产物：{args.output_dir.resolve()}")
    print(f"[总览] {(args.output_dir / '000_总览.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
