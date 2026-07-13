"""表格区域检测器。

YOLO 检测器负责复杂或无边框表格；投影检测器不含模型参数，主要识别
有明显横向网格线的金融表格，同时也是缺少 Ultralytics 时的安全后备。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from .config import TableConfig
from .models import Box, DetectedBox


class TableDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, preview: Image.Image) -> list[DetectedBox]:
        pass


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """把一维布尔数组转换成左闭右开连续区间。"""

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks] + 1, indices[-1] + 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def find_content_box(preview: Image.Image, threshold: int = 235) -> Box:
    """根据主要非白内容找到外框；失败时返回整图。

    页面底部偶尔存在很小的页码或扫描标记。若直接取所有非白像素的最小外框，
    会把主体表格到页脚之间的大段空白也保留下来。这里先按行聚类，再过滤墨迹
    量低于总量 1% 的孤立内容带；正常的多块表格仍会全部保留。
    """

    gray = np.asarray(preview.convert("L"))
    ink = gray < threshold
    row_mask = ink.mean(axis=1) > 0.001
    row_runs = _runs(row_mask)
    if not row_runs:
        return Box(0, 0, preview.width, preview.height)

    # 将距离很近的文字/表格行归为同一内容带，避免逐行被拆开。
    max_gap = max(4, round(preview.height * 0.03))
    bands: list[list[int]] = [[row_runs[0][0], row_runs[0][1]]]
    for start, end in row_runs[1:]:
        if start - bands[-1][1] <= max_gap:
            bands[-1][1] = end
        else:
            bands.append([start, end])

    masses = [int(ink[start:end].sum()) for start, end in bands]
    total_mass = max(1, sum(masses))
    significant = [
        band for band, mass in zip(bands, masses) if mass / total_mass >= 0.01
    ] or bands
    y1 = min(start for start, _ in significant)
    y2 = max(end for _, end in significant)
    columns = np.flatnonzero(ink[y1:y2].mean(axis=0) > 0.001)
    if columns.size == 0:
        return Box(0, y1, preview.width, y2)
    return Box(
        int(columns[0]),
        y1,
        int(columns[-1] + 1),
        y2,
    )


class ProjectionTableDetector(TableDetector):
    name = "projection"

    def __init__(self, config: TableConfig):
        self.config = config

    def detect(self, preview: Image.Image) -> list[DetectedBox]:
        gray = np.asarray(preview.convert("L"))
        ink = gray < self.config.projection_threshold

        # 表格横线通常横跨较大页面宽度，而普通文字行的黑色覆盖率明显更低。
        row_ratio = ink.mean(axis=1)
        line_mask = row_ratio >= self.config.projection_min_line_ratio
        line_runs = _runs(line_mask)
        line_centers = [round((start + end - 1) / 2) for start, end in line_runs]

        if not line_centers:
            content = find_content_box(preview, self.config.projection_threshold)
            return [DetectedBox(content, confidence=0.1, source="projection-fallback")]

        max_gap = max(8, round(preview.height * self.config.projection_max_line_gap_ratio))
        clusters: list[list[int]] = [[line_centers[0]]]
        for center in line_centers[1:]:
            if center - clusters[-1][-1] <= max_gap:
                clusters[-1].append(center)
            else:
                clusters.append([center])

        detected: list[DetectedBox] = []
        for cluster in clusters:
            if len(cluster) < self.config.projection_min_lines:
                continue
            y1 = max(0, cluster[0] - 4)
            y2 = min(preview.height, cluster[-1] + 5)
            region_ink = ink[y1:y2]
            columns = np.flatnonzero(region_ink.mean(axis=0) > 0.002)
            if columns.size == 0:
                continue
            box = Box(int(columns[0]), y1, int(columns[-1] + 1), y2)
            if box.width < preview.width * 0.15 or box.height < preview.height * 0.02:
                continue
            confidence = min(0.95, 0.35 + len(cluster) * 0.04)
            detected.append(DetectedBox(box, confidence=confidence, source=self.name))

        if not detected:
            content = find_content_box(preview, self.config.projection_threshold)
            detected.append(DetectedBox(content, confidence=0.1, source="projection-fallback"))
        return detected


class YoloTableDetector(TableDetector):
    name = "yolo"

    def __init__(self, config: TableConfig):
        from ultralytics import YOLO  # type: ignore

        self.config = config
        self.model = YOLO(config.yolo_model_path)

    def detect(self, preview: Image.Image) -> list[DetectedBox]:
        results = self.model.predict(
            source=np.asarray(preview.convert("RGB")),
            conf=self.config.yolo_confidence,
            imgsz=self.config.yolo_imgsz,
            verbose=False,
            device="cpu",
        )
        result = results[0]
        names = result.names
        detected: list[DetectedBox] = []
        for xyxy, class_id, confidence in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            label = str(names[int(class_id)])
            if "table" not in label.lower() and "表" not in label:
                continue
            x1, y1, x2, y2 = (round(value) for value in xyxy)
            detected.append(
                DetectedBox(
                    Box(x1, y1, x2, y2).clamp(preview.width, preview.height),
                    label=label,
                    confidence=float(confidence),
                    source=self.name,
                )
            )
        return detected


def _yolo_available(config: TableConfig) -> bool:
    return (
        importlib.util.find_spec("ultralytics") is not None
        and Path(config.yolo_model_path).is_file()
    )


def create_detector(config: TableConfig) -> TableDetector:
    if config.detector == "projection":
        return ProjectionTableDetector(config)
    if config.detector == "yolo":
        if not _yolo_available(config):
            raise RuntimeError("配置要求 YOLO，但 ultralytics 或模型权重不可用")
        return YoloTableDetector(config)
    if _yolo_available(config):
        return YoloTableDetector(config)
    return ProjectionTableDetector(config)


def suppress_duplicate_boxes(boxes: list[DetectedBox], iou_threshold: float = 0.65) -> list[DetectedBox]:
    """按置信度执行简单 NMS，避免重复裁切同一张表。"""

    kept: list[DetectedBox] = []
    for candidate in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        if any(candidate.box.iou(existing.box) >= iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.box.y1, item.box.x1))
