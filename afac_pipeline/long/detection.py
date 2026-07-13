"""长图滑窗规划、general6 YOLO 检测与全局框去重。"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path

from .config import LongConfig
from .models import DetectionWindow, LayoutBlock
from ..common.models import Box


CANONICAL_LABELS = {
    "text": "Text",
    "title": "Title",
    "figure": "Figure",
    "table": "Table",
    "equation": "Equation",
    "caption": "Caption",
}


def plan_detection_windows(image_height: int, config: LongConfig) -> list[DetectionWindow]:
    """生成 2048 高、1792 步长的窗口，并计算重叠区责任边界。"""

    if image_height <= config.window_height:
        starts = [0]
    else:
        last_start = image_height - config.window_height
        starts = list(range(0, last_start + 1, config.window_step))
        if starts[-1] != last_start:
            starts.append(last_start)

    ends = [min(image_height, start + config.window_height) for start in starts]
    boundaries = [
        round((ends[index] + starts[index + 1]) / 2)
        for index in range(len(starts) - 1)
    ]
    windows: list[DetectionWindow] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        ownership_start = 0 if index == 0 else boundaries[index - 1]
        ownership_end = image_height if index == len(starts) - 1 else boundaries[index]
        windows.append(
            DetectionWindow(
                index=index,
                start_y=start,
                end_y=end,
                ownership_start_y=ownership_start,
                ownership_end_y=ownership_end,
                file_name=f"window_{index:05d}_y{start:07d}.png",
            )
        )
    return windows


class LongLayoutDetector(ABC):
    name: str

    @abstractmethod
    def detect(
        self,
        window_paths: list[Path],
        windows: list[DetectionWindow],
        image_width: int,
        image_height: int,
    ) -> list[LayoutBlock]:
        pass


class GeneralYoloDetector(LongLayoutDetector):
    """只使用 general6 的基础版面标签，不依赖 Toc 标签。"""

    name = "general6-yolo"

    def __init__(self, config: LongConfig):
        if importlib.util.find_spec("ultralytics") is None:
            raise RuntimeError("长图检测需要 ultralytics，请先安装 requirements.txt")
        if not Path(config.yolo_model_path).is_file():
            raise FileNotFoundError(f"general6 权重不存在：{config.yolo_model_path}")
        from ultralytics import YOLO  # type: ignore

        self.config = config
        self.model = YOLO(config.yolo_model_path)

    def _threshold(self, label: str) -> float:
        if label == "Title":
            return self.config.title_confidence
        if label == "Text":
            return self.config.text_confidence
        return self.config.other_confidence

    def detect(
        self,
        window_paths: list[Path],
        windows: list[DetectionWindow],
        image_width: int,
        image_height: int,
    ) -> list[LayoutBlock]:
        if len(window_paths) != len(windows):
            raise ValueError("窗口图片与窗口元数据数量不一致")
        blocks: list[LayoutBlock] = []
        batch_size = self.config.yolo_batch_size
        for batch_start in range(0, len(window_paths), batch_size):
            batch_paths = window_paths[batch_start : batch_start + batch_size]
            batch_windows = windows[batch_start : batch_start + batch_size]
            results = self.model.predict(
                source=[str(path) for path in batch_paths],
                conf=self.config.yolo_base_confidence,
                imgsz=self.config.yolo_imgsz,
                device="cpu",
                verbose=False,
                save=False,
                stream=False,
            )
            for result, window in zip(results, batch_windows):
                names = result.names
                for local_index, (xyxy, class_id, confidence) in enumerate(
                    zip(
                        result.boxes.xyxy.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                    )
                ):
                    raw_label = str(names[int(class_id)]).strip().lower()
                    label = CANONICAL_LABELS.get(raw_label)
                    if label is None or float(confidence) < self._threshold(label):
                        continue
                    x1, y1, x2, y2 = (round(value) for value in xyxy)
                    global_box = Box(
                        x1,
                        window.start_y + y1,
                        x2,
                        window.start_y + y2,
                    ).clamp(image_width, image_height)
                    # 重叠区以框中心点决定归属；完整框通常会由离边缘更远的窗口保留。
                    center_y = (global_box.y1 + global_box.y2) / 2
                    if not (
                        window.ownership_start_y <= center_y < window.ownership_end_y
                        or (window.index == len(windows) - 1 and center_y == image_height)
                    ):
                        continue
                    blocks.append(
                        LayoutBlock(
                            id=f"w{window.index:05d}_b{local_index:04d}",
                            label=label,
                            box=global_box,
                            confidence=float(confidence),
                            source_window=window.index,
                        )
                    )
        return deduplicate_layout_blocks(blocks, self.config.deduplicate_iou)


def _axis_overlap(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def _same_layout_element(first: LayoutBlock, second: LayoutBlock, iou_threshold: float) -> bool:
    if first.label != second.label:
        return False
    if first.box.iou(second.box) >= iou_threshold:
        return True
    vertical = _axis_overlap(first.box.y1, first.box.y2, second.box.y1, second.box.y2)
    horizontal = _axis_overlap(first.box.x1, first.box.x2, second.box.x1, second.box.x2)
    min_height = max(1, min(first.box.height, second.box.height))
    min_width = max(1, min(first.box.width, second.box.width))
    return vertical / min_height >= 0.80 and horizontal / min_width >= 0.80


def deduplicate_layout_blocks(
    blocks: list[LayoutBlock], iou_threshold: float = 0.50
) -> list[LayoutBlock]:
    """同类别框按置信度做全局 NMS，并恢复阅读顺序。"""

    kept: list[LayoutBlock] = []
    for candidate in sorted(blocks, key=lambda item: item.confidence, reverse=True):
        if any(_same_layout_element(candidate, existing, iou_threshold) for existing in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.box.y1, item.box.x1, item.box.y2))
