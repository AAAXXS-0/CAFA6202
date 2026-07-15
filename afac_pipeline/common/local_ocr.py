"""本地 OCR 公共接口、分块识别和磁盘缓存。

正式后处理只依赖 OCRBox，不依赖 RapidOCR 的原始返回格式。以后换成
PaddleOCR、其他 GPU OCR 或本地视觉模型时，只需新增一个引擎适配器。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Protocol

import numpy as np
from PIL import Image

from .hashing import sha256_file


@dataclass(frozen=True)
class OCRBox:
    """一个 OCR 文字框，坐标采用当前输入图片的像素坐标。"""

    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    def translated(self, x: float, y: float) -> "OCRBox":
        return OCRBox(
            self.text,
            self.confidence,
            self.x1 + x,
            self.y1 + y,
            self.x2 + x,
            self.y2 + y,
        )


@dataclass
class OCRLine:
    """按视觉位置归并后的一行文字。"""

    boxes: list[OCRBox]

    @property
    def center_y(self) -> float:
        return (min(box.y1 for box in self.boxes) + max(box.y2 for box in self.boxes)) / 2

    @property
    def text(self) -> str:
        output = ""
        for box in sorted(self.boxes, key=lambda item: item.x1):
            text = box.text.strip()
            if not text:
                continue
            if output:
                previous = output[-1]
                first = text[0]
                if previous.isascii() and first.isascii() and previous.isalnum() and first.isalnum():
                    output += " "
            output += text
        return output


class OCREngine(Protocol):
    """任何本地 OCR 引擎只需要实现这两个成员。"""

    signature: str

    def recognize(self, image: Image.Image) -> list[OCRBox]:
        ...


class RapidOCREngine:
    """对仓库现有 RapidOCR 中文 CPU 模型的轻量适配。"""

    def __init__(
        self,
        *,
        package_path: str | Path | None = "/tmp/afac_rapidocr",
        detection_side: int = 2000,
        box_threshold: float = 0.35,
        text_threshold: float = 0.35,
    ) -> None:
        if package_path is not None:
            path = str(Path(package_path))
            if Path(path).is_dir() and path not in sys.path:
                sys.path.insert(0, path)
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as error:
            raise RuntimeError(
                "没有找到 RapidOCR。可执行：python3 -m pip install --target "
                "/tmp/afac_rapidocr rapidocr_onnxruntime==1.2.3"
            ) from error

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.signature = (
            f"rapidocr-v1.2.3-side{detection_side}-"
            f"box{box_threshold:.3f}-text{text_threshold:.3f}"
        )
        self._engine = RapidOCR(
            det_model_path=None,
            det_limit_side_len=detection_side,
            det_limit_type="max",
            det_box_thresh=box_threshold,
            text_score=text_threshold,
            use_angle_cls=False,
            print_verbose=False,
        )

    def recognize(self, image: Image.Image) -> list[OCRBox]:
        result, _ = self._engine(
            np.asarray(image.convert("RGB")),
            box_thresh=self.box_threshold,
            text_score=self.text_threshold,
        )
        boxes: list[OCRBox] = []
        for raw_box, raw_text, raw_score in result or []:
            text = str(raw_text).strip()
            if not text:
                continue
            xs = [float(point[0]) for point in raw_box]
            ys = [float(point[1]) for point in raw_box]
            boxes.append(
                OCRBox(
                    text,
                    float(raw_score),
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                )
            )
        return boxes


@dataclass(frozen=True)
class OCRPatch:
    """OCR 小块和它独占的中心点范围；重叠区中的文字只会保留一次。"""

    x1: int
    y1: int
    x2: int
    y2: int
    ownership_x1: int
    ownership_y1: int
    ownership_x2: int
    ownership_y2: int


def _axis_patches(length: int, maximum: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if length <= maximum:
        return [(0, length, 0, length)]
    usable = maximum - overlap
    count = max(2, (length - overlap + usable - 1) // usable)
    # 不能让最后一块为了凑满 maximum 而与前一块大面积重复。例如 3900
    # 像素不应切成三个 2000 像素块。这里在已确定块数后重新均分尺寸，
    # 只保留目标 overlap，计算量约等于原图面积。
    balanced_size = (length + (count - 1) * overlap + count - 1) // count
    stride = balanced_size - overlap
    starts = [index * stride for index in range(count)]
    starts[-1] = length - balanced_size
    spans: list[tuple[int, int, int, int]] = []
    for index, start in enumerate(starts):
        end = min(length, start + balanced_size)
        previous_end = starts[index - 1] + balanced_size if index else 0
        own_start = 0 if index == 0 else round((start + previous_end) / 2)
        own_end = length if index + 1 == count else round((end + starts[index + 1]) / 2)
        spans.append((start, end, own_start, own_end))
    return spans


def plan_ocr_patches(
    width: int,
    height: int,
    maximum_side: int = 2000,
    overlap: int = 160,
) -> list[OCRPatch]:
    """把大图均匀分成有少量重叠的 OCR 小块，避免整体强缩小丢掉小字。"""

    if maximum_side <= 0 or not 0 <= overlap < maximum_side:
        raise ValueError("OCR 分块尺寸必须为正数，重叠必须小于分块尺寸")
    xs = _axis_patches(width, maximum_side, overlap)
    ys = _axis_patches(height, maximum_side, overlap)
    return [
        OCRPatch(x1, y1, x2, y2, ox1, oy1, ox2, oy2)
        for y1, y2, oy1, oy2 in ys
        for x1, x2, ox1, ox2 in xs
    ]


class CachedLocalOCR:
    """按“图片标识＋分块坐标＋引擎参数”缓存 OCR 框，可安全断点续跑。"""

    def __init__(
        self,
        engine: OCREngine,
        cache_dir: str | Path,
        *,
        patch_side: int = 2000,
        patch_overlap: int = 160,
    ) -> None:
        self.engine = engine
        self.cache_dir = Path(cache_dir)
        self.patch_side = patch_side
        self.patch_overlap = patch_overlap

    def _cache_path(self, image_key: str, patch: OCRPatch) -> Path:
        payload = (
            f"{image_key}\0{self.engine.signature}\0{asdict(patch)}"
        ).encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def _recognize_patch(
        self, image: Image.Image, image_key: str, patch: OCRPatch
    ) -> list[OCRBox]:
        path = self._cache_path(image_key, patch)
        if path.is_file():
            return [
                OCRBox(**item)
                for item in json.loads(path.read_text(encoding="utf-8"))
            ]
        crop = image.crop((patch.x1, patch.y1, patch.x2, patch.y2))
        boxes = self.engine.recognize(crop)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(box) for box in boxes], ensure_ascii=False),
            encoding="utf-8",
        )
        return boxes

    def recognize_path(self, image_path: str | Path, image_key: str) -> list[OCRBox]:
        image_path = Path(image_path)
        # 坐标相同但预处理参数变化时，切块内容仍可能变化。把实际文件哈希
        # 纳入键中，避免错误复用旧 OCR 框。
        image_key = f"{image_key}\0{sha256_file(image_path)}"
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        output: list[OCRBox] = []
        patches = plan_ocr_patches(
            image.width, image.height, self.patch_side, self.patch_overlap
        )
        for patch in patches:
            local_boxes = self._recognize_patch(image, image_key, patch)
            for box in local_boxes:
                global_box = box.translated(patch.x1, patch.y1)
                # 以文字框中心归属重叠区，既避免重复，也让跨缝文字能在两侧
                # 至少有一次完整识别机会。
                if (
                    patch.ownership_x1 <= global_box.center_x < patch.ownership_x2
                    and patch.ownership_y1 <= global_box.center_y < patch.ownership_y2
                ):
                    output.append(global_box)
        output.sort(key=lambda item: (item.center_y, item.x1))
        return output


def group_ocr_lines(boxes: list[OCRBox]) -> list[OCRLine]:
    """用相对框高聚合同一视觉行，适配正文与大号标题。"""

    rows: list[OCRLine] = []
    for box in sorted(boxes, key=lambda item: (item.center_y, item.x1)):
        best: OCRLine | None = None
        best_distance = float("inf")
        for row in rows[-5:]:
            row_height = max(item.height for item in row.boxes)
            distance = abs(row.center_y - box.center_y)
            tolerance = max(8.0, 0.55 * max(row_height, box.height))
            if distance <= tolerance and distance < best_distance:
                best = row
                best_distance = distance
        if best is None:
            rows.append(OCRLine([box]))
        else:
            best.boxes.append(box)
    rows.sort(key=lambda row: (row.center_y, min(box.x1 for box in row.boxes)))
    return rows
