"""不依赖模型的低清墨水密度表格区域实验。

这里解决的是“整张表在哪里”，不负责判断内部行列。算法故意强烈缩小、
模糊并连接文字，使单元格内部的小空隙消失，而表格外围的大白边仍保留。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..common.models import Box


@dataclass(frozen=True)
class InkRegion:
    """墨水连通区域在预览图中的轮廓和外接矩形。"""

    contour: tuple[tuple[int, int], ...]
    box: Box
    contour_area: float
    box_area_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "contour": [list(point) for point in self.contour],
            "box": self.box.to_dict(),
            "contour_area": self.contour_area,
            "box_area_ratio": self.box_area_ratio,
        }


@dataclass(frozen=True)
class InkRegionResult:
    coarse_density: np.ndarray
    connected_mask: np.ndarray
    regions: tuple[InkRegion, ...]
    preview_size: tuple[int, int]
    coarse_size: tuple[int, int]


def _odd(value: int) -> int:
    value = max(3, value)
    return value if value % 2 else value + 1


def detect_ink_regions(
    preview: Image.Image,
    *,
    coarse_max_side: int = 384,
    ink_threshold: int = 245,
    minimum_density: float = 0.008,
    blur_ratio: float = 0.012,
    closing_ratio: float = 0.018,
    minimum_box_area_ratio: float = 0.01,
) -> InkRegionResult:
    """把稀疏文字模糊成二维内容块，并返回较大的外轮廓。

    缩放使用面积平均，因此 coarse_density 的每个像素表示对应原区域内的
    墨水比例。高斯模糊和闭运算只在低清图上进行，计算量很小。
    """

    if not 64 <= coarse_max_side <= 2048:
        raise ValueError("coarse_max_side 应位于 64 到 2048 之间")
    gray = np.asarray(preview.convert("L"))
    ink = (gray < ink_threshold).astype(np.float32)
    scale = min(1.0, coarse_max_side / max(preview.size))
    coarse_width = max(1, round(preview.width * scale))
    coarse_height = max(1, round(preview.height * scale))
    density = cv2.resize(
        ink,
        (coarse_width, coarse_height),
        interpolation=cv2.INTER_AREA,
    )

    sigma_x = max(1.0, coarse_width * blur_ratio)
    sigma_y = max(1.0, coarse_height * blur_ratio)
    blurred = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    mask = (blurred >= minimum_density).astype(np.uint8) * 255

    # 闭运算连接同一表格内尚未被模糊填平的空隙；内核随低清图尺寸变化，
    # 避免同一参数只能适配某一种分辨率。
    kernel_width = _odd(round(coarse_width * closing_ratio))
    kernel_height = _odd(round(coarse_height * closing_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_width, kernel_height),
    )
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    regions: list[InkRegion] = []
    preview_area = preview.width * preview.height
    scale_x = preview.width / coarse_width
    scale_y = preview.height / coarse_height
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        mapped_box = Box(
            round(x * scale_x),
            round(y * scale_y),
            round((x + width) * scale_x),
            round((y + height) * scale_y),
        ).clamp(preview.width, preview.height)
        area_ratio = mapped_box.area / max(1, preview_area)
        if area_ratio < minimum_box_area_ratio:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.012 * perimeter, True)
        points = tuple(
            (
                round(int(point[0][0]) * scale_x),
                round(int(point[0][1]) * scale_y),
            )
            for point in polygon
        )
        regions.append(
            InkRegion(
                contour=points,
                box=mapped_box,
                contour_area=float(cv2.contourArea(contour) * scale_x * scale_y),
                box_area_ratio=area_ratio,
            )
        )
    regions.sort(key=lambda item: item.contour_area, reverse=True)
    return InkRegionResult(
        coarse_density=density,
        connected_mask=connected,
        regions=tuple(regions),
        preview_size=preview.size,
        coarse_size=(coarse_width, coarse_height),
    )


def density_visualization(density: np.ndarray) -> Image.Image:
    """生成白底黑墨的可视化；高密度区域越黑。"""

    reference = max(0.02, float(np.quantile(density, 0.98)))
    normalized = np.clip(density / reference, 0.0, 1.0)
    image = np.round((1.0 - normalized) * 255).astype(np.uint8)
    return Image.fromarray(image, mode="L")
