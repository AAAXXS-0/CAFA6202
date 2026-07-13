"""超大图片读取后端。

赛题图片扩展名为 JPG，但实际编码是 PNG；这里始终由图像库读取文件头，
不根据扩展名猜测格式。优先使用 libvips 流式处理，缺少依赖时退回 Pillow。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from PIL import Image

from .hashing import sha256_file
from .models import Box, ImageMeta


# 这些是受信任的比赛数据，关闭 Pillow 针对网络未知图片的解压炸弹阈值。
# 真正的内存控制由优先使用 pyvips、限制预览尺寸以及避免整图副本来完成。
Image.MAX_IMAGE_PIXELS = None


class ImageBackend(ABC):
    name: str

    @abstractmethod
    def make_preview(self, path: Path, max_side: int) -> Image.Image:
        """生成最长边不超过 max_side 的 RGB 预览图。"""

    @abstractmethod
    def save_crop(self, path: Path, box: Box, output_path: Path, scale: float = 1.0) -> None:
        """从原图裁切并保存，scale 小于 1 时等比例缩小。"""

    def read_meta(self, path: Path, known_sha256: str | None = None) -> ImageMeta:
        with Image.open(path) as image:
            width, height = image.size
            actual_format = image.format or "UNKNOWN"
        return ImageMeta(
            path=path,
            file_name=path.name,
            width=width,
            height=height,
            actual_format=actual_format,
            file_size=path.stat().st_size,
            sha256=known_sha256 or sha256_file(path),
        )


class PillowBackend(ImageBackend):
    name = "pillow"

    def make_preview(self, path: Path, max_side: int) -> Image.Image:
        with Image.open(path) as source:
            # Pillow 对 PNG 生成缩略图时仍可能完整解码；因此大图会给出清晰警告。
            if source.width * source.height > 200_000_000:
                warnings.warn(
                    f"{path.name} 超过 2 亿像素，Pillow 可能占用较多内存；建议安装 libvips/pyvips。",
                    ResourceWarning,
                    stacklevel=2,
                )
            preview = source.convert("RGB")
            preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return preview.copy()

    def save_crop(self, path: Path, box: Box, output_path: Path, scale: float = 1.0) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            crop = source.crop((box.x1, box.y1, box.x2, box.y2)).convert("RGB")
            if scale < 1.0:
                size = (
                    max(1, round(crop.width * scale)),
                    max(1, round(crop.height * scale)),
                )
                crop = crop.resize(size, Image.Resampling.LANCZOS)
            # 使用 PNG 保留表格小字和细线，避免 JPEG 压缩影响识别。
            crop.save(output_path, format="PNG", optimize=True)


class VipsBackend(ImageBackend):
    name = "vips"

    def __init__(self) -> None:
        import pyvips  # type: ignore

        self.pyvips = pyvips

    @staticmethod
    def _to_pillow(image: Any) -> Image.Image:
        """将已经缩小的 vips 图像转换为 Pillow；此时内存规模可控。"""

        if image.bands > 3:
            image = image[:3]
        if image.bands == 1:
            image = image.bandjoin([image, image])
        array = np.frombuffer(image.write_to_memory(), dtype=np.uint8)
        array = array.reshape(image.height, image.width, image.bands)
        return Image.fromarray(array[:, :, :3], mode="RGB")

    def make_preview(self, path: Path, max_side: int) -> Image.Image:
        image = self.pyvips.Image.thumbnail(
            str(path),
            max_side,
            height=max_side,
            size="down",
            access="sequential",
        )
        return self._to_pillow(image)

    def save_crop(self, path: Path, box: Box, output_path: Path, scale: float = 1.0) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = self.pyvips.Image.new_from_file(str(path), access="sequential")
        crop = source.crop(box.x1, box.y1, box.width, box.height)
        if scale < 1.0:
            crop = crop.resize(scale, kernel="lanczos3")
        crop.write_to_file(str(output_path), compression=6, strip=True)


def create_backend(name: str = "auto") -> ImageBackend:
    """创建图像后端；auto 会在 pyvips 真正可导入时才使用它。"""

    if name in {"auto", "vips"} and importlib.util.find_spec("pyvips") is not None:
        try:
            return VipsBackend()
        except Exception:
            if name == "vips":
                raise
    if name == "vips":
        raise RuntimeError("配置要求 vips，但当前环境无法导入 pyvips/libvips")
    return PillowBackend()
