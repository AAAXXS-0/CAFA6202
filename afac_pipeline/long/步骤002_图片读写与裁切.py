"""长图批量裁切。

同一张 PNG 的几十个窗口若逐次重新打开，会反复从文件头解码。这里一次打开原图
并连续输出全部裁块，Pillow 后端牺牲较高峰值内存换取可接受的执行时间；正式
环境优先使用 libvips。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..common.image_backend import ImageBackend, PillowBackend, VipsBackend
from ..common.models import Box


def save_many_crops(
    image_path: Path,
    requests: list[tuple[Box, Path, float]],
    backend: ImageBackend,
) -> None:
    """按原图纵坐标顺序保存多个无损裁块。"""

    if not requests:
        return
    ordered = sorted(requests, key=lambda item: (item[0].y1, item[0].x1))
    for _, output_path, _ in ordered:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(backend, PillowBackend):
        with Image.open(image_path) as source:
            source.load()
            for box, output_path, scale in ordered:
                crop = source.crop((box.x1, box.y1, box.x2, box.y2)).convert("RGB")
                if scale < 1.0:
                    crop = crop.resize(
                        (
                            max(1, round(crop.width * scale)),
                            max(1, round(crop.height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                crop.save(output_path, format="PNG", compress_level=4)
        return

    if isinstance(backend, VipsBackend):
        source = backend.pyvips.Image.new_from_file(str(image_path), access="sequential")
        for box, output_path, scale in ordered:
            crop = source.crop(box.x1, box.y1, box.width, box.height)
            if scale < 1.0:
                crop = crop.resize(scale, kernel="lanczos3")
            crop.write_to_file(str(output_path), compression=4, strip=True)
        return

    # 为将来新增的后端保留通用实现。
    for box, output_path, scale in ordered:
        backend.save_crop(image_path, box, output_path, scale)
