"""长图批量裁切和带祖先标题的复合请求图生成。"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from ..common.image_backend import ImageBackend, PillowBackend, VipsBackend
from ..common.models import Box

if TYPE_CHECKING:
    from .步骤005_大模型请求打包 import RecognitionPack


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

    for box, output_path, scale in ordered:
        backend.save_crop(image_path, box, output_path, scale)


def save_recognition_pack_images(
    image_path: Path,
    packs: list["RecognitionPack"],
    output_dir: Path,
    backend: ImageBackend,
    *,
    context_gap: int,
    maximum_height: int,
) -> None:
    """保存正文块，并把祖先 H2/H3 的原图标题条拼到续块顶部。

    请求块可通过 body_scale 只缩放输出图片而保留原图坐标；目录整块输入
    使用这一能力。超大原图仍由图像后端负责裁切，最终只打开小图片。
    ``vlm_request_parts`` 故意保留，方便检查复合请求的每一个来源块。
    """

    if not packs:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir.parent / "vlm_request_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    crop_requests: list[tuple[Box, Path, float]] = []
    body_paths: dict[str, Path] = {}
    context_paths: dict[tuple[int, int, int, int], Path] = {}
    for pack in packs:
        body_path = parts_dir / f"{pack.id}_body.png"
        body_paths[pack.id] = body_path
        crop_requests.append((pack.source_box, body_path, pack.body_scale))
        for box in pack.context_boxes:
            key = (box.x1, box.y1, box.x2, box.y2)
            if key in context_paths:
                continue
            path = parts_dir / f"context_y{box.y1:07d}_{box.y2:07d}.png"
            context_paths[key] = path
            crop_requests.append((box, path, 1.0))
    save_many_crops(image_path, crop_requests, backend)

    for pack in packs:
        output_path = output_dir / pack.file_name
        body_path = body_paths[pack.id]
        if not pack.context_boxes:
            shutil.copyfile(body_path, output_path)
            continue
        images: list[Image.Image] = []
        for box in pack.context_boxes:
            key = (box.x1, box.y1, box.x2, box.y2)
            with Image.open(context_paths[key]) as source:
                images.append(source.convert("RGB").copy())
        with Image.open(body_path) as source:
            images.append(source.convert("RGB").copy())
        width = max(image.width for image in images)
        height = sum(image.height for image in images) + context_gap * (len(images) - 1)
        if height > maximum_height:
            raise RuntimeError(
                f"复合请求 {pack.id} 高 {height}px，超过限制 {maximum_height}px"
            )
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        cursor = 0
        for index, image in enumerate(images):
            canvas.paste(image, (0, cursor))
            cursor += image.height
            if index + 1 < len(images):
                line_y = cursor + context_gap // 2
                draw.line((0, line_y, width, line_y), fill=(190, 190, 190), width=1)
                cursor += context_gap
        canvas.save(output_path, format="PNG", compress_level=4)
