"""图片发现与 SHA-256 精确去重。"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_images(input_dir: str | Path) -> list[Path]:
    """只返回真实图片文件，并排除 Windows Zone.Identifier 附加文件。"""

    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and ":" not in path.name
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 SHA-256，避免一次把超大图片读进内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def group_exact_duplicates(paths: list[Path]) -> dict[str, list[Path]]:
    """按文件字节哈希分组；同组图片可安全复用完整解析结果。"""

    groups: defaultdict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[sha256_file(path)].append(path)
    return dict(groups)
