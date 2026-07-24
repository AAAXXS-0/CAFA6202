"""A/B 榜数据目录、提交顺序与输出文件名的统一解析。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .hashing import discover_images


@dataclass(frozen=True)
class 竞赛数据集:
    榜单: str
    长图目录: Path
    图表目录: Path
    提交顺序: tuple[str, ...]
    # 键是磁盘上的实际文件名，值是提交 CSV 必须使用的官方文件名。
    文件名映射: dict[str, str]
    模板路径: Path | None

    @property
    def 长图数量(self) -> int:
        return len(discover_images(self.长图目录))

    @property
    def 图表数量(self) -> int:
        return len(discover_images(self.图表目录))

    @property
    def 输出文件名(self) -> str:
        return f"finix_ab_{self.榜单}_submit.csv"


def _候选目录(项目根目录: Path, 榜单: str) -> tuple[Path, Path]:
    """兼容 B 榜当前扁平目录和 A 榜历史嵌套目录。"""

    direct = 项目根目录 / "raw_data"
    nested = direct / f"AFAC {榜单}榜评测数据集(2)"
    for root in (direct, nested):
        long_dir = root / f"finix_huge_long_rest_{榜单}/images"
        table_dir = root / f"finix_huge_table_rest_{榜单}/images"
        if long_dir.is_dir() and table_dir.is_dir():
            return long_dir, table_dir
    return (
        direct / f"finix_huge_long_rest_{榜单}/images",
        direct / f"finix_huge_table_rest_{榜单}/images",
    )


def _读取模板顺序(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["file_name", "ground_truth"]:
            raise ValueError(
                f"{path.name} 表头必须严格为 file_name,ground_truth，"
                f"实际为 {reader.fieldnames}"
            )
        names = [row["file_name"] for row in reader]
    if not names or any(not name for name in names):
        raise ValueError(f"{path.name} 没有有效的 file_name")
    if len(names) != len(set(names)):
        raise ValueError(f"{path.name} 中存在重复 file_name")
    return names


def _可逆纠正乱码(name: str) -> str | None:
    """只接受 GB18030→UTF-8 能完整往返的文件名纠正。"""

    try:
        fixed = name.encode("gb18030").decode("utf-8")
        if fixed.encode("utf-8").decode("utf-8") != fixed:
            return None
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def _稳定文件身份(name: str) -> tuple[str, str] | None:
    """乱码无法直接还原时，用两端不含中文的赛事编号做唯一对应。"""

    match = re.match(
        r"^(\d+_\d+\.\d+)_.*(FXTK\d+_page\d+\.[^.]+)$",
        name,
    )
    return (match.group(1), match.group(2)) if match else None


def _建立官方文件名映射(
    raw_names: set[str], template_names: list[str]
) -> dict[str, str]:
    """把乱码原图名严格一对一映射到官方模板名，拒绝猜测和多解。"""

    official = set(template_names)
    by_identity: dict[tuple[str, str], list[str]] = {}
    for name in template_names:
        identity = _稳定文件身份(name)
        if identity is not None:
            by_identity.setdefault(identity, []).append(name)

    mapping: dict[str, str] = {}
    for raw_name in raw_names:
        if raw_name in official:
            mapping[raw_name] = raw_name
            continue
        repaired = _可逆纠正乱码(raw_name)
        if repaired in official:
            mapping[raw_name] = repaired
            continue
        identity = _稳定文件身份(raw_name)
        candidates = by_identity.get(identity, []) if identity is not None else []
        if len(candidates) == 1:
            mapping[raw_name] = candidates[0]
            continue
        raise RuntimeError(
            f"原图文件名无法唯一对应官方模板：{raw_name}；候选 {candidates}"
        )

    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("多个原图文件名映射到了同一个官方 file_name")
    return mapping


def 解析竞赛数据集(项目根目录: Path, 榜单: str = "auto") -> 竞赛数据集:
    """选择榜单，并按文件名而不是目录位置绑定每一个识别结果。

    B 榜没有随仓库提供 mock 模板时，使用全部真实图片文件名的稳定字典序。
    CSV 的真正对应关系始终由 ``file_name`` 决定，绝不假设前 50 行是哪一类图。
    """

    normalized = 榜单.strip().upper()
    if normalized == "AUTO":
        normalized = next(
            (
                candidate
                for candidate in ("B", "A")
                if all(path.is_dir() for path in _候选目录(项目根目录, candidate))
            ),
            "B",
        )
    if normalized not in {"A", "B"}:
        raise ValueError("榜单只能是 A、B 或 auto")

    long_dir, table_dir = _候选目录(项目根目录, normalized)
    long_names = {path.name for path in discover_images(long_dir)}
    table_names = {path.name for path in discover_images(table_dir)}
    duplicates = long_names & table_names
    if duplicates:
        raise RuntimeError(f"长图和图表存在同名图片：{sorted(duplicates)}")
    all_names = long_names | table_names
    if not all_names:
        raise RuntimeError(f"{normalized} 榜数据目录中没有图片")

    template = 项目根目录 / f"finix_ab_{normalized}_submit_mock.csv"
    if template.is_file():
        order = _读取模板顺序(template)
        name_mapping = _建立官方文件名映射(all_names, order)
        mapped_names = set(name_mapping.values())
        if mapped_names != set(order):
            raise RuntimeError(
                f"{normalized} 榜数据与模板文件名不一致："
                f"模板缺少原图 {sorted(set(order) - mapped_names)}；"
                f"原图多出 {sorted(mapped_names - set(order))}"
            )
        template_path: Path | None = template
    else:
        order = sorted(all_names)
        name_mapping = {name: name for name in all_names}
        template_path = None

    return 竞赛数据集(
        榜单=normalized,
        长图目录=long_dir,
        图表目录=table_dir,
        提交顺序=tuple(order),
        文件名映射=name_mapping,
        模板路径=template_path,
    )
