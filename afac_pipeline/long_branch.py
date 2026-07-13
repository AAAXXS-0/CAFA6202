"""长图分支预留位置。

本文件刻意不实现任何切块逻辑。后续长图方案只要实现 prepare/recognize 两个
接口即可接入公共缓存与 CSV 输出，不需要修改图表分支。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LongImageBranch(Protocol):
    def prepare(self, input_dir: Path, work_dir: Path) -> Path:
        """准备长图切片并返回清单路径。"""

    def recognize(self, manifest_path: Path) -> dict[str, str]:
        """识别并返回 file_name 到 Markdown 的映射。"""


class ReservedLongImageBranch:
    """占位实现，防止主流程误把图表逻辑应用到长图。"""

    def prepare(self, input_dir: Path, work_dir: Path) -> Path:
        raise NotImplementedError("长图分支已预留，请在 long_branch.py 中实现你的方案")

    def recognize(self, manifest_path: Path) -> dict[str, str]:
        raise NotImplementedError("长图分支已预留，请在 long_branch.py 中实现你的方案")
