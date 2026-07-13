"""AFAC 2026 文档解析工作流。

当前版本实现公共处理、图表分支与长图分支。两个分支共享缓存和 CSV 输出，
但检测、切块与聚合逻辑保持相互隔离。
"""

from .long import LongConfig, LongPipeline
from .table import TableConfig, TablePipeline

__all__ = ["TableConfig", "TablePipeline", "LongConfig", "LongPipeline"]
