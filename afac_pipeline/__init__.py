"""AFAC 2026 文档解析工作流。

当前版本只实现公共处理与图表分支。长图分支故意保留为空接口，
方便后续按独立方案开发，避免两个分支互相耦合。
"""

from .config import TableConfig
from .table_branch import TablePipeline

__all__ = ["TableConfig", "TablePipeline"]
