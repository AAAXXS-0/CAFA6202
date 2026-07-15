"""图表解析分支的公共入口。"""

from .config import TableConfig
from .local_ocr import LocalTableRecognizer
from .pipeline import TablePipeline

__all__ = ["TableConfig", "TablePipeline", "LocalTableRecognizer"]
