"""长图分支公开入口。

具体实现按职责拆分在 long_detection、long_structure 和 long_pipeline 中；此文件
保留稳定导入位置，方便后续调整内部算法而不影响命令行与调用方。
"""

from .long_config import LongConfig
from .long_pipeline import LongPipeline

LongImageBranch = LongPipeline

__all__ = ["LongConfig", "LongPipeline", "LongImageBranch"]
