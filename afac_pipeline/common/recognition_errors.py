"""识别编排层共享的不完整结果异常。"""

from __future__ import annotations

from typing import Any


class IncompleteImageRecognitionError(RuntimeError):
    """一张原图的所有切块都已尝试，但仍有切块失败。"""

    def __init__(self, message: str, failed_parts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.failed_parts = failed_parts
