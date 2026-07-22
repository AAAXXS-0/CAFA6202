"""离线检查正式流程已保存的安全线切块响应，不发起新的 API 请求。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


项目根目录 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(项目根目录))

from afac_pipeline.table.步骤009_HTML表格软对齐 import (  # noqa: E402
    parse_table_response_checked,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("响应目录", type=Path)
    args = parser.parse_args()

    print("文件\t字符数\t解析结果")
    for path in sorted(args.响应目录.glob("*.md")):
        response = path.read_text(encoding="utf-8")
        try:
            table = parse_table_response_checked(response)
            result = f"成功 {table.row_count}×{table.column_count}"
        except Exception as error:
            result = f"失败 {type(error).__name__}：{error}"
        print(f"{path.name}\t{len(response)}\t{result}")


if __name__ == "__main__":
    main()
