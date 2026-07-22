"""检查粗切实验的矩阵尺寸与首列序列，定位重复行或漏行问题。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("矩阵目录", type=Path)
    args = parser.parse_args()

    print("文件\t尺寸\t首列开头\t首列结尾")
    for path in sorted(args.矩阵目录.glob("第*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("单元格", [])
        first_column = [row[0] if row else "" for row in rows]
        width = max((len(row) for row in rows), default=0)
        print(
            f"{path.stem}\t{len(rows)}×{width}\t"
            f"{first_column[:8]}\t{first_column[-8:]}"
        )


if __name__ == "__main__":
    main()
