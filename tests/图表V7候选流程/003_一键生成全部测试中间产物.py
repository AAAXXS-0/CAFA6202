"""顺序运行V7候选预处理和模型共识离线拼接；绝不调用模型或API。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


当前目录 = Path(__file__).resolve().parent
项目根目录 = 当前目录.parents[1]


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键生成图表V7两部分测试中间产物")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            项目根目录
            / "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images"
        ),
    )
    parser.add_argument(
        "--prepare-work",
        type=Path,
        default=项目根目录 / "work/正式运行/图表_c811361c0b5b",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=项目根目录 / "work/验证/图表V7四项改造",
    )
    parser.add_argument("--name-contains", action="append", default=[])
    parser.add_argument("--只跑预处理", action="store_true", dest="only_preprocess")
    parser.add_argument("--只跑模型共识", action="store_true", dest="only_consensus")
    return parser.parse_args()


def main() -> int:
    args = 解析参数()
    if args.only_preprocess and args.only_consensus:
        raise ValueError("--只跑预处理 与 --只跑模型共识 不能同时使用")
    filters = [
        value
        for name in args.name_contains
        for value in ("--name-contains", name)
    ]
    if not args.only_consensus:
        subprocess.run(
            [
                sys.executable,
                str(当前目录 / "001_四项改造预处理测试.py"),
                "--input-dir",
                str(args.input_dir),
                "--output-dir",
                str(args.output_root / "预处理候选"),
                *filters,
            ],
            cwd=项目根目录,
            check=True,
        )
    if not args.only_preprocess:
        subprocess.run(
            [
                sys.executable,
                str(当前目录 / "002_模型共识矩阵离线测试.py"),
                "--prepare-work",
                str(args.prepare_work),
                "--output-dir",
                str(args.output_root / "模型共识拼接"),
                *filters,
            ],
            cwd=项目根目录,
            check=True,
        )
    print(f"[全部完成] {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
