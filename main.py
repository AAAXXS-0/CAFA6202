"""AFAC 图表分支命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afac_pipeline.config import TableConfig
from afac_pipeline.hashing import discover_images, group_exact_duplicates
from afac_pipeline.table_branch import TablePipeline
from afac_pipeline.vlm_client import FinixDocClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFAC 2026 图表文档解析工作流")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash-report", help="统计字节完全相同的图片")
    hash_parser.add_argument("--input-dir", required=True, type=Path)

    prepare = subparsers.add_parser("prepare-tables", help="检测并切分图表，不调用 API")
    prepare.add_argument("--input-dir", required=True, type=Path)
    prepare.add_argument("--work-dir", default=Path("work/tables"), type=Path)
    prepare.add_argument("--config", type=Path)

    run = subparsers.add_parser("run-tables", help="调用 FinixDoc-VL 并输出图表 CSV")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--work-dir", default=Path("work/tables"), type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--api-url", required=True)
    run.add_argument("--api-key-env", default="FINIXDOC_API_KEY")
    run.add_argument("--model", default="FinixDoc-VL")
    run.add_argument("--output-csv", default=Path("outputs/table_submission.csv"), type=Path)
    return parser


def _config_for_run(config_path: Path | None, manifest_path: Path) -> TableConfig:
    if config_path is not None:
        return TableConfig.from_json(config_path)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return TableConfig(**manifest["config"])


def main() -> None:
    args = _parser().parse_args()
    if args.command == "hash-report":
        paths = discover_images(args.input_dir)
        groups = group_exact_duplicates(paths)
        duplicate_groups = [
            [path.name for path in group]
            for group in groups.values()
            if len(group) > 1
        ]
        print(
            json.dumps(
                {
                    "image_count": len(paths),
                    "unique_image_count": len(groups),
                    "reusable_count": len(paths) - len(groups),
                    "duplicate_groups": duplicate_groups,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "prepare-tables":
        config = TableConfig.from_json(args.config)
        pipeline = TablePipeline(config, args.work_dir)
        manifest = pipeline.prepare_directory(args.input_dir)
        print(f"图表切分完成：{manifest}")
        return

    if args.command == "run-tables":
        config = _config_for_run(args.config, args.manifest)
        pipeline = TablePipeline(config, args.work_dir)
        client = FinixDocClient(
            api_url=args.api_url,
            model=args.model,
            api_key_env=args.api_key_env,
        )
        results = pipeline.recognize_dataset(args.manifest, client, args.output_csv)
        print(f"识别完成，共 {len(results)} 张：{args.output_csv}")


if __name__ == "__main__":
    main()
