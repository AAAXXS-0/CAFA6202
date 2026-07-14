"""AFAC 图表与长图分支命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afac_pipeline.common.hashing import discover_images, group_exact_duplicates
from afac_pipeline.long import LongConfig, LongPipeline
from afac_pipeline.table import TableConfig, TablePipeline
from afac_pipeline.common.submission import combine_submissions
from afac_pipeline.common.vlm_client import FinixDocClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFAC 2026 文档解析工作流")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash-report", help="统计字节完全相同的图片")
    hash_parser.add_argument("--input-dir", required=True, type=Path)

    prepare = subparsers.add_parser("prepare-tables", help="检测并切分图表，不调用 API")
    prepare.add_argument("--input-dir", required=True, type=Path)
    prepare.add_argument("--work-dir", default=Path("work/tables"), type=Path)
    prepare.add_argument("--config", type=Path)

    prepare_long = subparsers.add_parser("prepare-long", help="滑窗检测并二次切分长图")
    prepare_long.add_argument("--input-dir", required=True, type=Path)
    prepare_long.add_argument("--work-dir", default=Path("work/long"), type=Path)
    prepare_long.add_argument("--config", type=Path)

    run = subparsers.add_parser("run-tables", help="调用 FinixDoc-VL 并输出图表 CSV")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--work-dir", default=Path("work/tables"), type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--api-url")
    run.add_argument("--credentials-file", type=Path)
    run.add_argument("--user-id")
    run.add_argument(
        "--protocol",
        choices=("official_multipart", "chat_completions"),
        default="official_multipart",
    )
    run.add_argument("--api-key-env", default="FINIXDOC_API_KEY")
    run.add_argument("--request-timeout", type=int, default=240)
    run.add_argument("--max-retries", type=int, default=50)
    run.add_argument("--model", default="FinixDoc-VL")
    run.add_argument("--output-csv", default=Path("outputs/table_submission.csv"), type=Path)

    run_long = subparsers.add_parser("run-long", help="调用 FinixDoc-VL 并输出长图 CSV")
    run_long.add_argument("--manifest", required=True, type=Path)
    run_long.add_argument("--work-dir", default=Path("work/long"), type=Path)
    run_long.add_argument("--config", type=Path)
    run_long.add_argument("--api-url")
    run_long.add_argument("--credentials-file", type=Path)
    run_long.add_argument("--user-id")
    run_long.add_argument(
        "--protocol",
        choices=("official_multipart", "chat_completions"),
        default="official_multipart",
    )
    run_long.add_argument("--api-key-env", default="FINIXDOC_API_KEY")
    run_long.add_argument("--request-timeout", type=int, default=240)
    run_long.add_argument("--max-retries", type=int, default=50)
    run_long.add_argument("--model", default="FinixDoc-VL")
    run_long.add_argument("--output-csv", default=Path("outputs/long_submission.csv"), type=Path)

    combine = subparsers.add_parser("combine-submissions", help="按官方模板合并分支 CSV")
    combine.add_argument("--template", required=True, type=Path)
    combine.add_argument("--input-csv", required=True, action="append", type=Path)
    combine.add_argument("--output-csv", required=True, type=Path)
    return parser


def _config_for_run(config_path: Path | None, manifest_path: Path) -> TableConfig:
    if config_path is not None:
        return TableConfig.from_json(config_path)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return TableConfig(**manifest["config"])


def _long_config_for_run(config_path: Path | None, manifest_path: Path) -> LongConfig:
    if config_path is not None:
        return LongConfig.from_json(config_path)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return LongConfig(**manifest["config"])


def _client(args: argparse.Namespace) -> FinixDocClient:
    """根据命令行参数创建官方 multipart 或兼容 Chat 协议客户端。"""

    if args.credentials_file is not None:
        if args.protocol != "official_multipart":
            raise ValueError("--credentials-file 只适用于 official_multipart 协议")
        return FinixDocClient.from_official_doc(
            args.credentials_file,
            user_id=args.user_id,
            model=args.model,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
    if not args.api_url:
        raise ValueError("未提供 --api-url；也可以用 --credentials-file 读取官方说明")
    return FinixDocClient(
        api_url=args.api_url,
        model=args.model,
        api_key_env=args.api_key_env,
        protocol=args.protocol,
        user_id=args.user_id,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "combine-submissions":
        output = combine_submissions(args.input_csv, args.template, args.output_csv)
        print(f"提交文件合并完成：{output}")
        return

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
        pipeline = TablePipeline(TableConfig.from_json(args.config), args.work_dir)
        manifest = pipeline.prepare_directory(args.input_dir)
        print(f"图表切分完成：{manifest}")
        return

    if args.command == "prepare-long":
        pipeline = LongPipeline(LongConfig.from_json(args.config), args.work_dir)
        manifest = pipeline.prepare_directory(args.input_dir)
        print(f"长图二次切分完成：{manifest}")
        return

    if args.command == "run-tables":
        pipeline = TablePipeline(_config_for_run(args.config, args.manifest), args.work_dir)
        results = pipeline.recognize_dataset(args.manifest, _client(args), args.output_csv)
        print(f"图表识别完成，共 {len(results)} 张：{args.output_csv}")
        return

    if args.command == "run-long":
        pipeline = LongPipeline(_long_config_for_run(args.config, args.manifest), args.work_dir)
        results = pipeline.recognize_dataset(args.manifest, _client(args), args.output_csv)
        print(f"长图识别完成，共 {len(results)} 张：{args.output_csv}")


if __name__ == "__main__":
    main()
