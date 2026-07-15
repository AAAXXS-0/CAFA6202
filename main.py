"""AFAC 图表与长图分支命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afac_pipeline.common.hashing import discover_images, group_exact_duplicates
from afac_pipeline.long import LongConfig, LongPipeline
from afac_pipeline.long.步骤007_本地OCR识别 import LocalLongRecognizer
from afac_pipeline.table import TableConfig, TablePipeline
from afac_pipeline.table.local_ocr import LocalTableRecognizer
from afac_pipeline.common.local_ocr import CachedLocalOCR, RapidOCREngine
from afac_pipeline.common.submission import combine_submissions
from afac_pipeline.common.vlm_client import FinixDocClient, MAX_RETRY_COUNT


def _add_local_ocr_args(parser: argparse.ArgumentParser) -> None:
    """三个本地 OCR 命令共用的速度、精度和缓存参数。"""

    parser.add_argument("--work-dir", default=Path("work/local_ocr"), type=Path)
    parser.add_argument("--rapidocr-path", default=Path("/tmp/afac_rapidocr"), type=Path)
    parser.add_argument("--ocr-detection-side", default=2000, type=int)
    parser.add_argument("--ocr-patch-side", default=2000, type=int)
    parser.add_argument("--ocr-patch-overlap", default=160, type=int)
    parser.add_argument("--ocr-box-threshold", default=0.35, type=float)
    parser.add_argument("--ocr-text-threshold", default=0.35, type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFAC 2026 文档解析工作流")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash-report", help="统计字节完全相同的图片")
    hash_parser.add_argument("--input-dir", required=True, type=Path)

    prepare = subparsers.add_parser("prepare-tables", help="检测并切分图表，不调用 API")
    prepare.add_argument("--input-dir", required=True, type=Path)
    prepare.add_argument("--work-dir", default=Path("work/tables"), type=Path)
    prepare.add_argument("--config", type=Path)

    prepare_long = subparsers.add_parser("prepare-long", help="滑窗检测并按 H2/H3 语义准备长图")
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
    run.add_argument("--max-retries", type=int, default=MAX_RETRY_COUNT)
    run.add_argument("--workers", type=int, default=1, help="并行识别的唯一图片数")
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
    run_long.add_argument("--max-retries", type=int, default=MAX_RETRY_COUNT)
    run_long.add_argument("--workers", type=int, default=1, help="并行识别的唯一图片数")
    run_long.add_argument("--model", default="FinixDoc-VL")
    run_long.add_argument("--output-csv", default=Path("outputs/long_submission.csv"), type=Path)

    local_tables = subparsers.add_parser("run-local-tables", help="本地 OCR 识别图表并按网格重建 HTML")
    local_tables.add_argument("--manifest", required=True, type=Path)
    local_tables.add_argument("--output-csv", default=Path("outputs/table_local_ocr.csv"), type=Path)
    _add_local_ocr_args(local_tables)

    local_long = subparsers.add_parser("run-local-long", help="本地 OCR 识别长图并恢复 Markdown 标题")
    local_long.add_argument("--manifest", required=True, type=Path)
    local_long.add_argument("--output-csv", default=Path("outputs/long_local_ocr.csv"), type=Path)
    _add_local_ocr_args(local_long)

    local_all = subparsers.add_parser("run-local-all", help="本地 OCR 一键生成两个分支的最终提交 CSV")
    local_all.add_argument("--long-manifest", required=True, type=Path)
    local_all.add_argument("--table-manifest", required=True, type=Path)
    local_all.add_argument("--template", required=True, type=Path)
    local_all.add_argument("--output-csv", default=Path("outputs/local_ocr_submission.csv"), type=Path)
    _add_local_ocr_args(local_all)

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


def _local_ocr(args: argparse.Namespace) -> CachedLocalOCR:
    """创建可断点续跑的本地 OCR；RapidOCR 只在执行本地命令时才加载。"""

    engine = RapidOCREngine(
        package_path=args.rapidocr_path,
        detection_side=args.ocr_detection_side,
        box_threshold=args.ocr_box_threshold,
        text_threshold=args.ocr_text_threshold,
    )
    return CachedLocalOCR(
        engine,
        args.work_dir / "cache",
        patch_side=args.ocr_patch_side,
        patch_overlap=args.ocr_patch_overlap,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "combine-submissions":
        output = combine_submissions(args.input_csv, args.template, args.output_csv)
        print(f"提交文件合并完成：{output}")
        return

    if args.command in {"run-local-tables", "run-local-long", "run-local-all"}:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        ocr = _local_ocr(args)
        if args.command == "run-local-tables":
            results = LocalTableRecognizer(ocr, args.work_dir).recognize_dataset(
                args.manifest, args.output_csv
            )
            print(f"本地图表 OCR 完成，共 {len(results)} 张：{args.output_csv}")
            return
        if args.command == "run-local-long":
            results = LocalLongRecognizer(ocr, args.work_dir).recognize_dataset(
                args.manifest, args.output_csv
            )
            print(f"本地长图 OCR 完成，共 {len(results)} 张：{args.output_csv}")
            return
        long_csv = args.work_dir / "long_local_ocr.csv"
        table_csv = args.work_dir / "table_local_ocr.csv"
        LocalLongRecognizer(ocr, args.work_dir).recognize_dataset(
            args.long_manifest, long_csv
        )
        LocalTableRecognizer(ocr, args.work_dir).recognize_dataset(
            args.table_manifest, table_csv
        )
        output = combine_submissions(
            [long_csv, table_csv], args.template, args.output_csv
        )
        print(f"本地 OCR 最终提交完成：{output}")
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
        results = pipeline.recognize_dataset(
            args.manifest,
            _client(args),
            args.output_csv,
            max_workers=args.workers,
        )
        print(f"图表识别完成，共 {len(results)} 张：{args.output_csv}")
        return

    if args.command == "run-long":
        pipeline = LongPipeline(_long_config_for_run(args.config, args.manifest), args.work_dir)
        results = pipeline.recognize_dataset(
            args.manifest,
            _client(args),
            args.output_csv,
            max_workers=args.workers,
        )
        print(f"长图识别完成，共 {len(results)} 张：{args.output_csv}")


if __name__ == "__main__":
    main()
