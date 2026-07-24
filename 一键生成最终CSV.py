"""AFAC 2026 一键预处理、识别并生成最终提交 CSV。

默认先完整检查两类图片：任何单图出现 fatal 都会被记录，待预处理全部结束后
统一报告，并且不会进入 API。使用 --force-api 时可跳过没有成功预处理清单的
图片，只识别已有成功清单；这种模式只生成部分结果，不能直接提交。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

from afac_pipeline.common.cache import merge_result_caches
from afac_pipeline.common.hashing import discover_images
from afac_pipeline.common.submission import (
    combine_submissions,
    combine_submissions_in_order,
)
from afac_pipeline.common.竞赛数据集 import 解析竞赛数据集
from afac_pipeline.common.vlm_client import (
    FinixDocClient,
    MAX_RETRY_COUNT,
    retry_delay_seconds,
)
from afac_pipeline.long import LongConfig, LongPipeline
from afac_pipeline.table import TableConfig, TablePipeline


项目根目录 = Path(__file__).resolve().parent
# 自动优先选择当前存在的 B 榜；可用 AFAC_DATASET=A 显式回看 A 榜。
数据集 = 解析竞赛数据集(项目根目录, os.environ.get("AFAC_DATASET", "auto"))
长图输入目录 = 数据集.长图目录
图表输入目录 = 数据集.图表目录
长图配置文件 = 项目根目录 / "afac_pipeline/long/config.example.json"
图表配置文件 = 项目根目录 / "afac_pipeline/table/config.example.json"
官方接口说明 = 项目根目录 / "FinixDoc_VL调用.txt"
输出目录 = 项目根目录 / "outputs/最终提交"
默认并行数 = 6
最大并行数 = 32
默认请求超时秒数 = 600

# Python 的 csv 模块默认只允许单个字段约 128 KiB。图表识别结果保存的是完整
# HTML，大表很容易超过这个值；这里只放宽读取限制，不会修改或截断识别内容。
csv.field_size_limit(sys.maxsize)


def 解析参数() -> argparse.Namespace:
    """解析一键脚本参数；保留中文别名，方便直接照 README 使用。"""

    parser = argparse.ArgumentParser(
        description="默认先完整预处理；存在 fatal 时汇总错误并停在 API 之前。"
    )
    parser.add_argument(
        "--force-api",
        "--强制进入API",
        action="store_true",
        dest="force_api",
        help=(
            "即使预处理不完整，也仅用现有成功清单进入 API；"
            "没有预处理缓存的图片直接跳过，只输出部分结果。"
        ),
    )
    return parser.parse_args()


def 检查固定文件() -> None:
    """在耗时处理前检查输入、模型与凭据，并核对图片名集合。"""

    required = [
        长图输入目录,
        图表输入目录,
        长图配置文件,
        图表配置文件,
        官方接口说明,
        项目根目录 / "360LayoutAnalysis/general6-8n.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少一键运行所需文件：\n" + "\n".join(missing))

    long_names = {path.name for path in discover_images(长图输入目录)}
    table_names = {path.name for path in discover_images(图表输入目录)}
    expected = long_names | table_names
    if expected != set(数据集.文件名映射):
        raise RuntimeError("运行时图片集合与启动时解析的数据集不一致")
    order_source = 数据集.模板路径 or "按全部 B 榜图片名稳定排序"
    print(
        f"[检查完成] {数据集.榜单} 榜：长图 {len(long_names)} 张，"
        f"图表 {len(table_names)} 张，共 {len(expected)} 张；"
        f"提交顺序来源：{order_source}",
        flush=True,
    )


def 读取准备清单(manifest_path: Path) -> dict:
    """读取数据集准备清单；损坏清单视为不可复用。"""

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def 准备清单可复用(
    manifest_path: Path,
    config_digest: str,
    input_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> bool:
    """检查清单是否匹配；默认不复用含 fatal 或缺图的不完整清单。"""

    if not manifest_path.is_file():
        return False
    manifest = 读取准备清单(manifest_path)
    if (
        manifest.get("config_digest") != config_digest
        or Path(manifest.get("input_dir", "")).resolve() != input_dir.resolve()
    ):
        return False
    if allow_incomplete:
        # 强制模式保留成功图片，避免为了失败图片再跑整套预处理。
        return True
    # schema 3 才具备逐图断点、中文中间产物和持续汇总。旧清单会
    # 进入一次新调度，但完整的单图 manifest 会直接复用，不会重新计算。
    if int(manifest.get("schema_version", 0)) < 3:
        return False
    summary_html = manifest.get("preprocessing_summary_html")
    if not summary_html or not Path(summary_html).is_file():
        return False
    image_count = int(manifest.get("image_count", -1))
    prepared_count = int(
        manifest.get("prepared_image_count", len(manifest.get("items", [])))
    )
    return (
        not manifest.get("preprocessing_failures")
        and image_count >= 0
        and prepared_count == image_count
    )


def 准备长图(
    config: LongConfig,
    work_dir: Path,
    *,
    allow_incomplete: bool,
) -> Path:
    manifest = work_dir / "dataset_manifest.json"
    if 准备清单可复用(
        manifest,
        config.digest(),
        长图输入目录,
        allow_incomplete=allow_incomplete,
    ):
        print(f"[长图准备] 复用现有清单：{manifest}", flush=True)
        return manifest
    print("[长图准备] 配置或清单有变化，开始检测与二次切块", flush=True)
    return LongPipeline(config, work_dir).prepare_directory(
        长图输入目录,
        continue_on_error=True,
    )


def 准备图表(
    config: TableConfig,
    work_dir: Path,
    *,
    allow_incomplete: bool,
) -> Path:
    manifest = work_dir / "dataset_manifest.json"
    if 准备清单可复用(
        manifest,
        config.digest(),
        图表输入目录,
        allow_incomplete=allow_incomplete,
    ):
        print(f"[图表准备] 复用现有清单：{manifest}", flush=True)
        return manifest
    print("[图表准备] 配置或清单有变化，开始检测与切块", flush=True)
    return TablePipeline(config, work_dir).prepare_directory(
        图表输入目录,
        continue_on_error=True,
    )


def 收集预处理错误(branch: str, manifest_path: Path) -> list[dict]:
    """汇总 fatal，并找出旧清单中没有成功预处理缓存的图片。"""

    manifest = 读取准备清单(manifest_path)
    report_path = manifest.get("preprocessing_failure_report")
    failures: list[dict] = []
    failed_names: set[str] = set()
    for raw in manifest.get("preprocessing_failures", []):
        item = dict(raw)
        item["branch"] = branch
        item["report_path"] = report_path
        failures.append(item)
        failed_names.update(item.get("file_names", []))

    # 兼容以前生成的不完整清单：旧清单可能缺少明确的 fatal 数组，但
    # items 之外的图片同样没有可供 API 使用的预处理缓存。
    prepared_names = {
        str(item.get("file_name"))
        for item in manifest.get("items", [])
        if item.get("file_name")
    }
    input_value = manifest.get("input_dir")
    input_dir = Path(str(input_value)) if input_value else None
    if input_dir is not None and input_dir.is_dir():
        all_names = {path.name for path in discover_images(input_dir)}
        for file_name in sorted(all_names - prepared_names - failed_names):
            failures.append(
                {
                    "branch": branch,
                    "canonical_file_name": file_name,
                    "file_names": [file_name],
                    "error_type": "MissingPreprocessingCache",
                    "error": "准备清单中没有该图片的成功预处理缓存",
                    "report_path": report_path,
                }
            )
    return failures

def 打印预处理错误(failures: list[dict], *, forced: bool) -> None:
    """一次性报告所有预处理失败图片，并说明是否会进入 API。"""

    title = (
        "[强制 API] 以下图片没有成功预处理，将跳过"
        if forced
        else "[预处理未通过] 以下图片出现 fatal"
    )
    print(f"\n{title}：", flush=True)
    for index, failure in enumerate(failures, start=1):
        file_names = "、".join(
            failure.get("file_names")
            or [failure.get("canonical_file_name", "未知图片")]
        )
        print(
            f"{index}. [{failure.get('branch', '未知分支')}] {file_names}\n"
            f"   {failure.get('error_type', 'Error')}："
            f"{failure.get('error', '未知错误')}",
            flush=True,
        )
    reports = sorted(
        {
            str(failure["report_path"])
            for failure in failures
            if failure.get("report_path")
        }
    )
    for report in reports:
        print(f"   详细记录：{report}", flush=True)


def 迁移旧缓存(work_root: Path, branch: str, current_work: Path) -> None:
    """配置改变后复用旧目录中图片字节完全相同的 API 成功切片。"""

    destination = current_work / "cache.sqlite3"
    candidates = sorted(
        (
            path
            for path in work_root.glob(f"{branch}_*/cache.sqlite3")
            if path.resolve() != destination.resolve()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    inserted = merge_result_caches(destination, candidates)
    print(
        f"[缓存迁移] {branch}：旧目录 {len(candidates)} 个，"
        f"新增整图 {inserted['image_results']} 条、切片 "
        f"{inserted['tile_results']} 条",
        flush=True,
    )


def 检查输出行数(path: Path, expected: int) -> None:
    """避免分支缺图时仍然生成表面上可提交的 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected:
        raise RuntimeError(f"{path.name} 应为 {expected} 行，实际为 {len(rows)} 行")


def main() -> int:
    args = 解析参数()
    os.chdir(项目根目录)
    检查固定文件()

    long_config = LongConfig.from_json(长图配置文件)
    table_config = TableConfig.from_json(图表配置文件)
    # 配置摘要进入工作目录名；参数变化时自动使用新目录，不污染旧缓存。
    work_root = 项目根目录 / "work/正式运行"
    long_work = work_root / f"长图_{long_config.digest()[:12]}"
    table_work = work_root / f"图表_{table_config.digest()[:12]}"
    print(f"[工作目录] 长图：{long_work}", flush=True)
    print(f"[工作目录] 图表：{table_work}", flush=True)
    print(f"[断点缓存] 长图：{long_work / 'cache.sqlite3'}", flush=True)
    print(f"[断点缓存] 图表：{table_work / 'cache.sqlite3'}", flush=True)
    workers = int(os.environ.get("FINIXDOC_WORKERS", str(默认并行数)))
    if not 1 <= workers <= 最大并行数:
        raise ValueError(
            f"FINIXDOC_WORKERS 必须位于 1 到 {最大并行数} 之间"
        )
    print(
        f"[API 并行] {workers} 个唯一图片任务；单张图片内部仍按顺序聚合",
        flush=True,
    )
    if os.environ.get("AFAC_DRY_RUN") == "1":
        print("[检查模式] 固定文件、模板、配置和工作目录均正常；不切图、不调用 API", flush=True)
        return 0

    long_manifest = 准备长图(
        long_config,
        long_work,
        allow_incomplete=args.force_api,
    )
    table_manifest = 准备图表(
        table_config,
        table_work,
        allow_incomplete=args.force_api,
    )
    preprocessing_failures = [
        *收集预处理错误("长图", long_manifest),
        *收集预处理错误("图表", table_manifest),
    ]
    if preprocessing_failures:
        打印预处理错误(preprocessing_failures, forced=args.force_api)
        if not args.force_api:
            print(
                "\n[已停止] 预处理已检查完毕，但存在 fatal；"
                "本轮不会初始化客户端，也不会调用 API。",
                flush=True,
            )
            return 1
        print(
            "\n[强制 API] 仅识别清单中已成功预处理的图片；"
            "最终只生成部分结果，不能直接提交。",
            flush=True,
        )

    迁移旧缓存(work_root, "长图", long_work)
    迁移旧缓存(work_root, "图表", table_work)
    request_timeout = int(
        os.environ.get("FINIXDOC_TIMEOUT", str(默认请求超时秒数))
    )
    if request_timeout < 30:
        raise ValueError("FINIXDOC_TIMEOUT 不应小于 30 秒")
    user_id = os.environ.get("FINIXDOC_USER_ID", "finixB2002")
    max_retries = int(
        os.environ.get("FINIXDOC_MAX_RETRIES", str(MAX_RETRY_COUNT))
    )
    if not 0 <= max_retries <= MAX_RETRY_COUNT:
        raise ValueError(
            f"FINIXDOC_MAX_RETRIES 必须位于 0 到 {MAX_RETRY_COUNT} 之间"
        )
    print(
        f"[API 重试] 最多 {max_retries} 次；"
        f"等待从 {retry_delay_seconds(1):.0f}s 开始，按 8²、9²、10²……增长",
        flush=True,
    )
    print(f"[API 超时] 每次请求最多等待 {request_timeout} 秒", flush=True)
    print("[缓存策略] 每个成功切片立即写入 SQLite；重新运行会跳过已成功切片", flush=True)
    client = FinixDocClient.from_official_doc(
        官方接口说明,
        user_id=user_id,
        timeout=request_timeout,
        max_retries=max_retries,
    )
    partial_mode = bool(preprocessing_failures)
    run_output_dir = (
        输出目录 / "强制API部分结果" if partial_mode else 输出目录
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = run_output_dir / (
        "长图部分结果.csv" if partial_mode else "长图结果.csv"
    )
    table_csv = run_output_dir / (
        "图表部分结果.csv" if partial_mode else "图表结果.csv"
    )
    final_csv = 输出目录 / 数据集.输出文件名

    failures: list[str] = []

    def 分支项目数(manifest_path: Path) -> int:
        return len(读取准备清单(manifest_path).get("items", []))

    if 分支项目数(long_manifest) == 0:
        print("[长图识别] 没有成功预处理的图片，整条分支跳过。", flush=True)
    else:
        try:
            print("[长图识别] 开始调用 FinixDoc-VL", flush=True)
            LongPipeline(long_config, long_work).recognize_dataset(
                long_manifest,
                client,
                long_csv,
                max_workers=workers,
            )
            if not partial_mode:
                检查输出行数(long_csv, 数据集.长图数量)
        except Exception as error:  # 保留另一分支继续积累缓存
            failures.append(f"长图识别失败：{error}")
            print(f"[长图识别失败] {error}", flush=True)

    if 分支项目数(table_manifest) == 0:
        print("[图表识别] 没有成功预处理的图片，整条分支跳过。", flush=True)
    else:
        try:
            print("[图表识别] 开始调用 FinixDoc-VL", flush=True)
            TablePipeline(table_config, table_work).recognize_dataset(
                table_manifest,
                client,
                table_csv,
                max_workers=workers,
            )
            if not partial_mode:
                检查输出行数(table_csv, 数据集.图表数量)
        except Exception as error:  # 保留长图已完成的缓存
            failures.append(f"图表识别失败：{error}")
            print(f"[图表识别失败] {error}", flush=True)

    if failures:
        print("\n本轮没有生成最终 CSV。无需清理，稍后重新运行本文件即可续跑：")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if partial_mode:
        print(
            "\n[强制 API 完成] 已跳过无预处理清单的图片；"
            "本轮不会生成或覆盖正式 100 行提交 CSV。",
            flush=True,
        )
        if long_csv.exists():
            print(f"- 长图部分结果：{long_csv}", flush=True)
        if table_csv.exists():
            print(f"- 图表部分结果：{table_csv}", flush=True)
        return 0

    combine_submissions_in_order(
        [long_csv, table_csv],
        数据集.提交顺序,
        final_csv,
        file_name_mapping=数据集.文件名映射,
    )
    检查输出行数(final_csv, len(数据集.提交顺序))
    print(f"\n[全部完成] 最终提交文件：{final_csv}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。已成功的切片仍在 SQLite 缓存中，重新运行即可续跑。")
        raise SystemExit(130)
    except Exception as error:
        print(f"\n一键生成失败：{error}", file=sys.stderr)
        print("修正问题后重新运行本文件即可，已有缓存不会丢失。", file=sys.stderr)
        raise SystemExit(1)
