"""AFAC 2026 一键准备、识别并生成最终 100 行提交 CSV。

直接运行本文件，不需要输入任何命令行参数。默认并行识别 6 张唯一图片；
API 或网络中断后再次运行即可，SQLite 会复用成功的切片和整图结果。
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

from afac_pipeline.common.cache import merge_result_caches
from afac_pipeline.common.hashing import discover_images
from afac_pipeline.common.submission import combine_submissions
from afac_pipeline.common.vlm_client import (
    FinixDocClient,
    MAX_RETRY_COUNT,
    retry_delay_seconds,
)
from afac_pipeline.long import LongConfig, LongPipeline
from afac_pipeline.table import TableConfig, TablePipeline


项目根目录 = Path(__file__).resolve().parent
长图输入目录 = 项目根目录 / "raw_data/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images"
图表输入目录 = 项目根目录 / "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images"
长图配置文件 = 项目根目录 / "afac_pipeline/long/config.example.json"
图表配置文件 = 项目根目录 / "afac_pipeline/table/config.example.json"
官方接口说明 = 项目根目录 / "FinixDoc_VL调用.txt"
官方提交模板 = 项目根目录 / "finix_ab_A_submit_mock.csv"
输出目录 = 项目根目录 / "outputs/最终提交"
默认并行数 = 6
最大并行数 = 32
默认请求超时秒数 = 600


def 检查固定文件() -> None:
    """在耗时处理前检查输入、模型、凭据说明和模板是否齐全。"""

    required = [
        长图输入目录,
        图表输入目录,
        长图配置文件,
        图表配置文件,
        官方接口说明,
        官方提交模板,
        项目根目录 / "360LayoutAnalysis/general6-8n.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少一键运行所需文件：\n" + "\n".join(missing))

    long_names = {path.name for path in discover_images(长图输入目录)}
    table_names = {path.name for path in discover_images(图表输入目录)}
    with 官方提交模板.open("r", encoding="utf-8-sig", newline="") as file:
        template_names = {row["file_name"] for row in csv.DictReader(file)}
    expected = long_names | table_names
    if long_names & table_names:
        raise RuntimeError("长图和图表目录存在同名图片，无法安全合并")
    if expected != template_names:
        raise RuntimeError(
            "数据目录与官方模板文件名不一致："
            f"模板缺少 {sorted(expected - template_names)}；"
            f"模板多出 {sorted(template_names - expected)}"
        )
    print(
        f"[检查完成] 长图 {len(long_names)} 张，图表 {len(table_names)} 张，"
        f"模板 {len(template_names)} 行",
        flush=True,
    )


def 准备清单可复用(manifest_path: Path, config_digest: str, input_dir: Path) -> bool:
    """只有配置和输入目录都一致时才复用准备清单。"""

    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("config_digest") == config_digest
        and Path(manifest.get("input_dir", "")).resolve() == input_dir.resolve()
    )


def 准备长图(config: LongConfig, work_dir: Path) -> Path:
    manifest = work_dir / "dataset_manifest.json"
    if 准备清单可复用(manifest, config.digest(), 长图输入目录):
        print(f"[长图准备] 复用现有清单：{manifest}", flush=True)
        return manifest
    print("[长图准备] 配置或清单有变化，开始滑窗检测与二次切块", flush=True)
    return LongPipeline(config, work_dir).prepare_directory(长图输入目录)


def 准备图表(config: TableConfig, work_dir: Path) -> Path:
    manifest = work_dir / "dataset_manifest.json"
    if 准备清单可复用(manifest, config.digest(), 图表输入目录):
        print(f"[图表准备] 复用现有清单：{manifest}", flush=True)
        return manifest
    print("[图表准备] 配置或清单有变化，开始检测与切块", flush=True)
    return TablePipeline(config, work_dir).prepare_directory(图表输入目录)


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

    long_manifest = 准备长图(long_config, long_work)
    table_manifest = 准备图表(table_config, table_work)
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
    输出目录.mkdir(parents=True, exist_ok=True)
    long_csv = 输出目录 / "长图结果.csv"
    table_csv = 输出目录 / "图表结果.csv"
    final_csv = 输出目录 / "finix_ab_A_submit.csv"

    failures: list[str] = []
    try:
        print("[长图识别] 开始调用 FinixDoc-VL", flush=True)
        LongPipeline(long_config, long_work).recognize_dataset(
            long_manifest,
            client,
            long_csv,
            max_workers=workers,
        )
        检查输出行数(long_csv, 50)
    except Exception as error:  # 保留另一分支继续积累缓存
        failures.append(f"长图识别失败：{error}")
        print(f"[长图识别失败] {error}", flush=True)

    try:
        print("[图表识别] 开始调用 FinixDoc-VL", flush=True)
        TablePipeline(table_config, table_work).recognize_dataset(
            table_manifest,
            client,
            table_csv,
            max_workers=workers,
        )
        检查输出行数(table_csv, 50)
    except Exception as error:  # 保留长图已完成的缓存
        failures.append(f"图表识别失败：{error}")
        print(f"[图表识别失败] {error}", flush=True)

    if failures:
        print("\n本轮没有生成最终 CSV。无需清理，稍后重新运行本文件即可续跑：")
        for failure in failures:
            print(f"- {failure}")
        return 1

    combine_submissions([long_csv, table_csv], 官方提交模板, final_csv)
    检查输出行数(final_csv, 100)
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
