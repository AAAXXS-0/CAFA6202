"""使用唯一一份 FireRed-OCR-2B 顺序生成最终 CSV。

默认先完整检查两类图片。任一图片预处理 fatal 时，脚本会列出全部失败图片，
并在加载 FireRed 模型前停止。使用 --force-recognition 可以只识别已有成功
预处理清单的图片；这种模式只生成部分结果，不能直接提交。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


项目根目录 = Path(__file__).resolve().parent
from afac_pipeline.common.竞赛数据集 import 解析竞赛数据集

数据集 = 解析竞赛数据集(项目根目录, os.environ.get("AFAC_DATASET", "auto"))
FireRed环境Python = Path("/home/zero/miniconda3/envs/AFAC_FIRERED/bin/python")
系统Python = Path("/usr/bin/python3")
长图输入目录 = 数据集.长图目录
图表输入目录 = 数据集.图表目录
长图配置文件 = 项目根目录 / "afac_pipeline/long/config.example.json"
图表配置文件 = 项目根目录 / "afac_pipeline/table/config.example.json"
准备工作根目录 = 项目根目录 / "work/正式运行"
FireRed工作根目录 = 项目根目录 / "work/FireRed正式运行"
输出目录 = 项目根目录 / "outputs/FireRed最终提交"
模型缓存根目录 = Path.home() / ".cache/huggingface/hub/models--FireRedTeam--FireRed-OCR"

# 大表 HTML 可能超过 Python csv 模块默认约 128 KiB 的单字段上限。
csv.field_size_limit(sys.maxsize)


def 解析参数() -> argparse.Namespace:
    """解析 FireRed 一键脚本参数。"""

    parser = argparse.ArgumentParser(
        description="默认先完整预处理；存在 fatal 时汇总错误并停在模型加载之前。"
    )
    parser.add_argument(
        "--force-recognition",
        "--强制进入识别",
        "--force-api",
        action="store_true",
        dest="force_recognition",
        help=(
            "即使预处理不完整，也只识别现有成功清单；"
            "没有预处理缓存的图片会跳过，并且只输出部分结果。"
        ),
    )
    return parser.parse_args()


def 环境开关(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 只能是 1/0、true/false、yes/no 或 on/off")


def 切换到FireRed环境() -> None:
    """在独立 Torch 环境重启自身；绝不进入或导入 Paddle 环境。"""

    is_firered_python = (
        Path(sys.executable).resolve() == FireRed环境Python.resolve()
    )
    if is_firered_python and os.environ.get("PYTHONNOUSERSITE") == "1":
        return
    if not FireRed环境Python.is_file():
        raise FileNotFoundError(
            f"FireRed 环境不存在：{FireRed环境Python}。请按 README 重建。"
        )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    os.execve(
        str(FireRed环境Python),
        [str(FireRed环境Python), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def 读取准备清单(manifest_path: Path) -> dict:
    """读取准备清单；不存在、损坏或不是对象时返回空字典。"""

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
    """检查清单是否可复用；默认拒绝含 fatal 或缺图的不完整清单。"""

    if not manifest_path.is_file():
        return False
    manifest = 读取准备清单(manifest_path)
    if (
        manifest.get("config_digest") != config_digest
        or Path(manifest.get("input_dir", "")).resolve() != input_dir.resolve()
    ):
        return False
    if allow_incomplete:
        return True
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


def 运行预处理命令(arguments: list[str]) -> None:
    """预处理使用原环境；此时 FireRed 尚未实例化，不占用模型显存。"""

    environment = os.environ.copy()
    environment.pop("PYTHONNOUSERSITE", None)
    subprocess.run(
        [str(系统Python), str(项目根目录 / "main.py"), *arguments],
        cwd=项目根目录,
        env=environment,
        check=True,
    )


def 收集预处理错误(branch: str, manifest_path: Path) -> list[dict]:
    """汇总 fatal，并兼容旧清单中没有成功缓存的图片。"""

    from afac_pipeline.common.hashing import discover_images

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
    """显式打印图片名、错误原因和汇总文件位置。"""

    title = (
        "[强制 FireRed] 以下图片没有成功预处理，将跳过"
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


def 检查输出行数(path: Path, expected: int) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected:
        raise RuntimeError(
            f"{path.name} 应为 {expected} 行，实际为 {len(rows)} 行"
        )


def 迁移旧缓存(branch: str, current_work: Path) -> None:
    """配置变化时复用图片字节完全相同的旧 FireRed 切片。"""

    from afac_pipeline.common.cache import merge_result_caches

    destination = current_work / "cache.sqlite3"
    candidates = sorted(
        (
            path
            for path in FireRed工作根目录.glob(f"{branch}_*/cache.sqlite3")
            if path.resolve() != destination.resolve()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    inserted = merge_result_caches(destination, candidates)
    print(
        f"[FireRed 缓存迁移] {branch}：新增整图 "
        f"{inserted['image_results']} 条、切片 {inserted['tile_results']} 条",
        flush=True,
    )


def main() -> int:
    args = 解析参数()
    切换到FireRed环境()
    os.chdir(项目根目录)

    import torch

    from afac_pipeline.common.firered_vl_client import FireRedOCRClient
    from afac_pipeline.common.submission import combine_submissions_in_order
    from afac_pipeline.long import LongConfig, LongPipeline
    from afac_pipeline.table import TableConfig, TablePipeline

    required = [
        长图输入目录,
        图表输入目录,
        长图配置文件,
        图表配置文件,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 FireRed 一键运行所需文件：\n" + "\n".join(missing)
        )
    long_config = LongConfig.from_json(长图配置文件)
    table_config = TableConfig.from_json(图表配置文件)
    long_prepare_work = 准备工作根目录 / f"长图_{long_config.digest()[:12]}"
    table_prepare_work = 准备工作根目录 / f"图表_{table_config.digest()[:12]}"
    long_manifest = long_prepare_work / "dataset_manifest.json"
    table_manifest = table_prepare_work / "dataset_manifest.json"
    long_ready = 准备清单可复用(
        long_manifest,
        long_config.digest(),
        长图输入目录,
        allow_incomplete=args.force_recognition,
    )
    table_ready = 准备清单可复用(
        table_manifest,
        table_config.digest(),
        图表输入目录,
        allow_incomplete=args.force_recognition,
    )

    if 环境开关("AFAC_FIRERED_DRY_RUN"):
        if not torch.cuda.is_available():
            raise RuntimeError("FireRed 没有检测到 CUDA GPU，已停止以避免 CPU 慢跑")
        print("[自检通过] FireRed Torch 已识别到 CUDA GPU", flush=True)
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(
            f"[长图清单] {'可复用' if long_ready else '需要重新预处理'}："
            f"{long_manifest}"
        )
        print(
            f"[图表清单] {'可复用' if table_ready else '需要重新预处理'}："
            f"{table_manifest}"
        )
        print(
            f"[模型权重] {'已缓存' if 模型缓存根目录.is_dir() else '首次会下载'}："
            f"{模型缓存根目录}"
        )
        print("[实例数量] 正式运行固定为 1；自检不会加载模型")
        return 0

    if long_ready:
        print(f"[长图预处理] 复用：{long_manifest}", flush=True)
    else:
        print("[长图预处理] 调用现有 YOLO 预处理", flush=True)
        运行预处理命令(
            [
                "prepare-long",
                "--input-dir", str(长图输入目录),
                "--work-dir", str(long_prepare_work),
                "--config", str(长图配置文件),
                "--continue-on-error",
            ]
        )

    if table_ready:
        print(f"[图表预处理] 复用：{table_manifest}", flush=True)
    else:
        print("[图表预处理] 调用现有无模型图表预处理", flush=True)
        运行预处理命令(
            [
                "prepare-tables",
                "--input-dir", str(图表输入目录),
                "--work-dir", str(table_prepare_work),
                "--config", str(图表配置文件),
                "--continue-on-error",
            ]
        )

    preprocessing_failures = [
        *收集预处理错误("长图", long_manifest),
        *收集预处理错误("图表", table_manifest),
    ]
    if preprocessing_failures:
        打印预处理错误(
            preprocessing_failures,
            forced=args.force_recognition,
        )
        if not args.force_recognition:
            print(
                "\n[已停止] 预处理已全部检查，但存在 fatal；"
                "本轮不会加载 FireRed，也不会占用显存。",
                flush=True,
            )
            return 1
        print(
            "\n[强制 FireRed] 仅识别清单中已成功预处理的图片；"
            "最终只生成部分结果，不能直接提交。",
            flush=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("FireRed 没有检测到 CUDA GPU，已停止以避免 CPU 慢跑")

    long_work = FireRed工作根目录 / f"长图_{long_config.digest()[:12]}"
    table_work = FireRed工作根目录 / f"图表_{table_config.digest()[:12]}"
    迁移旧缓存("长图", long_work)
    迁移旧缓存("图表", table_work)

    long_max_pixels = int(
        os.environ.get("FIRERED_LONG_MAX_PIXELS", str(1024 * 28 * 28))
    )
    table_max_pixels = int(
        os.environ.get("FIRERED_TABLE_MAX_PIXELS", str(2048 * 28 * 28))
    )
    long_max_tokens = int(
        os.environ.get("FIRERED_LONG_MAX_NEW_TOKENS", "4096")
    )
    table_max_tokens = int(
        os.environ.get("FIRERED_TABLE_MAX_NEW_TOKENS", "8192")
    )
    print(
        "[FireRed] 单模型、单并行、顺序执行，"
        f"长图={long_max_pixels}px/{long_max_tokens}tok，"
        f"图表={table_max_pixels}px/{table_max_tokens}tok",
        flush=True,
    )
    client = FireRedOCRClient(
        max_pixels=long_max_pixels,
        table_max_pixels=table_max_pixels,
        max_new_tokens=long_max_tokens,
        table_max_new_tokens=table_max_tokens,
    )

    partial_mode = bool(preprocessing_failures)
    run_output_dir = (
        输出目录 / "强制识别部分结果" if partial_mode else 输出目录
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = run_output_dir / (
        "长图部分结果.csv" if partial_mode else "长图结果.csv"
    )
    table_csv = run_output_dir / (
        "图表部分结果.csv" if partial_mode else "图表结果.csv"
    )
    final_csv = 输出目录 / 数据集.输出文件名

    print(f"[缓存签名] {client.model}", flush=True)
    print(f"[FireRed 缓存] 长图：{long_work / 'cache.sqlite3'}")
    print(f"[FireRed 缓存] 图表：{table_work / 'cache.sqlite3'}")
    failures: list[str] = []

    def 分支项目数(manifest_path: Path) -> int:
        return len(读取准备清单(manifest_path).get("items", []))

    if 分支项目数(long_manifest) == 0:
        print("[FireRed 长图] 没有成功预处理的图片，整条分支跳过。", flush=True)
    else:
        try:
            print("[FireRed 长图] 开始顺序识别", flush=True)
            LongPipeline(long_config, long_work).recognize_dataset(
                long_manifest,
                client,
                long_csv,
                max_workers=1,
            )
            if not partial_mode:
                检查输出行数(long_csv, 数据集.长图数量)
        except Exception as error:
            failures.append(f"长图识别失败：{error}")
            print(f"[FireRed 长图失败] {error}", flush=True)

    if 分支项目数(table_manifest) == 0:
        print("[FireRed 图表] 没有成功预处理的图片，整条分支跳过。", flush=True)
    else:
        try:
            print("[FireRed 图表] 开始顺序识别", flush=True)
            TablePipeline(table_config, table_work).recognize_dataset(
                table_manifest,
                client,
                table_csv,
                max_workers=1,
            )
            if not partial_mode:
                检查输出行数(table_csv, 数据集.图表数量)
        except Exception as error:
            failures.append(f"图表识别失败：{error}")
            print(f"[FireRed 图表失败] {error}", flush=True)

    if failures:
        print("\n本轮没有生成最终 CSV；已有成功缓存会保留：", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1

    if partial_mode:
        print(
            "\n[强制 FireRed 完成] 已跳过无预处理清单的图片；"
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
    print(f"\n[全部完成] FireRed 最终提交：{final_csv}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断；已成功结果留在 FireRed SQLite，重跑即可续跑。")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nFireRed 一键生成失败：{error}", file=sys.stderr)
        print("修正后重跑即可；已成功的 FireRed 缓存不会丢失。", file=sys.stderr)
        raise SystemExit(1)
