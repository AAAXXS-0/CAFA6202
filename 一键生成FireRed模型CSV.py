"""使用唯一一份 FireRed-OCR-2B 顺序生成最终 100 行 CSV。"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


项目根目录 = Path(__file__).resolve().parent
FireRed环境Python = Path("/home/zero/miniconda3/envs/AFAC_FIRERED/bin/python")
系统Python = Path("/usr/bin/python3")
长图输入目录 = 项目根目录 / "raw_data/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images"
图表输入目录 = 项目根目录 / "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images"
长图配置文件 = 项目根目录 / "afac_pipeline/long/config.example.json"
图表配置文件 = 项目根目录 / "afac_pipeline/table/config.example.json"
官方提交模板 = 项目根目录 / "finix_ab_A_submit_mock.csv"
准备工作根目录 = 项目根目录 / "work/正式运行"
FireRed工作根目录 = 项目根目录 / "work/FireRed正式运行"
输出目录 = 项目根目录 / "outputs/FireRed最终提交"
模型缓存根目录 = Path.home() / ".cache/huggingface/hub/models--FireRedTeam--FireRed-OCR"


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


def 准备清单可复用(
    manifest_path: Path,
    config_digest: str,
    input_dir: Path,
) -> bool:
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


def 检查输出行数(path: Path, expected: int) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected:
        raise RuntimeError(
            f"{path.name} 应为 {expected} 行，实际为 {len(rows)} 行"
        )


def main() -> int:
    切换到FireRed环境()
    os.chdir(项目根目录)

    import torch

    from afac_pipeline.common.firered_vl_client import FireRedOCRClient
    from afac_pipeline.common.submission import combine_submissions
    from afac_pipeline.long import LongConfig, LongPipeline
    from afac_pipeline.table import TableConfig, TablePipeline

    required = [
        长图输入目录,
        图表输入目录,
        长图配置文件,
        图表配置文件,
        官方提交模板,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 FireRed 一键运行所需文件：\n" + "\n".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("FireRed 没有检测到 CUDA GPU，已停止以避免 CPU 慢跑")

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
    )
    table_ready = 准备清单可复用(
        table_manifest,
        table_config.digest(),
        图表输入目录,
    )

    if 环境开关("AFAC_FIRERED_DRY_RUN"):
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
            ]
        )

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

    long_work = FireRed工作根目录 / f"长图_{long_config.digest()[:12]}"
    table_work = FireRed工作根目录 / f"图表_{table_config.digest()[:12]}"
    输出目录.mkdir(parents=True, exist_ok=True)
    long_csv = 输出目录 / "长图结果.csv"
    table_csv = 输出目录 / "图表结果.csv"
    final_csv = 输出目录 / "finix_ab_A_submit.csv"

    print(f"[缓存签名] {client.model}", flush=True)
    print(f"[FireRed 缓存] 长图：{long_work / 'cache.sqlite3'}")
    print(f"[FireRed 缓存] 图表：{table_work / 'cache.sqlite3'}")
    print("[FireRed 长图] 开始顺序识别", flush=True)
    LongPipeline(long_config, long_work).recognize_dataset(
        long_manifest,
        client,
        long_csv,
        max_workers=1,
    )
    检查输出行数(long_csv, 50)

    print("[FireRed 图表] 开始顺序识别", flush=True)
    TablePipeline(table_config, table_work).recognize_dataset(
        table_manifest,
        client,
        table_csv,
        max_workers=1,
    )
    检查输出行数(table_csv, 50)

    combine_submissions([long_csv, table_csv], 官方提交模板, final_csv)
    检查输出行数(final_csv, 100)
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
