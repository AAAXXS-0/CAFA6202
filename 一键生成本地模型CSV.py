"""使用本地 PaddleOCR-VL-1.6 生成最终 100 行 CSV。

用户可直接用系统 Python 运行本文件；脚本会自动切换到独立 Conda 环境。
预处理清单沿用正式流程，本地模型的 SQLite 缓存和输出单独存放。
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


项目根目录 = Path(__file__).resolve().parent
from afac_pipeline.common.竞赛数据集 import 解析竞赛数据集

数据集 = 解析竞赛数据集(项目根目录, os.environ.get("AFAC_DATASET", "auto"))
本地环境Python = Path("/home/zero/miniconda3/envs/AFAC_LOCAL_VL/bin/python")
系统Python = Path("/usr/bin/python3")
长图输入目录 = 数据集.长图目录
图表输入目录 = 数据集.图表目录
长图配置文件 = 项目根目录 / "afac_pipeline/long/config.example.json"
图表配置文件 = 项目根目录 / "afac_pipeline/table/config.example.json"
准备工作根目录 = 项目根目录 / "work/正式运行"
本地工作根目录 = 项目根目录 / "work/本地模型正式运行"
输出目录 = 项目根目录 / "outputs/本地模型最终提交"
本地模型目录 = Path.home() / ".paddlex/official_models/PaddleOCR-VL-1.6"
版面模型目录 = Path.home() / ".paddlex/official_models/PP-DocLayoutV3"


def 环境开关(name: str, default: bool = False) -> bool:
    """把常见的 1/0、true/false 环境变量转换成布尔值。"""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 只能是 1/0、true/false、yes/no 或 on/off")


def 切换到本地环境() -> None:
    """携带必要 WSL/CUDA 变量，在专用 Conda 环境中重新执行自身。"""

    is_local_python = Path(sys.executable).resolve() == 本地环境Python.resolve()
    library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    environment_ready = (
        os.environ.get("PYTHONNOUSERSITE") == "1"
        and "/usr/lib/wsl/lib" in library_paths
        and os.environ.get("FLAGS_allocator_strategy") == "auto_growth"
    )
    if is_local_python and environment_ready:
        return
    if not 本地环境Python.is_file():
        raise FileNotFoundError(
            f"本地模型环境不存在：{本地环境Python}。请按根目录 README 重建环境。"
        )

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    wsl_cuda = "/usr/lib/wsl/lib"
    current_library_path = environment.get("LD_LIBRARY_PATH", "")
    current_parts = [item for item in current_library_path.split(":") if item]
    if wsl_cuda not in current_parts:
        current_parts.insert(0, wsl_cuda)
    environment["LD_LIBRARY_PATH"] = ":".join(current_parts)
    environment["FLAGS_allocator_strategy"] = "auto_growth"
    environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.execve(
        str(本地环境Python),
        [str(本地环境Python), str(Path(__file__).resolve()), *sys.argv[1:]],
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
    """预处理仍使用现有系统环境，避免在 Paddle 环境重复安装 YOLO/Torch。"""

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
        raise RuntimeError(f"{path.name} 应为 {expected} 行，实际为 {len(rows)} 行")


def main() -> int:
    切换到本地环境()
    os.chdir(项目根目录)

    import paddle

    from afac_pipeline.common.local_vl_client import PaddleOCRVLClient
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
        raise FileNotFoundError("缺少本地一键运行所需文件：\n" + "\n".join(missing))
    if paddle.device.cuda.device_count() < 1:
        raise RuntimeError("Paddle 没有检测到 CUDA GPU，已停止，避免误用 CPU 慢跑")

    long_config = LongConfig.from_json(长图配置文件)
    table_config = TableConfig.from_json(图表配置文件)
    long_prepare_work = 准备工作根目录 / f"长图_{long_config.digest()[:12]}"
    table_prepare_work = 准备工作根目录 / f"图表_{table_config.digest()[:12]}"
    long_manifest = long_prepare_work / "dataset_manifest.json"
    table_manifest = table_prepare_work / "dataset_manifest.json"
    long_ready = 准备清单可复用(
        long_manifest, long_config.digest(), 长图输入目录
    )
    table_ready = 准备清单可复用(
        table_manifest, table_config.digest(), 图表输入目录
    )

    if 环境开关("AFAC_LOCAL_VL_DRY_RUN"):
        print("[自检通过] Paddle 已识别到 CUDA GPU", flush=True)
        print(f"[长图清单] {'可复用' if long_ready else '需要重新预处理'}：{long_manifest}")
        print(f"[图表清单] {'可复用' if table_ready else '需要重新预处理'}：{table_manifest}")
        print(f"[版面模型] {'已缓存' if 版面模型目录.is_dir() else '首次运行会下载'}：{版面模型目录}")
        print(f"[识别模型] {'已缓存' if 本地模型目录.is_dir() else '首次运行会下载'}：{本地模型目录}")
        return 0

    if long_ready:
        print(f"[长图预处理] 复用：{long_manifest}", flush=True)
    else:
        print("[长图预处理] 清单不可复用，调用现有 YOLO 预处理", flush=True)
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
        print("[图表预处理] 清单不可复用，调用现有无模型表格预处理", flush=True)
        运行预处理命令(
            [
                "prepare-tables",
                "--input-dir", str(图表输入目录),
                "--work-dir", str(table_prepare_work),
                "--config", str(图表配置文件),
            ]
        )

    # 长图 30 万像素是当前 4060 的速度/质量甜点位；50 万以上在高长比
    # 切块上进入非线性慢区间。图表文字更密，暂时保留较高上限。
    max_pixels = int(os.environ.get("PADDLEOCR_MAX_PIXELS", "300000"))
    max_new_tokens = int(os.environ.get("PADDLEOCR_MAX_NEW_TOKENS", "1024"))
    table_max_pixels = int(
        os.environ.get("PADDLEOCR_TABLE_MAX_PIXELS", "1000000")
    )
    table_max_new_tokens = int(
        os.environ.get("PADDLEOCR_TABLE_MAX_NEW_TOKENS", "4096")
    )
    print("[输出策略] 默认强制完成：识别坏块补空并记录降级", flush=True)
    print(
        "[本地模型] PaddleOCR-VL-1.6，GPU 单并行，"
        f"长图={max_pixels}px/{max_new_tokens}tok，"
        f"图表={table_max_pixels}px/{table_max_new_tokens}tok",
        flush=True,
    )
    client = PaddleOCRVLClient(
        pipeline_version="v1.6",
        device="gpu:0",
        max_pixels=max_pixels,
        table_max_pixels=table_max_pixels,
        max_new_tokens=max_new_tokens,
        table_max_new_tokens=table_max_new_tokens,
    )

    long_work = 本地工作根目录 / f"长图_{long_config.digest()[:12]}"
    table_work = 本地工作根目录 / f"图表_{table_config.digest()[:12]}"
    输出目录.mkdir(parents=True, exist_ok=True)
    long_csv = 输出目录 / "长图结果.csv"
    table_csv = 输出目录 / "图表结果.csv"
    final_csv = 输出目录 / 数据集.输出文件名

    print(f"[缓存签名] {client.model}", flush=True)
    print(f"[本地缓存] 长图：{long_work / 'cache.sqlite3'}", flush=True)
    print(f"[本地缓存] 图表：{table_work / 'cache.sqlite3'}", flush=True)
    print("[本地长图] 开始顺序识别", flush=True)
    LongPipeline(long_config, long_work).recognize_dataset(
        long_manifest,
        client,
        long_csv,
        max_workers=1,
        allow_degraded_output=True,
    )
    检查输出行数(long_csv, 数据集.长图数量)

    print("[本地图表] 开始顺序识别", flush=True)
    TablePipeline(table_config, table_work).recognize_dataset(
        table_manifest,
        client,
        table_csv,
        max_workers=1,
        allow_degraded_output=True,
    )
    检查输出行数(table_csv, 数据集.图表数量)

    combine_submissions_in_order(
        [long_csv, table_csv],
        数据集.提交顺序,
        final_csv,
        file_name_mapping=数据集.文件名映射,
    )
    检查输出行数(final_csv, len(数据集.提交顺序))
    print(f"\n[全部完成] 本地模型最终提交：{final_csv}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断；成功切块已写入本地 SQLite，重新运行即可续跑。")
        raise SystemExit(130)
    except Exception as error:
        print(f"\n本地模型一键生成失败：{error}", file=sys.stderr)
        print("修正后重新运行即可；已成功的本地识别缓存不会丢失。", file=sys.stderr)
        raise SystemExit(1)
