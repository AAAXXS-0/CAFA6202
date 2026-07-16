"""顺序测试 FireRed-OCR-2B；整个进程始终只有一个模型实例。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


项目根目录 = Path(__file__).resolve().parent
FireRed环境Python = Path("/home/zero/miniconda3/envs/AFAC_FIRERED/bin/python")
默认输出目录 = 项目根目录 / "work/FireRed单模型测试"


def 切换到FireRed环境() -> None:
    """使用独立 Torch 环境重启自身，不污染 PaddleOCR-VL 环境。"""

    if Path(sys.executable).resolve() == FireRed环境Python.resolve():
        return
    if not FireRed环境Python.is_file():
        raise FileNotFoundError(
            f"FireRed 环境不存在：{FireRed环境Python}"
        )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    os.execve(
        str(FireRed环境Python),
        [str(FireRed环境Python), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def 参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用唯一一份 FireRed-OCR-2B 顺序识别指定图片",
    )
    parser.add_argument("图片", nargs="+", type=Path)
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument(
        "--最大像素",
        type=int,
        default=int(os.environ.get("FIRERED_MAX_PIXELS", str(1024 * 28 * 28))),
    )
    parser.add_argument(
        "--最大输出",
        type=int,
        default=int(os.environ.get("FIRERED_MAX_NEW_TOKENS", "4096")),
    )
    return parser.parse_args()


def main() -> int:
    切换到FireRed环境()
    options = 参数()
    os.chdir(项目根目录)

    from afac_pipeline.common.firered_vl_client import FireRedOCRClient

    images = [path.resolve() for path in options.图片]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到测试图片：\n" + "\n".join(missing))

    options.输出目录.mkdir(parents=True, exist_ok=True)
    print(
        f"[FireRed 单模型测试] {len(images)} 张图片，严格顺序执行",
        flush=True,
    )
    client = FireRedOCRClient(
        max_pixels=options.最大像素,
        table_max_pixels=options.最大像素,
        max_new_tokens=options.最大输出,
        table_max_new_tokens=options.最大输出,
    )
    for index, image in enumerate(images, start=1):
        print(f"[测试 {index:02d}/{len(images):02d}] {image}", flush=True)
        markdown = client.recognize(image)
        output = options.输出目录 / f"{index:02d}_{image.stem}.md"
        output.write_text(markdown, encoding="utf-8")
        print(f"[已保存] {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断，FireRed 模型进程即将释放显存。")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nFireRed 单模型测试失败：{error}", file=sys.stderr)
        raise SystemExit(1)
