"""使用轻量离线 OCR 快速生成一份可提交的 AFAC A 榜 CSV。

这个脚本只用于官方 FinixDoc-VL 较慢时快速验证提交和获得基线分数，
不会接入正式的一键流水线。处理时优先复用磁盘上已经成功返回的官方
Markdown；缺失切块才交给 RapidOCR 的本地 CPU 模型。

运行前需要把 RapidOCR 安装到临时目录：

    python3 -m pip install --target /tmp/afac_rapidocr rapidocr_onnxruntime==1.2.3

然后在项目根目录运行：

    PYTHONPATH=/tmp/afac_rapidocr python3 experiments/快速离线OCR/快速生成提交CSV.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
长图总清单 = 项目根目录 / "work/正式运行/长图_174f99c092a2/dataset_manifest.json"
图表总清单 = 项目根目录 / "work/正式运行/图表_bfca2d571b5a/dataset_manifest.json"
提交模板 = 项目根目录 / "finix_ab_A_submit_mock.csv"
输出目录 = 项目根目录 / "outputs/快速提交"
输出CSV = 输出目录 / "finix_ab_A_submit_本地OCR快速版.csv"
缓存目录 = 输出目录 / "离线OCR缓存"


@dataclass
class 文字框:
    """RapidOCR 的一个识别框，以及便于排序的矩形边界。"""

    文本: str
    置信度: float
    左: float
    上: float
    右: float
    下: float

    @property
    def 中心纵坐标(self) -> float:
        return (self.上 + self.下) / 2

    @property
    def 高度(self) -> float:
        return max(1.0, self.下 - self.上)


@dataclass
class 文字行:
    """处于同一视觉行内的一个或多个 OCR 文本框。"""

    文本框: list[文字框]

    @property
    def 上(self) -> float:
        return min(item.上 for item in self.文本框)

    @property
    def 下(self) -> float:
        return max(item.下 for item in self.文本框)

    @property
    def 中心纵坐标(self) -> float:
        return (self.上 + self.下) / 2


def 读取JSON(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def 官方结果有效(path: Path) -> bool:
    """排除空文件和接口偶尔返回的“服务器繁忙”HTML。"""

    if not path.is_file() or path.stat().st_size < 10:
        return False
    head = path.read_text(encoding="utf-8", errors="replace")[:1000].lower()
    return "服务器繁忙" not in head and "<title>error" not in head


def 规范化比较文本(text: str) -> str:
    """只用于相邻块去重，不修改最终输出中的原始字符。"""

    return re.sub(r"[\s#|<>/tdr]+", "", text, flags=re.IGNORECASE)


def 去除相邻重复行(existing: list[str], incoming: list[str]) -> list[str]:
    """删除相邻图片块边缘处完全相同的若干行。

    长图最后一个物理切块可能和下一个切块重叠 200 像素。这里最多比较
    末尾/开头 12 行，只删除连续且完全一致的部分，避免模糊去重误删正文。
    """

    max_overlap = min(12, len(existing), len(incoming))
    for count in range(max_overlap, 0, -1):
        left = [规范化比较文本(x) for x in existing[-count:]]
        right = [规范化比较文本(x) for x in incoming[:count]]
        if left == right and all(left):
            return incoming[count:]
    return incoming


class 本地OCR:
    """对 RapidOCR 做一次初始化，并把原始框整理成视觉文字行。"""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as error:
            raise RuntimeError(
                "没有找到 RapidOCR。请按本文件开头的命令安装到 /tmp，"
                "并设置 PYTHONPATH=/tmp/afac_rapidocr。"
            ) from error

        # 图片块最长边限制为 1600：相比默认 736 能保留更多小号金融文字，
        # 又不会让 CPU 推理和内存占用增长得过于夸张。文档全部是正向截图，
        # 关闭方向分类可以明显加快这份临时基线的生成速度。
        self.engine = RapidOCR(
            det_model_path=None,
            det_limit_side_len=1600,
            det_limit_type="max",
            det_box_thresh=0.35,
            text_score=0.35,
            use_angle_cls=False,
            print_verbose=False,
        )

    def 识别(self, image_path: Path) -> list[文字行]:
        result, _ = self.engine(str(image_path), box_thresh=0.35, text_score=0.35)
        if not result:
            return []

        boxes: list[文字框] = []
        for raw_box, raw_text, raw_score in result:
            text = str(raw_text).strip()
            if not text:
                continue
            xs = [float(point[0]) for point in raw_box]
            ys = [float(point[1]) for point in raw_box]
            boxes.append(
                文字框(text, float(raw_score), min(xs), min(ys), max(xs), max(ys))
            )
        boxes.sort(key=lambda item: (item.中心纵坐标, item.左))

        rows: list[文字行] = []
        for box in boxes:
            # OCR 框顶端会因字体大小略有偏差。以中心纵坐标和框高共同判断，
            # 比固定 10 像素阈值更适合同时处理正文和大号标题。
            best: 文字行 | None = None
            best_distance = float("inf")
            for row in rows[-4:]:
                distance = abs(row.中心纵坐标 - box.中心纵坐标)
                tolerance = max(8.0, 0.55 * max(row.下 - row.上, box.高度))
                if distance <= tolerance and distance < best_distance:
                    best = row
                    best_distance = distance
            if best is None:
                rows.append(文字行([box]))
            else:
                best.文本框.append(box)

        rows.sort(key=lambda row: (row.中心纵坐标, min(x.左 for x in row.文本框)))
        for row in rows:
            row.文本框.sort(key=lambda item: item.左)
        return rows


def 合并普通文字行(row: 文字行) -> str:
    """把同一行中的碎片重新连接；英文数字之间保留一个空格。"""

    output = ""
    for box in row.文本框:
        text = box.文本.strip()
        if not output:
            output = text
            continue
        previous = output[-1]
        first = text[0]
        separator = " " if previous.isascii() and first.isascii() and previous.isalnum() and first.isalnum() else ""
        output += separator + text
    return output


def 读取或写入OCR缓存(
    ocr: 本地OCR,
    image_path: Path,
    cache_path: Path,
) -> list[文字行]:
    """逐切块落盘，意外中断后重跑时无需重新识别已经完成的图片。"""

    if cache_path.is_file():
        raw_rows = 读取JSON(cache_path)
        return [文字行([文字框(**item) for item in row]) for row in raw_rows]

    rows = ocr.识别(image_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        [
            {
                "文本": box.文本,
                "置信度": box.置信度,
                "左": box.左,
                "上": box.上,
                "右": box.右,
                "下": box.下,
            }
            for box in row.文本框
        ]
        for row in rows
    ]
    cache_path.write_text(
        json.dumps(serializable, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def 长图单张识别(
    ocr: 本地OCR,
    image_manifest_path: Path,
    sha256: str,
) -> str:
    manifest = 读取JSON(image_manifest_path)
    prepared_dir = image_manifest_path.parent
    headings = manifest.get("headings", [])
    output_lines: list[str] = []

    for index, request in enumerate(manifest.get("request_packs", [])):
        request_id = request["id"]
        response_path = prepared_dir / "responses" / f"{request_id}.md"
        if 官方结果有效(response_path):
            chunk_lines = response_path.read_text(encoding="utf-8").strip().splitlines()
            source = "官方缓存"
        else:
            image_path = prepared_dir / "vlm_requests" / request["file_name"]
            cache_path = 缓存目录 / "长图" / sha256[:16] / f"{request_id}.json"
            rows = 读取或写入OCR缓存(ocr, image_path, cache_path)
            source_y = float(request["source_box"]["y1"])
            already_used_headings: set[str] = set()
            chunk_lines = []
            for row in rows:
                text = 合并普通文字行(row)
                global_center_y = source_y + row.中心纵坐标
                matched = None
                for heading in headings:
                    box = heading["box"]
                    margin = max(20.0, (box["y2"] - box["y1"]) * 0.35)
                    if (
                        heading["id"] not in already_used_headings
                        and box["y1"] - margin <= global_center_y <= box["y2"] + margin
                    ):
                        matched = heading
                        break
                if matched is not None:
                    text = "#" * int(matched["level"]) + " " + text.lstrip("# ")
                    already_used_headings.add(matched["id"])
                chunk_lines.append(text)
            source = "离线OCR"

        chunk_lines = 去除相邻重复行(output_lines, chunk_lines)
        output_lines.extend(chunk_lines)
        print(
            f"    长图切块 {index + 1:02d}/{len(manifest.get('request_packs', [])):02d} "
            f"{request_id}：{source}，{len(chunk_lines)} 行",
            flush=True,
        )

    return "\n\n".join(line for line in output_lines if line.strip()).strip()


def 图表行转HTML(rows: list[文字行]) -> str:
    """将 OCR 同一视觉行的文本框视为表格单元格，生成 TEDS 可读 HTML。"""

    if not rows:
        return ""
    html_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(box.文本.strip())}</td>" for box in row.文本框 if box.文本.strip()
        )
        if cells:
            html_rows.append(f"  <tr>{cells}</tr>")
    if not html_rows:
        return ""
    return "<table>\n" + "\n".join(html_rows) + "\n</table>"


def 图表单张识别(
    ocr: 本地OCR,
    image_manifest_path: Path,
    sha256: str,
) -> str:
    manifest = 读取JSON(image_manifest_path)
    prepared_dir = image_manifest_path.parent
    chunks: list[str] = []
    all_tiles = [tile for region in manifest.get("regions", []) for tile in region["tiles"]]

    for index, tile in enumerate(all_tiles):
        stem = Path(tile["file_name"]).stem
        response_path = prepared_dir / "responses" / f"{stem}.md"
        if 官方结果有效(response_path):
            markdown = response_path.read_text(encoding="utf-8").strip()
            source = "官方缓存"
        else:
            image_path = prepared_dir / "tiles" / tile["file_name"]
            cache_path = 缓存目录 / "图表" / sha256[:16] / f"{stem}.json"
            rows = 读取或写入OCR缓存(ocr, image_path, cache_path)
            markdown = 图表行转HTML(rows)
            source = "离线OCR"
        if markdown:
            chunks.append(markdown)
        print(
            f"    图表切块 {index + 1:02d}/{len(all_tiles):02d} {stem}：{source}",
            flush=True,
        )
    return "\n\n".join(chunks).strip()


def 识别一个分支(
    ocr: 本地OCR,
    dataset_manifest_path: Path,
    branch: str,
) -> dict[str, str]:
    dataset = 读取JSON(dataset_manifest_path)
    by_sha: dict[str, str] = {}
    results: dict[str, str] = {}
    items = dataset["items"]

    for index, item in enumerate(items):
        sha256 = item["sha256"]
        if sha256 in by_sha:
            markdown = by_sha[sha256]
            print(
                f"[{branch} {index + 1:02d}/{len(items):02d}] {item['file_name']}：复用重复图片",
                flush=True,
            )
        else:
            print(
                f"[{branch} {index + 1:02d}/{len(items):02d}] {item['file_name']}：开始识别",
                flush=True,
            )
            manifest_path = Path(item["image_manifest"])
            if branch == "长图":
                markdown = 长图单张识别(ocr, manifest_path, sha256)
            else:
                markdown = 图表单张识别(ocr, manifest_path, sha256)
            by_sha[sha256] = markdown
        results[item["file_name"]] = markdown
    return results


def 写入提交CSV(results: dict[str, str]) -> None:
    with 提交模板.open("r", encoding="utf-8-sig", newline="") as file:
        template_rows = list(csv.DictReader(file))
    template_names = {row["file_name"] for row in template_rows}
    if set(results) != template_names:
        missing = sorted(template_names - set(results))
        extra = sorted(set(results) - template_names)
        raise RuntimeError(f"结果与模板不一致：缺少 {missing}；多出 {extra}")

    输出目录.mkdir(parents=True, exist_ok=True)
    with 输出CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
        writer.writeheader()
        for row in template_rows:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "ground_truth": results[row["file_name"]],
                }
            )


def main() -> int:
    os.chdir(项目根目录)
    for required in (长图总清单, 图表总清单, 提交模板):
        if not required.is_file():
            raise FileNotFoundError(f"缺少准备结果：{required}")

    ocr = 本地OCR()
    results = 识别一个分支(ocr, 长图总清单, "长图")
    results.update(识别一个分支(ocr, 图表总清单, "图表"))
    写入提交CSV(results)

    empty_count = sum(not text.strip() for text in results.values())
    total_chars = sum(len(text) for text in results.values())
    print(f"\n[完成] {输出CSV}", flush=True)
    print(f"[统计] 100 行中空结果 {empty_count} 行，总字符数 {total_chars}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断；已经完成的离线 OCR 缓存会保留，下次可继续。")
        raise SystemExit(130)
    except Exception as error:
        print(f"\n快速提交生成失败：{error}", file=sys.stderr)
        raise SystemExit(1)
