"""长图分支本地 OCR：按阅读顺序恢复正文，并复用小模型标题层级。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.local_ocr import CachedLocalOCR, OCRLine, group_ocr_lines
from ..common.models import Box
from ..common.submission import write_submission
from .config import LongConfig
from .步骤001_数据定义 import Heading, LayoutBlock
from .步骤005_大模型请求打包 import RecognitionPack
from .步骤006_全流程调度 import merge_markdown_overlap
from .工具.工具004_旧标题层级分析 import infer_heading_hierarchy


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _layout_block(raw: dict[str, Any]) -> LayoutBlock:
    return LayoutBlock(
        id=str(raw["id"]),
        label=str(raw["label"]),
        box=Box.from_dict(raw["box"]),
        confidence=float(raw["confidence"]),
        source_window=int(raw["source_window"]),
        member_ids=tuple(str(value) for value in raw.get("member_ids", [])),
    )


def _match_heading(
    line: OCRLine,
    pack: RecognitionPack,
    headings: list[Heading],
) -> Heading | None:
    """按纵向重叠匹配标题；标题框略向外放宽以容忍 OCR 框高偏差。"""

    local_y1 = min(box.y1 for box in line.boxes)
    local_y2 = max(box.y2 for box in line.boxes)
    y1 = pack.source_box.y1 + local_y1
    y2 = pack.source_box.y1 + local_y2
    best: Heading | None = None
    best_overlap = 0.0
    for heading in headings:
        margin = max(12.0, heading.box.height * 0.35)
        overlap = max(
            0.0,
            min(y2, heading.box.y2 + margin)
            - max(y1, heading.box.y1 - margin),
        )
        if overlap > best_overlap:
            best = heading
            best_overlap = overlap
    return best


def _match_text_block(
    line: OCRLine,
    pack: RecognitionPack,
    blocks: list[LayoutBlock],
) -> LayoutBlock | None:
    """把正文视觉行归到小模型 Text 框，用于连接同一段落的自动换行。"""

    center_y = pack.source_box.y1 + line.center_y
    candidates = [
        block
        for block in blocks
        if block.label in {"Text", "Caption"}
        and block.box.y1 - 12 <= center_y <= block.box.y2 + 12
    ]
    if not candidates:
        return None
    # 多个保护框重叠时选纵向范围更小者，通常更接近真实段落。
    return min(candidates, key=lambda block: (block.box.height, block.box.width))


def _join_wrapped_lines(parts: list[str]) -> str:
    output = ""
    for part in parts:
        if output and output[-1].isascii() and part[0].isascii() and output[-1].isalnum() and part[0].isalnum():
            output += " "
        output += part
    return output


def _pack_markdown(
    lines: list[OCRLine],
    pack: RecognitionPack,
    headings: list[Heading],
    blocks: list[LayoutBlock] | None = None,
) -> str:
    """把同一逻辑标题的多行 OCR 合并成一个 Markdown 标题。"""

    blocks = blocks or []
    items: list[tuple[str | None, int, str]] = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        heading = _match_heading(line, pack, headings)
        if heading is None:
            text_block = _match_text_block(line, pack, blocks)
            items.append(
                (f"body:{text_block.id}" if text_block else None, 0, text)
            )
        else:
            items.append((f"heading:{heading.id}", heading.level, text))

    output: list[str] = []
    index = 0
    while index < len(items):
        heading_id, level, text = items[index]
        if heading_id is None:
            output.append(text)
            index += 1
            continue
        if heading_id.startswith("body:"):
            parts = [text]
            index += 1
            while index < len(items) and items[index][0] == heading_id:
                parts.append(items[index][2])
                index += 1
            output.append(_join_wrapped_lines(parts))
            continue
        parts = [text]
        index += 1
        while index < len(items) and items[index][0] == heading_id:
            parts.append(items[index][2])
            index += 1
        output.append(f"{'#' * level} {_join_wrapped_lines(parts)}")
    return "\n\n".join(output)


class LocalLongRecognizer:
    """对 prepare-long 生成的原图请求块执行本地 OCR。"""

    def __init__(self, ocr: CachedLocalOCR, work_dir: str | Path) -> None:
        self.ocr = ocr
        self.work_dir = Path(work_dir)

    def recognize_manifest(self, manifest_path: str | Path, image_sha256: str) -> str:
        manifest_path = Path(manifest_path)
        manifest = _load_json(manifest_path)
        config = LongConfig(**manifest["config"])
        blocks = [
            _layout_block(raw)
            for raw in manifest.get("layout_blocks", [])
            if (
                raw.get("label") != "Title"
                or float(raw.get("confidence", 0.0)) >= config.title_confidence
            )
        ]
        _, headings = infer_heading_hierarchy(
            blocks, int(manifest["image"]["width"]), config
        )
        current = ""
        quality: list[dict[str, Any]] = []
        packs = [RecognitionPack.from_dict(raw) for raw in manifest["request_packs"]]
        for index, pack in enumerate(packs, start=1):
            image_path = manifest_path.parent / "vlm_requests" / pack.file_name
            boxes = self.ocr.recognize_path(
                image_path,
                f"long/{image_sha256}/{pack.id}/{pack.file_name}",
            )
            lines = group_ocr_lines(boxes)
            markdown = _pack_markdown(lines, pack, headings, blocks)
            if not current:
                current = markdown
            elif pack.overlap_top != 0:
                current = merge_markdown_overlap(current, markdown)
            else:
                current = current.rstrip() + "\n\n" + markdown.lstrip()
            quality.append(
                {
                    "pack_id": pack.id,
                    "file_name": pack.file_name,
                    "ocr_boxes": len(boxes),
                    "ocr_lines": len(lines),
                    "characters": len(markdown),
                    "mean_confidence": (
                        sum(box.confidence for box in boxes) / len(boxes)
                        if boxes
                        else 0.0
                    ),
                }
            )
            print(
                f"    长图块 {index:02d}/{len(packs):02d}："
                f"{len(boxes)} 框、{len(markdown)} 字符",
                flush=True,
            )
        output_dir = manifest_path.parent / "local_ocr_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "heading_count": len(headings),
                    "headings": [
                        {
                            "id": heading.id,
                            "level": heading.level,
                            "role": heading.role,
                            "box": heading.box.to_dict(),
                        }
                        for heading in headings
                    ],
                    "packs": quality,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return current.strip()

    def recognize_dataset(
        self, dataset_manifest_path: str | Path, output_csv: str | Path
    ) -> dict[str, str]:
        dataset = _load_json(Path(dataset_manifest_path))
        canonical: dict[str, str] = {}
        unique = {item["canonical_file_name"] for item in dataset["items"]}
        for item in dataset["items"]:
            name = item["canonical_file_name"]
            if name in canonical:
                continue
            print(
                f"[本地长图 OCR {len(canonical) + 1:02d}/{len(unique):02d}] {name}",
                flush=True,
            )
            canonical[name] = self.recognize_manifest(
                item["image_manifest"], item["sha256"]
            )
        results = {
            item["file_name"]: canonical[item["canonical_file_name"]]
            for item in dataset["items"]
        }
        result_dir = self.work_dir / "long_results"
        result_dir.mkdir(parents=True, exist_ok=True)
        for name, text in results.items():
            (result_dir / f"{Path(name).stem}.md").write_text(text, encoding="utf-8")
        write_submission(results, output_csv)
        return results
