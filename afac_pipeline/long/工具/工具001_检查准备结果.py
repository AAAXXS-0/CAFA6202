"""校验长图 prepare-long 产物的覆盖率、安全切口和请求尺寸。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval_gaps(intervals: list[tuple[int, int]], image_height: int) -> list[tuple[int, int]]:
    """计算安全块并集没有覆盖到的纵向区间。"""

    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(intervals):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < image_height:
        gaps.append((cursor, image_height))
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description="校验长图自适应准备清单")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    dataset = load_json(args.manifest)
    unique_items = [item for item in dataset["items"] if item["duplicate_of"] is None]
    total_windows = 0
    total_chunks = 0
    max_chunk_height = 0
    over_limit = 0
    safe_cut_count = 0
    fallback_overlap_count = 0
    label_counts: Counter[str] = Counter()
    images_with_gaps: dict[str, list[tuple[int, int]]] = {}
    image_summaries: list[dict] = []

    for item in unique_items:
        manifest_path = Path(item["image_manifest"])
        manifest = load_json(manifest_path)
        total_windows += len(manifest["windows"])
        label_counts.update(block["label"] for block in manifest["layout_blocks"])
        packs = manifest["request_packs"]
        intervals = [
            (pack["source_box"]["y1"], pack["source_box"]["y2"])
            for pack in packs
        ]
        gaps = interval_gaps(intervals, manifest["image"]["height"])
        if gaps:
            images_with_gaps[item["file_name"]] = gaps

        for pack in packs:
            crop_path = manifest_path.parent / "vlm_requests" / pack["file_name"]
            with Image.open(crop_path) as image:
                width, height = image.size
            max_chunk_height = max(max_chunk_height, height)
            over_limit += int(max(width, height) > 4096)
        total_chunks += len(packs)

        adaptive = manifest.get("adaptive_cutting", {})
        safe_cut_count += int(adaptive.get("safe_cut_count", 0))
        fallback_overlap_count += int(adaptive.get("fallback_overlap_count", 0))
        image_summaries.append(
            {
                "file_name": item["file_name"],
                "windows": len(manifest["windows"]),
                "layout_blocks": len(manifest["layout_blocks"]),
                "blank_bands": len(adaptive.get("blank_bands", [])),
                "safe_cuts": adaptive.get("safe_cut_count", 0),
                "fallback_overlaps": adaptive.get("fallback_overlap_count", 0),
                "request_chunks": len(packs),
            }
        )

    print(
        json.dumps(
            {
                "image_count": dataset["image_count"],
                "unique_image_count": dataset["unique_image_count"],
                "duplicate_reuse_count": dataset["duplicate_reuse_count"],
                "total_windows": total_windows,
                "total_request_chunks": total_chunks,
                "max_chunk_height": max_chunk_height,
                "over_4096": over_limit,
                "safe_cut_count": safe_cut_count,
                "fallback_overlap_count": fallback_overlap_count,
                "layout_labels": dict(label_counts),
                "images_with_coverage_gaps": images_with_gaps,
                "images": image_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
