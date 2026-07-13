"""不重新切图，按现有长图清单估算 VLM 请求打包数量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_image(manifest: dict, max_height: int) -> tuple[int, int]:
    parts = [
        part
        for segment in manifest["segments"]
        for part in segment["parts"]
    ]
    parts.sort(key=lambda part: (part["source_box"]["y1"], part["source_box"]["y2"]))
    groups: list[list[dict]] = []
    for part in parts:
        if not groups:
            groups.append([part])
            continue
        current = groups[-1]
        start = current[0]["source_box"]["y1"]
        end = max(item["source_box"]["y2"] for item in current)
        box = part["source_box"]
        if box["y1"] >= end and box["y2"] - start <= max_height:
            current.append(part)
        else:
            groups.append([part])
    maximum = max(
        max(item["source_box"]["y2"] for item in group)
        - min(item["source_box"]["y1"] for item in group)
        for group in groups
    )
    return len(groups), maximum


def main() -> None:
    parser = argparse.ArgumentParser(description="估算长图 VLM 请求数")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    dataset = load(args.manifest)
    max_height = dataset["config"]["max_vlm_height"]
    rows = []
    total = 0
    maximum = 0
    for item in dataset["items"]:
        if item["duplicate_of"] is not None:
            continue
        count, image_max = estimate_image(load(Path(item["image_manifest"])), max_height)
        total += count
        maximum = max(maximum, image_max)
        rows.append({"file_name": item["file_name"], "estimated_requests": count})
    print(
        json.dumps(
            {
                "unique_images": len(rows),
                "estimated_requests": total,
                "max_request_height": maximum,
                "images": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
