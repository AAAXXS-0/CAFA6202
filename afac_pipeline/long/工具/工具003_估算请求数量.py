"""不重新切图，按现有长图清单统计实际 VLM 请求数量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="统计长图 VLM 请求数")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    dataset = load(args.manifest)

    rows = []
    total = 0
    maximum = 0
    safe_cuts = 0
    fallback_overlaps = 0
    for item in dataset["items"]:
        if item["duplicate_of"] is not None:
            continue
        manifest = load(Path(item["image_manifest"]))
        packs = manifest["request_packs"]
        count = len(packs)
        image_maximum = max(
            (
                pack["source_box"]["y2"] - pack["source_box"]["y1"]
                for pack in packs
            ),
            default=0,
        )
        adaptive = manifest.get("adaptive_cutting", {})
        total += count
        maximum = max(maximum, image_maximum)
        safe_cuts += int(adaptive.get("safe_cut_count", 0))
        fallback_overlaps += int(adaptive.get("fallback_overlap_count", 0))
        rows.append(
            {
                "file_name": item["file_name"],
                "requests": count,
                "safe_cuts": adaptive.get("safe_cut_count", 0),
                "fallback_overlaps": adaptive.get("fallback_overlap_count", 0),
            }
        )

    print(
        json.dumps(
            {
                "unique_images": len(rows),
                "requests": total,
                "max_request_height": maximum,
                "safe_cuts": safe_cuts,
                "fallback_overlaps": fallback_overlaps,
                "images": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
