"""赛事 CSV 输出。"""

from __future__ import annotations

import csv
from pathlib import Path


def write_submission(results: dict[str, str], output_path: str | Path) -> None:
    """严格输出且仅输出 file_name、ground_truth 两列。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
        writer.writeheader()
        for file_name in sorted(results):
            writer.writerow({"file_name": file_name, "ground_truth": results[file_name]})
