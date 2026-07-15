"""图表分支本地 OCR：把文字框投回 v6 逻辑单元格并确定性生成 HTML。"""

from __future__ import annotations

from bisect import bisect_right
from html import escape
import json
from pathlib import Path
from typing import Any

from ..common.local_ocr import CachedLocalOCR, OCRBox, group_ocr_lines
from ..common.models import Box
from ..common.submission import write_submission


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_text(boxes: list[OCRBox]) -> str:
    """单元格内可能有多行文字；同一行按横坐标连接，多行使用 HTML 换行。"""

    lines = [line.text for line in group_ocr_lines(boxes) if line.text.strip()]
    return "<br/>".join(escape(line) for line in lines)


def _matrix_to_html(cells: list[list[list[OCRBox]]]) -> tuple[str, dict[str, int]]:
    """删除整行/整列完全无字的冗余边界，单元格内部空值仍然保留。"""

    nonempty = [
        (row, column)
        for row, values in enumerate(cells)
        for column, boxes in enumerate(values)
        if boxes
    ]
    if not nonempty:
        return "", {
            "source_rows": len(cells),
            "source_columns": len(cells[0]) if cells else 0,
            "output_rows": 0,
            "output_columns": 0,
        }
    row_indices = sorted({row for row, _ in nonempty})
    column_indices = sorted({column for _, column in nonempty})
    html_rows: list[str] = []
    for row in row_indices:
        values = "".join(
            f"<td>{_cell_text(cells[row][column])}</td>"
            for column in column_indices
        )
        html_rows.append(f"  <tr>{values}</tr>")
    quality = {
        "source_rows": len(cells),
        "source_columns": len(cells[0]) if cells else 0,
        "removed_empty_rows": len(cells) - len(row_indices),
        "removed_empty_columns": (
            len(cells[0]) - len(column_indices) if cells else 0
        ),
        "output_rows": len(row_indices),
        "output_columns": len(column_indices),
    }
    return "<table>\n" + "\n".join(html_rows) + "\n</table>", quality


class LocalTableRecognizer:
    """使用准备清单中的精确行列边界重建表格，不调用视觉大模型。"""

    def __init__(self, ocr: CachedLocalOCR, work_dir: str | Path) -> None:
        self.ocr = ocr
        self.work_dir = Path(work_dir)

    @staticmethod
    def _map_body_center(
        box: OCRBox,
        tile: dict[str, Any],
        rows: list[int],
        columns: list[int],
    ) -> tuple[float, float] | None:
        """把切块坐标还原成原图坐标，并排除重复表头/行名列上下文。"""

        source = Box.from_dict(tile["source_box"])
        header_rows = int(tile.get("header_context_rows", 0))
        stub_columns = int(tile.get("stub_context_columns", 0))
        if header_rows or stub_columns:
            top_height = rows[header_rows] - rows[0]
            left_width = columns[stub_columns] - columns[0]
            if box.center_x < left_width or box.center_y < top_height:
                return None
            return (
                source.x1 + box.center_x - left_width,
                source.y1 + box.center_y - top_height,
            )
        # 整表单块可能等比例缩小。分别使用清单中的实际输出宽高映射，
        # 避免浮点 scale 取整造成最后几列偏移。
        output_width = max(1, int(tile["output_width"]))
        output_height = max(1, int(tile["output_height"]))
        return (
            source.x1 + box.center_x * source.width / output_width,
            source.y1 + box.center_y * source.height / output_height,
        )

    @staticmethod
    def _pixel_tile_ownership(
        tile: dict[str, Any],
        all_tiles: list[dict[str, Any]],
        region_box: Box,
    ) -> Box:
        """用重叠区中线划分像素切块责任区，确保同一文字只回填一次。"""

        source = Box.from_dict(tile["source_box"])
        row = int(tile["row_index"])
        column = int(tile["column_index"])
        by_position = {
            (int(item["row_index"]), int(item["column_index"])): item
            for item in all_tiles
        }
        previous_column = by_position.get((row, column - 1))
        next_column = by_position.get((row, column + 1))
        previous_row = by_position.get((row - 1, column))
        next_row = by_position.get((row + 1, column))
        x1 = (
            region_box.x1
            if previous_column is None
            else round(
                (
                    source.x1
                    + Box.from_dict(previous_column["source_box"]).x2
                )
                / 2
            )
        )
        x2 = (
            region_box.x2
            if next_column is None
            else round((source.x2 + Box.from_dict(next_column["source_box"]).x1) / 2)
        )
        y1 = (
            region_box.y1
            if previous_row is None
            else round((source.y1 + Box.from_dict(previous_row["source_box"]).y2) / 2)
        )
        y2 = (
            region_box.y2
            if next_row is None
            else round((source.y2 + Box.from_dict(next_row["source_box"]).y1) / 2)
        )
        return Box(x1, y1, x2, y2)

    def _recognize_grid_region(
        self,
        manifest_path: Path,
        image_sha256: str,
        region: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        rows = [int(value) for value in region["row_boundaries"]]
        columns = [int(value) for value in region["column_boundaries"]]
        cells: list[list[list[OCRBox]]] = [
            [[] for _ in range(len(columns) - 1)]
            for _ in range(len(rows) - 1)
        ]
        detected_count = 0
        assigned_count = 0
        all_tiles = region["tiles"]
        region_box = Box.from_dict(region["box"])
        for tile in region["tiles"]:
            tile_path = manifest_path.parent / "tiles" / tile["file_name"]
            key = (
                f"table/{image_sha256}/region-{region['index']}/"
                f"{tile['file_name']}"
            )
            boxes = self.ocr.recognize_path(tile_path, key)
            detected_count += len(boxes)
            logical_mode = tile.get("tiling_mode") == "logical_grid"
            row_start = int(tile.get("logical_row_start", 0)) if logical_mode else 0
            row_end = int(tile.get("logical_row_end", len(rows) - 1)) if logical_mode else len(rows) - 1
            column_start = int(tile.get("logical_column_start", 0)) if logical_mode else 0
            column_end = int(tile.get("logical_column_end", len(columns) - 1)) if logical_mode else len(columns) - 1
            ownership = (
                None
                if logical_mode
                else self._pixel_tile_ownership(tile, all_tiles, region_box)
            )
            for box in boxes:
                center = self._map_body_center(box, tile, rows, columns)
                if center is None:
                    continue
                x, y = center
                if ownership is not None and not (ownership.x1 <= x < ownership.x2 and ownership.y1 <= y < ownership.y2):
                    continue
                row = bisect_right(rows, y) - 1
                column = bisect_right(columns, x) - 1
                if not (
                    row_start <= row < row_end
                    and column_start <= column < column_end
                    and 0 <= row < len(cells)
                    and 0 <= column < len(cells[row])
                ):
                    continue
                # 后续需要在单元格内恢复阅读顺序，因此把框同时换算成原图
                # 附近坐标；这里只需保持相对次序，不要求恢复精确宽高。
                cells[row][column].append(
                    OCRBox(
                        box.text,
                        box.confidence,
                        x,
                        y - box.height / 2,
                        x + max(1.0, box.x2 - box.x1),
                        y + box.height / 2,
                    )
                )
                assigned_count += 1
        html, matrix_quality = _matrix_to_html(cells)
        quality = {
            **matrix_quality,
            "region_index": region["index"],
            "grid_source": region.get("grid_source", "unknown"),
            "tiling_mode": region["tiles"][0].get("tiling_mode", "unknown"),
            "detected_ocr_boxes": detected_count,
            "assigned_ocr_boxes": assigned_count,
            "unassigned_ocr_boxes": detected_count - assigned_count,
        }
        return html, quality

    def _recognize_fallback_region(
        self,
        manifest_path: Path,
        image_sha256: str,
        region: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """无逻辑网格时保守输出普通文字行，不伪造表格列结构。"""

        output: list[str] = []
        detected = 0
        for tile in region["tiles"]:
            tile_path = manifest_path.parent / "tiles" / tile["file_name"]
            boxes = self.ocr.recognize_path(
                tile_path,
                f"table-fallback/{image_sha256}/{region['index']}/{tile['file_name']}",
            )
            detected += len(boxes)
            output.extend(line.text for line in group_ocr_lines(boxes) if line.text.strip())
        return "\n\n".join(output), {
            "region_index": region["index"],
            "grid_source": region.get("grid_source", "unavailable"),
            "detected_ocr_boxes": detected,
            "mode": "plain-text-fallback",
        }

    def recognize_manifest(self, manifest_path: str | Path, image_sha256: str) -> str:
        manifest_path = Path(manifest_path)
        manifest = _load_json(manifest_path)
        output: list[str] = []
        quality_dir = manifest_path.parent / "local_ocr_quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        for region in manifest["regions"]:
            tiles = region.get("tiles", [])
            logical = bool(
                tiles
                and len(region.get("row_boundaries", [])) >= 2
                and len(region.get("column_boundaries", [])) >= 2
            )
            if logical:
                text, quality = self._recognize_grid_region(
                    manifest_path, image_sha256, region
                )
            else:
                text, quality = self._recognize_fallback_region(
                    manifest_path, image_sha256, region
                )
            (quality_dir / f"region_{int(region['index']):03d}.json").write_text(
                json.dumps(quality, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if text.strip():
                output.append(text.strip())
        return "\n\n".join(output)

    def recognize_dataset(
        self, dataset_manifest_path: str | Path, output_csv: str | Path
    ) -> dict[str, str]:
        dataset = _load_json(Path(dataset_manifest_path))
        canonical: dict[str, str] = {}
        unique = {
            item["canonical_file_name"]
            for item in dataset["items"]
        }
        for index, item in enumerate(dataset["items"], start=1):
            name = item["canonical_file_name"]
            if name in canonical:
                continue
            print(
                f"[本地图表 OCR {len(canonical) + 1:02d}/{len(unique):02d}] {name}",
                flush=True,
            )
            canonical[name] = self.recognize_manifest(
                item["image_manifest"], item["sha256"]
            )
        results = {
            item["file_name"]: canonical[item["canonical_file_name"]]
            for item in dataset["items"]
        }
        result_dir = self.work_dir / "table_results"
        result_dir.mkdir(parents=True, exist_ok=True)
        for name, text in results.items():
            (result_dir / f"{Path(name).stem}.html").write_text(text, encoding="utf-8")
        write_submission(results, output_csv)
        return results
