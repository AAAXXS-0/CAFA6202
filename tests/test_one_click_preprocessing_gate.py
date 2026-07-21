import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "一键生成最终CSV.py"
SPEC = importlib.util.spec_from_file_location("afac_one_click", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法载入一键脚本")
ONE_CLICK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ONE_CLICK)


class OneClickPreprocessingGateTest(unittest.TestCase):
    def _manifests(self, root: Path) -> tuple[Path, Path]:
        long_manifest = root / "long_manifest.json"
        table_manifest = root / "table_manifest.json"
        long_manifest.write_text(
            json.dumps(
                {
                    "image_count": 2,
                    "prepared_image_count": 1,
                    "items": [{"file_name": "good-long.png"}],
                    "preprocessing_failures": [
                        {
                            "canonical_file_name": "bad-long.png",
                            "file_names": ["bad-long.png"],
                            "error_type": "RuntimeError",
                            "error": "测试fatal",
                        }
                    ],
                    "preprocessing_failure_report": str(
                        root / "long_failures.json"
                    ),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        table_manifest.write_text(
            json.dumps(
                {
                    "image_count": 1,
                    "prepared_image_count": 1,
                    "items": [{"file_name": "good-table.png"}],
                    "preprocessing_failures": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return long_manifest, table_manifest

    def _common_patches(
        self,
        long_manifest: Path,
        table_manifest: Path,
        *,
        force_api: bool,
    ):
        long_config = SimpleNamespace(digest=lambda: "long-digest")
        table_config = SimpleNamespace(digest=lambda: "table-digest")
        return [
            patch.object(
                ONE_CLICK,
                "解析参数",
                return_value=argparse.Namespace(force_api=force_api),
            ),
            patch.object(ONE_CLICK, "检查固定文件"),
            patch.object(
                ONE_CLICK.LongConfig,
                "from_json",
                return_value=long_config,
            ),
            patch.object(
                ONE_CLICK.TableConfig,
                "from_json",
                return_value=table_config,
            ),
            patch.object(
                ONE_CLICK,
                "准备长图",
                return_value=long_manifest,
            ),
            patch.object(
                ONE_CLICK,
                "准备图表",
                return_value=table_manifest,
            ),
            patch.object(ONE_CLICK.os, "chdir"),
            patch.dict(
                ONE_CLICK.os.environ,
                {
                    "FINIXDOC_WORKERS": "1",
                    "FINIXDOC_TIMEOUT": "600",
                    "FINIXDOC_MAX_RETRIES": "1",
                },
                clear=True,
            ),
        ]

    def test_default_mode_stops_before_api_when_preprocessing_has_fatal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifests = self._manifests(Path(directory))
            patches = self._common_patches(
                *manifests,
                force_api=False,
            )
            migrate = patch.object(ONE_CLICK, "迁移旧缓存")
            client = patch.object(
                ONE_CLICK.FinixDocClient,
                "from_official_doc",
            )
            active = [item.start() for item in [*patches, migrate, client]]
            try:
                result = ONE_CLICK.main()
                self.assertEqual(result, 1)
                active[-2].assert_not_called()
                active[-1].assert_not_called()
            finally:
                for item in reversed([*patches, migrate, client]):
                    item.stop()

    def test_force_api_recognizes_only_manifest_items_without_final_merge(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifests = self._manifests(Path(directory))
            patches = self._common_patches(
                *manifests,
                force_api=True,
            )
            fake_long = MagicMock()
            fake_table = MagicMock()
            extras = [
                patch.object(ONE_CLICK, "迁移旧缓存"),
                patch.object(
                    ONE_CLICK.FinixDocClient,
                    "from_official_doc",
                    return_value=MagicMock(),
                ),
                patch.object(
                    ONE_CLICK,
                    "LongPipeline",
                    return_value=fake_long,
                ),
                patch.object(
                    ONE_CLICK,
                    "TablePipeline",
                    return_value=fake_table,
                ),
                patch.object(ONE_CLICK, "combine_submissions"),
            ]
            all_patches = [*patches, *extras]
            active = [item.start() for item in all_patches]
            try:
                result = ONE_CLICK.main()
                self.assertEqual(result, 0)
                fake_long.recognize_dataset.assert_called_once()
                fake_table.recognize_dataset.assert_called_once()
                active[-1].assert_not_called()
            finally:
                for item in reversed(all_patches):
                    item.stop()


if __name__ == "__main__":
    unittest.main()
