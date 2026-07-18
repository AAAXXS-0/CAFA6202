"""SQLite 结果缓存。

缓存分为图片级和切片级：图片级缓存用于 SHA-256 重复图片；切片级缓存
用于 API 失败后断点续跑，以及在重新聚合时避免重复消耗 FinixDoc-VL 调用。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator


class ResultCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=60.0)
        # 6 个识别线程可能同时命中缓存。WAL 允许读写并行，busy_timeout
        # 让短暂的写锁排队，而不是把已成功的 API 结果报成 database locked。
        connection.execute("PRAGMA busy_timeout=60000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS image_results (
                    image_sha256 TEXT NOT NULL,
                    config_digest TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (image_sha256, config_digest)
                );

                CREATE TABLE IF NOT EXISTS tile_results (
                    cache_key TEXT PRIMARY KEY,
                    markdown TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def tile_key(image_bytes: bytes, prompt: str, model: str) -> str:
        digest = hashlib.sha256()
        digest.update(image_bytes)
        digest.update(b"\0")
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(model.encode("utf-8"))
        return digest.hexdigest()

    def get_image(self, image_sha256: str, config_digest: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT markdown FROM image_results WHERE image_sha256=? AND config_digest=?",
                (image_sha256, config_digest),
            ).fetchone()
        return row[0] if row else None

    def put_image(
        self,
        image_sha256: str,
        config_digest: str,
        markdown: str,
        metadata: dict,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO image_results
                (image_sha256, config_digest, markdown, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    image_sha256,
                    config_digest,
                    markdown,
                    json.dumps(metadata, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_tile(self, cache_key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT markdown FROM tile_results WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        return row[0] if row else None

    def put_tile(self, cache_key: str, markdown: str, metadata: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO tile_results
                (cache_key, markdown, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cache_key,
                    markdown,
                    json.dumps(metadata, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def merge_result_caches(
    destination: str | Path,
    sources: Iterable[str | Path],
) -> dict[str, int]:
    """把旧工作目录的结果合并到新 SQLite，不覆盖已有成功结果。

    切片缓存键同时包含图片字节、请求内容和模型签名，因此只有完全相同的
    切片才会复用；旧的大块或其他模型结果不会误命中。
    """

    destination_path = Path(destination)
    ResultCache(destination_path)
    columns = {
        "image_results": (
            "image_sha256", "config_digest", "markdown", "metadata_json", "created_at"
        ),
        "tile_results": (
            "cache_key", "markdown", "metadata_json", "created_at"
        ),
    }
    inserted = {table: 0 for table in columns}
    with sqlite3.connect(destination_path, timeout=60.0) as output:
        output.execute("PRAGMA busy_timeout=60000")
        for source in sources:
            source_path = Path(source)
            if not source_path.is_file():
                continue
            if source_path.resolve() == destination_path.resolve():
                continue
            with sqlite3.connect(source_path, timeout=60.0) as old:
                old.execute("PRAGMA busy_timeout=60000")
                for table, names in columns.items():
                    name_list = ", ".join(names)
                    placeholders = ", ".join("?" for _ in names)
                    rows = old.execute(
                        f"SELECT {name_list} FROM {table} ORDER BY created_at DESC"
                    ).fetchall()
                    before = output.total_changes
                    output.executemany(
                        f"INSERT OR IGNORE INTO {table} ({name_list}) "
                        f"VALUES ({placeholders})",
                        rows,
                    )
                    inserted[table] += output.total_changes - before
    return inserted
