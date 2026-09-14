"""Write-Ahead Log (WAL) Engine for Second Brain Memory.

Provides immediate disk durability for all voice turns using an append-only
SQLite table with PRAGMA journal_mode=WAL. This guarantees zero data loss
even if the laptop sleeps, crashes, or shuts down mid-session.
Unprocessed turns are picked up by the Catch-up Engine on next boot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from core.logging_setup import get_logger
from core.db import _ensure_dir

_log = get_logger("memory_wal")

WAL_DDL = """
CREATE TABLE IF NOT EXISTS memory_wal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    processed BOOLEAN DEFAULT 0
)
"""

class MemoryWAL:
    """Write-Ahead Log for immediate conversation durability."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        _ensure_dir(self.db_path)

    async def initialize(self) -> None:
        """Initialize the WAL database and set PRAGMA journal_mode=WAL."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                # Enable Write-Ahead Logging for high-concurrency readers/writers
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA busy_timeout = 5000;")
                await conn.execute(WAL_DDL)
                # Index for fast querying of unprocessed turns
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_wal_processed ON memory_wal(processed);")
                await conn.commit()
            _log.info("memory_wal_initialized", path=str(self.db_path))
        except Exception as e:
            _log.error("memory_wal_init_failed", error=str(e))
            raise

    async def append_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        """Instantly durably write a turn to the log.
        
        Returns the inserted turn ID.
        """
        meta_str = json.dumps(metadata) if metadata else None
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                cursor = await conn.execute(
                    "INSERT INTO memory_wal (role, content, metadata, processed) VALUES (?, ?, ?, 0)",
                    (role, content, meta_str),
                )
                await conn.commit()
                return cursor.lastrowid or 0
        except Exception as e:
            _log.error("memory_wal_append_failed", error=str(e))
            return 0

    async def get_unprocessed_turns(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Fetch uncompacted turns that need to be processed into the Second Brain."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    "SELECT id, timestamp, role, content, metadata FROM memory_wal WHERE processed = 0 ORDER BY id ASC LIMIT ?",
                    (limit,)
                )
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    meta = None
                    if row["metadata"]:
                        try:
                            meta = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            pass
                    results.append({
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "role": row["role"],
                        "content": row["content"],
                        "metadata": meta,
                    })
                return results
        except Exception as e:
            _log.error("memory_wal_fetch_failed", error=str(e))
            return []

    async def mark_turns_processed(self, turn_ids: list[int]) -> None:
        """Mark specific turn IDs as successfully processed by the catch-up engine."""
        if not turn_ids:
            return
        
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                # Chunk to avoid SQLite bind limits if list is huge
                chunk_size = 500
                for i in range(0, len(turn_ids), chunk_size):
                    chunk = turn_ids[i:i + chunk_size]
                    placeholders = ",".join(["?"] * len(chunk))
                    await conn.execute(
                        f"UPDATE memory_wal SET processed = 1 WHERE id IN ({placeholders})",
                        chunk
                    )
                await conn.commit()
                _log.debug("memory_wal_marked_processed", count=len(turn_ids))
        except Exception as e:
            _log.error("memory_wal_mark_failed", error=str(e))

    async def prune_processed_turns(self, retention_days: int = 30) -> int:
        """Prune compacted turns older than retention_days to prevent unbounded WAL disk growth."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                cursor = await conn.execute(
                    "DELETE FROM memory_wal WHERE processed = 1 AND timestamp < datetime('now', ?)",
                    (f"-{retention_days} days",)
                )
                await conn.commit()
                pruned = cursor.rowcount
                if pruned > 0:
                    _log.info("memory_wal_pruned", count=pruned, retention_days=retention_days)
                return pruned
        except Exception as e:
            _log.error("memory_wal_prune_failed", error=str(e))
            return 0

    async def get_recent_turns(self, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch the most recent turns regardless of processed state for session recall."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(
                    "SELECT id, timestamp, role, content, metadata FROM memory_wal ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                rows = await cursor.fetchall()
                results = []
                for row in reversed(rows):
                    meta = None
                    if row["metadata"]:
                        try:
                            meta = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            pass
                    results.append({
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "role": row["role"],
                        "content": row["content"],
                        "metadata": meta,
                    })
                return results
        except Exception as e:
            _log.error("memory_wal_get_recent_failed", error=str(e))
            return []

