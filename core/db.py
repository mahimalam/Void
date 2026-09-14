"""Async SQLite helpers for the J.A.R.V.I.S. database.

Every async code path that needs to read or write ``data/jarvis.db``
goes through this module. Sync ``sqlite3.connect()`` calls would
block the asyncio event loop and stutter the voice pipeline, so this
module exposes only ``async`` functions. It uses ``aiosqlite`` (already
a project dependency for ``AuditLog``) so we have a single async
client of the same database file.

Centralising the schema and the common queries here also prevents the
class of bug where one tool writes to a slightly different table
shape than another tool reads from.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from core.logging_setup import get_logger

_log = get_logger("db")


# Shared DDL — every code path that touches daily_summaries uses this.
# Idempotent: ``CREATE TABLE IF NOT EXISTS`` is a no-op when present.
DAILY_SUMMARIES_DDL = """
CREATE TABLE IF NOT EXISTS daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    summary TEXT NOT NULL
)
"""

# M10: cache of directories that have already been created by
# ``_ensure_dir`` in this process. The mkdir syscall is cheap, but
# ``_log`` does it on every ``connect()`` call (which is a lot
# during the periodic compaction path). Memoising saves a few
# hundred syscalls per session.
_ENSURED_DIRS: set[Path] = set()


def _ensure_dir(path: Path) -> None:
    """Idempotently create ``path``'s parent. No-op on subsequent calls."""
    parent = path.parent
    if parent in _ENSURED_DIRS:
        return
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # Race with another worker; the directory exists, that's fine.
        pass
    except OSError as e:
        # Permission denied, etc. Log and continue — the
        # connection will fail later with a clear error.
        _log.warning("ensure_dir_failed", path=str(parent), error=str(e))
        return
    _ENSURED_DIRS.add(parent)


@asynccontextmanager
async def connect(db_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """Open an aiosqlite connection and close it on context exit.

    The caller is expected to issue ``await conn.execute(...)`` /
    ``await conn.commit()`` calls within the context. The parent
    directory of the db file is created if missing so callers don't
    have to pre-create it.
    """
    _ensure_dir(db_path)
    conn = await aiosqlite.connect(str(db_path))
    try:
        yield conn
    finally:
        await conn.close()



async def has_table(db_path: Path, table_name: str) -> bool:
    """True when ``table_name`` exists in the database."""
    async with connect(db_path) as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = await cursor.fetchone()
        return row is not None


async def insert_daily_summary(db_path: Path, summary: str) -> None:
    """Append a summary row. Creates the table if missing."""
    async with connect(db_path) as conn:
        await conn.execute(DAILY_SUMMARIES_DDL)
        await conn.execute(
            "INSERT INTO daily_summaries (summary) VALUES (?)",
            (summary,),
        )
        await conn.commit()


async def fetch_recent_summaries(
    db_path: Path,
    limit: int,
    days: int | None = None,
) -> list[tuple[str, str]]:
    """Return up to ``limit`` summary rows, newest first.

    When ``days`` is set, only rows within that window are returned.
    Each tuple is ``(timestamp, summary)``.
    """
    async with connect(db_path) as conn:
        await conn.execute(DAILY_SUMMARIES_DDL)
        if days is not None:
            cursor = await conn.execute(
                "SELECT timestamp, summary FROM daily_summaries "
                "WHERE timestamp >= datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT ?",
                (f"-{days} days", limit),
            )
        else:
            cursor = await conn.execute(
                "SELECT timestamp, summary FROM daily_summaries "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        return list(await cursor.fetchall())
