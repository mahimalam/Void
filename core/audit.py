"""SQLite audit log for tool calls.

Every tool invocation is recorded with timestamp, tool name, sanitized
arguments, verification state, result status, error, and duration.

Uses aiosqlite for non-blocking writes from the asyncio event loop.

Usage:
    audit = AuditLog(Path("data/jarvis.db"))
    await audit.initialize()
    await audit.log("read_file", {"path": "/tmp/x"}, "LOCKED", True, "", 42.5)
    await audit.close()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.logging_setup import get_logger


class AuditLog:
    """Persistent SQLite audit log for all tool calls."""

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        tool_name       TEXT    NOT NULL,
        arguments_json  TEXT    NOT NULL DEFAULT '{}',
        verification_state TEXT NOT NULL DEFAULT 'LOCKED',
        success         INTEGER NOT NULL DEFAULT 0,
        error           TEXT    NOT NULL DEFAULT '',
        duration_ms     REAL    NOT NULL DEFAULT 0.0
    );
    """

    _INSERT = """
    INSERT INTO tool_calls (tool_name, arguments_json, verification_state, success, error, duration_ms)
    VALUES (?, ?, ?, ?, ?, ?);
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: Any = None  # aiosqlite.Connection (lazy import)
        self._log = get_logger("audit_log")

    async def initialize(self) -> None:
        """Open the database and create the table if it doesn't exist."""
        try:
            import aiosqlite
        except ImportError as e:
            self._log.error("aiosqlite_not_installed", error=str(e))
            raise RuntimeError(
                "aiosqlite is required for audit logging. Run: pip install aiosqlite"
            ) from e

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.execute(self._CREATE_TABLE)
        await self._db.commit()
        self._log.info("audit_log_initialized", db_path=str(self._db_path))

    async def log(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        verification_state: str,
        success: bool,
        error: str,
        duration_ms: float,
    ) -> None:
        """Record a single tool invocation.

        Parameters
        ----------
        tool_name : str
            Name of the tool that was called.
        arguments : dict
            Tool arguments (sanitized — never log passwords or secrets).
        verification_state : str
            "LOCKED" or "UNLOCKED" at time of call.
        success : bool
            Whether the tool returned success=True.
        error : str
            Error message if the tool failed, empty string otherwise.
        duration_ms : float
            Wall-clock execution time in milliseconds.
        """
        if self._db is None:
            self._log.warning("audit_log_not_initialized")
            return

        # Sanitize arguments — truncate very long values
        sanitized = _sanitize_arguments(arguments)
        args_json = json.dumps(sanitized, ensure_ascii=False, default=str)

        try:
            await self._db.execute(
                self._INSERT,
                (
                    tool_name,
                    args_json,
                    verification_state,
                    1 if success else 0,
                    error[:1000],  # cap error length
                    round(duration_ms, 2),
                ),
            )
            await self._db.commit()
        except Exception as e:
            # Audit logging must never crash the system
            self._log.error("audit_log_write_failed", error=str(e))

    async def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve the most recent audit log entries.

        Returns a list of dicts, newest first.
        """
        if self._db is None:
            return []

        try:
            cursor = await self._db.execute(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self._log.error("audit_log_read_failed", error=str(e))
            return []

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None


def _sanitize_arguments(args: dict[str, Any], max_value_len: int = 500) -> dict[str, Any]:
    """Truncate long argument values to prevent bloated audit logs.

    Sensitive-looking keys (password, secret, token, key) are redacted.
    """
    sanitized: dict[str, Any] = {}
    sensitive_keys = {"password", "secret", "token", "key", "passphrase", "api_key"}

    for k, v in args.items():
        if k.lower() in sensitive_keys:
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, str) and len(v) > max_value_len:
            sanitized[k] = v[:max_value_len] + f"...[truncated {len(v)} chars]"
        else:
            sanitized[k] = v

    return sanitized
