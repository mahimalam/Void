"""In-session conversation history.

Stores the last N turns (user + assistant) in a ring buffer and formats
them for injection into Ollama /api/chat messages or as a system prompt
text block.

Cleared when Jarvis stops. NOT cross-session memory — that is Phase 4.

Usage:
    history = ConversationHistory(max_turns=20)
    history.add("user", "What time is it?")
    history.add("assistant", "It's three in the afternoon.")
    messages = history.as_messages()   # for /api/chat
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from core.memory.wal import MemoryWAL


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str
    metadata: dict | None = None


class ConversationHistory:
    """Rolling ring buffer of the last max_turns conversation turns.

    Parameters
    ----------
    max_turns : int
        Maximum number of individual turns (user + assistant) to keep.
        20 turns = 10 full exchanges.
    """

    def __init__(self, max_turns: int = 20, wal: 'MemoryWAL | None' = None) -> None:
        self._max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._wal = wal
        self._pending_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: Literal["user", "assistant"], text: str, metadata: dict | None = None) -> None:
        """Append a turn. Silently ignores empty/whitespace-only text."""
        stripped = text.strip()
        if stripped:
            self._turns.append(Turn(role=role, text=stripped, metadata=metadata))
            if self._wal is not None:
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(self._wal.append_turn(role, stripped, metadata))
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)
                except RuntimeError:
                    # No running event loop
                    pass

    async def load_from_wal(self) -> None:
        """Pre-load the most recent turns from the WAL database into memory on startup."""
        if not self._wal:
            return
        try:
            recent = await self._wal.get_recent_turns(limit=self._max_turns)
            for row in recent:
                role = row.get("role")
                content = row.get("content", "").strip()
                metadata = row.get("metadata")
                if role in ("user", "assistant") and content:
                    self._turns.append(Turn(role=role, text=content, metadata=metadata))
        except Exception:
            pass

    async def drain_wal(self) -> None:
        """Wait for in-flight WAL tasks to commit to disk before process exit."""
        if self._pending_tasks:
            await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)

    def clear(self) -> None:
        """Wipe the history (e.g. after a 'Jarvis, forget this session' command)."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def as_messages(self) -> list[dict[str, str]]:
        """Return history in Ollama /api/chat message format.

        Preserves tool execution summaries in assistant turns so multi-turn
        interactions have clear visibility into actions already taken.
        """
        messages: list[dict[str, str]] = []
        for t in self._turns:
            content = t.text
            if t.role == "assistant" and t.metadata and t.metadata.get("tools"):
                tools_str = "; ".join(t.metadata["tools"])
                content = f"{content}\n[Action: {tools_str}]"
            messages.append({"role": t.role, "content": content})
        return messages

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationHistory(turns={self.turn_count}, max={self._max_turns})"
