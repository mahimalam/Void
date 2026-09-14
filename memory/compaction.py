"""Memory Compaction Engine (Tier 2).

Reads the active session's conversation history and prompts an LLM to generate
a 3-sentence summary and 3 topic tags. Stores this in the local SQLite database.
Called automatically when the orchestrator shuts down (user says 'Jarvis, sleep'
or closes the app).
"""

import asyncio
from pathlib import Path
from typing import Any

from core.db import insert_daily_summary
from core.logging_setup import get_logger

logger = get_logger("memory_compaction")

# H9: a process-wide lock that serialises compaction. The orchestrator
# fires compaction both on every Nth turn (periodic) and on shutdown;
# without this lock the two paths can run in parallel and step on each
# other's insert. The lock is held for the full LLM call + DB write.
_compaction_lock: asyncio.Lock = asyncio.Lock()


async def run_compaction(
    history_messages: list[dict[str, str]],
    db_path: Path,
    llm_client: Any,
    model_name: str,
) -> None:
    """Run daily compaction on the provided conversation history.

    Args:
        history_messages: List of dicts with 'role' and 'content'.
        db_path: Path to the jarvis.db SQLite database.
        llm_client: LLM client — either LLMClient (Ollama) or GeminiClient.
        model_name: Model name for Ollama; ignored when llm_client is Gemini.
    """
    if not history_messages:
        logger.debug("compaction_skipped_empty_history")
        return

    # H9: serialise against any concurrent compaction. If another
    # compaction is in flight, wait for it to finish and then skip
    # our own run (the next periodic tick or the shutdown path will
    # pick up the latest history). Waiting-and-skipping is the
    # safer policy than waiting-and-redoing, because the LLM call
    # is expensive.
    if _compaction_lock.locked():
        logger.info("compaction_skipped_already_in_flight")
        return

    async with _compaction_lock:
        await _run_compaction_locked(
            history_messages, db_path, llm_client, model_name
        )


async def _run_compaction_locked(
    history_messages: list[dict[str, str]],
    db_path: Path,
    llm_client: Any,
    model_name: str,
) -> None:
    """Inner compaction body. Caller must hold ``_compaction_lock``."""
    logger.info("compaction_started", turns=len(history_messages))

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_messages]
    )

    prompt = (
        "Summarize the following conversation session into exactly 3 concise sentences "
        "capturing the most important factual details, decisions, or preferences expressed "
        "by the user. Then, on a new line, provide exactly 3 comma-separated topic tags.\n\n"
        f"Conversation:\n{history_text}\n\nSummary and Tags:"
    )
    system_prompt = "You are a meticulous archivist. Extract only facts and preferences."

    try:
        summary_text = ""

        # GeminiClient.stream() uses model_type= not model=
        # LLMClient.stream() uses model= as positional keyword
        # H34: model_type is no longer hardcoded to "flash". We
        # honour ``llm_client.preferred_model_type`` if the client
        # exposes one (set from ``ollama.compaction_brain`` in
        # config) and fall back to "flash" for clients that don't.
        is_gemini = hasattr(llm_client, "stream_with_tools")
        if is_gemini:
            model_type = getattr(llm_client, "preferred_model_type", "flash")
            async for token in llm_client.stream(
                prompt=prompt,
                system_prompt=system_prompt,
                model_type=model_type,
            ):
                summary_text += token
        else:
            async for token in llm_client.stream(
                model=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
            ):
                summary_text += token

        summary_text = summary_text.strip()
        # C5 fix: was a sync sqlite3 call that blocked the event loop.
        await insert_daily_summary(db_path, summary_text)
        logger.info("compaction_complete", chars=len(summary_text))

    except Exception as e:
        logger.error("compaction_failed", error=str(e))
