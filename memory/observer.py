"""Weekly Observer Engine (Tier 3).

Reads the daily summaries from the past 7 days from the local SQLite database,
and prompts an LLM to extract habits, preferences, and workflows.
Updates data/workstyle.md with the new insights.

Runs at most once per week (on Sundays). The orchestrator stores the last-run
date in data/.observer_last_run and checks weekday before calling this.
"""

import datetime
from pathlib import Path
from typing import Any

from core.db import fetch_recent_summaries
from core.logging_setup import get_logger

logger = get_logger("memory_observer")


def _should_run_today(last_run_file: Path) -> bool:
    """Return True only if today is Sunday AND we haven't run this week."""
    today = datetime.date.today()
    # Only run on Sundays (weekday() == 6)
    if today.weekday() != 6:
        return False
    if not last_run_file.exists():
        return True
    last_run_str = last_run_file.read_text(encoding="utf-8").strip()
    try:
        last_run = datetime.date.fromisoformat(last_run_str)
    except ValueError:
        return True
    # Don't run more than once on the same Sunday
    return last_run < today


async def run_observer(
    db_path: Path,
    workstyle_path: Path,
    llm_client: Any,
    model_name: str,
    last_run_file: Path | None = None,
    force: bool = False,
) -> None:
    """Run weekly pattern extraction on recent daily summaries.

    Args:
        db_path: Path to the jarvis.db SQLite database.
        workstyle_path: Path to the workstyle.md file.
        llm_client: LLM client — either LLMClient (Ollama) or GeminiClient.
        model_name: Model name for Ollama; ignored when llm_client is Gemini.
        last_run_file: Path to the .observer_last_run marker file.
        force: If True, bypass the Sunday-only check (for testing).
    """
    if last_run_file is not None and not force:
        if not _should_run_today(last_run_file):
            logger.debug("observer_skipped_not_sunday")
            return

    logger.info("observer_started")

    # 1. Fetch summaries from the last 7 days.
    #    C5 fix: was a sync sqlite3 call that blocked the event loop.
    try:
        rows = await fetch_recent_summaries(db_path, limit=1000, days=7)
    except Exception as e:
        logger.error("observer_db_read_failed", error=str(e))
        return

    if not rows:
        logger.info("observer_skipped_no_data")
        return

    summaries_text = "\n\n".join([f"[{row[0]}] {row[1]}" for row in rows])

    # 2. Read existing workstyle
    current_workstyle = ""
    if workstyle_path.exists():
        try:
            current_workstyle = workstyle_path.read_text(encoding="utf-8")
        except Exception:
            pass

    prompt = (
        "You are an AI analyst. Review the following daily summaries of user interactions "
        "from the past week.\n"
        "Identify recurring habits, explicit user preferences, and workflow patterns.\n"
        f"Current Workstyle Profile:\n{current_workstyle}\n\n"
        f"Recent Summaries:\n{summaries_text}\n\n"
        "Generate an updated Workstyle Profile in markdown format. Group insights into logical "
        "sections (e.g., ## File Preferences, ## Communication Style, ## Workflow Patterns). "
        "Keep it concise, bulleted, and factual. Only include the updated markdown."
    )
    system_prompt = (
        "You are a behavioral analyst. Output only the updated markdown profile "
        "without conversational filler."
    )

    try:
        updated_workstyle = ""

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
                updated_workstyle += token
        else:
            async for token in llm_client.stream(
                model=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
            ):
                updated_workstyle += token

        updated_workstyle = updated_workstyle.strip()

        # Strip markdown code fences if the LLM wrapped the output
        if updated_workstyle.startswith("```markdown"):
            updated_workstyle = updated_workstyle[11:]
        if updated_workstyle.startswith("```"):
            updated_workstyle = updated_workstyle[3:]
        if updated_workstyle.endswith("```"):
            updated_workstyle = updated_workstyle[:-3]
        updated_workstyle = updated_workstyle.strip()

        # Write to disk
        timestamp = datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")
        final_content = (
            f"# Jarvis Workstyle Profile — Last updated: {timestamp}\n\n"
            f"{updated_workstyle}"
        )
        workstyle_path.write_text(final_content, encoding="utf-8")
        logger.info("observer_complete", chars=len(final_content))

        # Mark last-run date
        if last_run_file is not None:
            last_run_file.write_text(
                datetime.date.today().isoformat(), encoding="utf-8"
            )

    except Exception as e:
        logger.error("observer_failed", error=str(e))
