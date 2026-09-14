"""Memory tools for J.A.R.V.I.S.

Provides tools to:
  - Add words/names to the STT vocabulary so they're recognised correctly.
  - Recall past session summaries from the memory database (Tier 2 memory).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.db import fetch_recent_summaries, has_table, insert_daily_summary
from tools.registry import ToolResult, tool


@tool(name="add_to_vocabulary", verify=False, category="memory")
async def add_to_vocabulary(word: str, canonical: str | None = None) -> ToolResult:
    """Add a name, technical term, or accent-correction pair to JARVIS's hearing vocabulary.

    Use this when JARVIS consistently mishears a name, acronym, or
    domain term — especially common with the user's Bengali accent
    (e.g. STT hearing "Meskineau" when the user said "match
    schedule", or "JWAV" when the user said "Jarvis").

    The word is stored in data/vocabulary.txt, which the STT engine
    loads as its initial_prompt on every transcribe() call. One
    term per line, lines starting with ``#`` are comments.

    Args:
        word: The term to add. If you specify ``canonical`` as well,
            this is the misheard form (e.g. "Meskineau").
        canonical: Optional corrected/correct form (e.g. "match
            schedule"). When provided, the line is written as
            "word → canonical" which the STT prompt uses to bias
            its predictions toward the canonical form.

    Examples:
        add_to_vocabulary("Antigravity")
        add_to_vocabulary("Meskineau", canonical="match schedule")
        add_to_vocabulary("JWAV", canonical="Jarvis")
    """
    word = word.strip()
    if not word:
        return ToolResult(success=False, error="Word cannot be empty.")

    canonical = canonical.strip() if canonical else None
    if canonical:
        entry = f"{word} -> {canonical}"
    else:
        entry = word

    vocab_path = Path("data/vocabulary.txt")

    def _write() -> str:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        if not vocab_path.exists():
            vocab_path.write_text(
                "# JARVIS custom STT vocabulary. One entry per line.\n"
                "# Format: 'term' or 'misheard -> canonical'.\n"
                "# Lines starting with '#' are comments.\n"
                f"{entry}\n",
                encoding="utf-8",
            )
            return "created"

        # Read existing entries (skip blank lines and comments).
        existing_lines = vocab_path.read_text(encoding="utf-8").splitlines()
        existing_entries: list[str] = []
        for ln in existing_lines:
            stripped = ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            existing_entries.append(stripped)

        # De-dup by the LHS term (so re-adding "Meskineau" with a
        # different canonical just overwrites the old mapping).
        new_word_lower = word.lower()
        kept: list[str] = []
        replaced = False
        for existing in existing_entries:
            lhs = existing.split("->", 1)[0].strip().lower()
            if lhs == new_word_lower:
                replaced = True
                continue  # drop the old entry
            kept.append(existing)
        kept.append(entry)
        # Re-emit the file with the leading comment block + entries.
        body = "\n".join(kept) + "\n"
        vocab_path.write_text(
            "# JARVIS custom STT vocabulary. One entry per line.\n"
            "# Format: 'term' or 'misheard -> canonical'.\n"
            "# Lines starting with '#' are comments.\n"
            + body,
            encoding="utf-8",
        )
        return "replaced" if replaced else "appended"

    try:
        outcome = await asyncio.to_thread(_write)
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to update vocabulary: {e}")

    if outcome == "created":
        return ToolResult(
            success=True,
            data=f"Vocabulary created and '{entry}' added. I will understand it on your next turn.",
        )
    if outcome == "replaced":
        return ToolResult(
            success=True,
            data=f"Updated '{word}' in my vocabulary to '{entry}'. I will understand it on your next turn.",
        )
    return ToolResult(
        success=True,
        data=f"Added '{entry}' to my hearing vocabulary. I will understand it on your next turn.",
    )


_wal_instance = None
_vector_instance = None
_graph_instance = None
_engine_lock = asyncio.Lock()


async def _get_second_brain():
    """Lazy-initialize or retrieve Second Brain engine singletons."""
    global _wal_instance, _vector_instance, _graph_instance
    if _wal_instance is not None and _vector_instance is not None and _graph_instance is not None:
        return _wal_instance, _vector_instance, _graph_instance

    async with _engine_lock:
        data_dir = Path("data")
        if _wal_instance is None:
            try:
                from core.memory.wal import MemoryWAL
                _wal_instance = MemoryWAL(data_dir / "memory_wal.db")
                await _wal_instance.initialize()
            except Exception:
                pass

        if _vector_instance is None:
            try:
                from core.memory.vector_engine import VectorEngine
                _vector_instance = VectorEngine(data_dir / "lancedb")
                await _vector_instance.initialize()
            except Exception:
                pass

        if _graph_instance is None:
            try:
                from core.memory.graph import KnowledgeGraph
                _graph_instance = KnowledgeGraph(data_dir / "jarvis_graph.db")
                await _graph_instance.initialize()
            except Exception:
                pass

        return _wal_instance, _vector_instance, _graph_instance


@tool(name="memorize", verify=False, category="memory")
async def memorize(fact: str) -> ToolResult:
    """Explicitly memorize a fact, preference, or detail for long-term recall.

    Saves the information into the Second Brain vector database, knowledge graph,
    and memory WAL for instant semantic recall in future sessions.
    """
    clean_fact = fact.strip()
    if not clean_fact:
        return ToolResult(success=False, error="Fact cannot be empty.")

    db_path = Path("data/jarvis.db")

    try:
        # 1. Store in legacy sqlite daily_summaries for backward compatibility and observer
        await insert_daily_summary(db_path, f"[Explicit Memory] {clean_fact}")

        # 2. Store in Second Brain WAL + Vector Database + Knowledge Graph
        wal, vector_engine, graph = await _get_second_brain()
        
        turn_id = None
        if wal is not None:
            turn_id = await wal.append_turn("user", clean_fact, metadata={"explicit": True})
            if turn_id is not None:
                await wal.mark_turns_processed([turn_id])

        if vector_engine is not None:
            from datetime import datetime
            tid = turn_id if turn_id is not None else int(datetime.now().timestamp() * 1000) % 2147483647
            await vector_engine.add_memory(
                turn_id=tid,
                text=clean_fact,
                is_local_only=False,
                timestamp=datetime.now(),
            )

        if graph is not None:
            # Extract basic subject-relation tokens into graph
            words = [w.strip(".,!?:;\"'()[]{}").lower() for w in clean_fact.split()]
            stop = {"the", "a", "an", "is", "are", "was", "were", "my", "i", "to", "of", "and", "in", "that", "it"}
            meaningful = [w for w in words if len(w) > 2 and w not in stop]
            for w in meaningful[:5]:
                await graph.add_entity(w, entity_type="concept")
            if len(meaningful) >= 2:
                await graph.add_relation(meaningful[0], meaningful[1], relation_type="associated_with")

        return ToolResult(success=True, data=f"Fact memorized successfully: \"{clean_fact}\"")
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to memorize fact: {e}")


@tool(name="recall_memory", verify=False, category="memory")
async def recall_memory(query: str, days: int = 7) -> ToolResult:
    """Search past session memories and Second Brain knowledge.

    Performs hybrid semantic search (LanceDB vectors on GPU), knowledge graph
    spreading activation, and historical session recall.

    Args:
        query: Keyword, question, or concept to search for in memory.
        days:  How many days back to search in session summaries (default 7, max 90).
    """
    clean_query = query.strip()
    if not clean_query:
        return ToolResult(success=False, error="Query cannot be empty.")

    days = max(1, min(days, 90))
    recalled_items: list[str] = []
    seen_texts: set[str] = set()

    # 1. Semantic search in LanceDB (GPU Accelerated)
    try:
        _, vector_engine, graph = await _get_second_brain()
        if vector_engine is not None:
            vector_results = await vector_engine.search(clean_query, limit=5, include_local_only=True, max_distance=1.2)
            for res in vector_results:
                txt = res.get("text", "").strip()
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    score_info = f" (relevance: {1.0 - res.get('_distance', 0.0):.2f})" if "_distance" in res else ""
                    recalled_items.append(f"[Semantic Match{score_info}]\n{txt}")

        if graph is not None:
            tokens = [w.strip(".,!?:;\"'()[]{}").lower() for w in clean_query.split()]
            start_nodes = [t for t in tokens if len(t) > 2]
            if start_nodes:
                graph_results = await graph.spread_activation(start_nodes[:4], max_hops=2, threshold=0.3)
                for res in graph_results[:3]:
                    node = res["node"]
                    conns = ", ".join(res["connections"][:3])
                    if node not in seen_texts:
                        recalled_items.append(f"[Concept Association]\n{node} linked to: {conns}")
    except Exception:
        pass

    # 2. Daily summaries fallback in SQLite
    db_path = Path("data/jarvis.db")
    if db_path.exists():
        try:
            if await has_table(db_path, "daily_summaries"):
                rows = await fetch_recent_summaries(db_path, limit=10_000, days=days)
                query_lower = clean_query.lower()
                for ts, summary in rows:
                    if query_lower in summary.lower():
                        stripped_s = summary.strip()
                        if stripped_s not in seen_texts:
                            seen_texts.add(stripped_s)
                            recalled_items.append(f"[Session Summary - {ts}]\n{stripped_s}")
        except Exception:
            pass

    if not recalled_items:
        return ToolResult(
            success=True,
            data=f"No memories matching '{clean_query}' found.",
        )

    results = "\n\n".join(recalled_items[:6])
    return ToolResult(
        success=True,
        data=f"Found {len(recalled_items)} memory match(es):\n\n{results}",
    )


@tool(name="list_memory_summaries", verify=False, category="memory")
async def list_memory_summaries(days: int = 7) -> ToolResult:
    """List all session summaries from the last N days.

    Returns a chronological list of all daily session summaries from
    long-term memory, without keyword filtering.

    Args:
        days: How many days back to retrieve (default 7, max 90).
    """
    days = max(1, min(days, 90))
    db_path = Path("data/jarvis.db")

    if not db_path.exists():
        return ToolResult(
            success=False, error="Memory database not found. No sessions recorded yet."
        )

    try:
        if not await has_table(db_path, "daily_summaries"):
            return ToolResult(
                success=True,
                data="No memory summaries exist yet.",
            )

        # C5 fix: was a sync sqlite3 call that blocked the event loop.
        rows = await fetch_recent_summaries(db_path, limit=10_000, days=days)
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to list memories: {e}")

    if not rows:
        return ToolResult(
            success=True, data=f"No summaries found in the last {days} days."
        )

    results = "\n\n---\n\n".join(
        [f"**{ts}**\n{summary}" for ts, summary in rows[:10]]
    )
    return ToolResult(
        success=True,
        data=f"{len(rows)} session(s) in the last {days} days:\n\n{results}",
    )
