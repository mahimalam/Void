"""Memory Catch-Up Engine.

Runs opportunistically on boot (or in the background) to process any
turns that were durably written to the WAL but not yet compacted into
the Second Brain Knowledge Graph/Vector DB.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.memory.wal import MemoryWAL
    from core.memory.vector_engine import VectorEngine
    from core.memory.graph import KnowledgeGraph

_log = get_logger("memory_catch_up")

# Module-level concurrency guard to prevent overlapping consolidation runs
_catch_up_lock = asyncio.Lock()

# Comprehensive Privacy / Sensitivity Patterns
# Catches financial data, credentials, PII, medical records, and security codes
SENSITIVE_PATTERNS = [
    # Credentials & Auth
    re.compile(r"\b(?:api[_\s-]?key|password|token|secret|credential|auth[_\s-]?header|private[_\s-]?key|bearer)\b", re.IGNORECASE),
    # Financial & Currency ($12,000, 500 usd, bank account, routing, salary)
    re.compile(r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+\s*(?:usd|dollars|euro|eur|taka|bdt)\b", re.IGNORECASE),
    re.compile(r"\b(?:salary|income|net worth|bank account|routing number|credit card|cvv|debit card|balance)\b", re.IGNORECASE),
    # PII & Identity
    re.compile(r"\b(?:ssn|social security|passport|driver['’]?s license|biometric)\b", re.IGNORECASE),
    # Medical & Health
    re.compile(r"\b(?:medical record|diagnosis|diagnosed|diagnosing|allergic to|allergy|prescription|prescribed|disease|medication)\b", re.IGNORECASE),
    # Security Codes & 2FA
    re.compile(r"\b(?:pin code|alarm code|door code|2fa|otp|passcode)\b", re.IGNORECASE),
]

def classify_sensitivity(text: str, metadata: dict[str, Any] | None = None) -> bool:
    """Classify if content contains private or sensitive data requiring local-only routing."""
    if metadata and (metadata.get("private") or metadata.get("is_local_only")):
        return True
    return any(pattern.search(text) is not None for pattern in SENSITIVE_PATTERNS)

def parse_turn_timestamp(raw_ts: Any) -> datetime:
    """Safely parse SQLite or ISO timestamps into a datetime object."""
    if isinstance(raw_ts, datetime):
        return raw_ts
    if isinstance(raw_ts, str):
        try:
            return datetime.fromisoformat(raw_ts.strip().replace(" ", "T"))
        except Exception:
            pass
    return datetime.now()

async def run_catch_up(wal: 'MemoryWAL', vector_engine: 'VectorEngine', graph: 'KnowledgeGraph') -> None:
    """Read unprocessed turns from the WAL and consolidate them into the Second Brain."""
    if _catch_up_lock.locked():
        _log.debug("catch_up_already_running_skipping")
        return

    async with _catch_up_lock:
        try:
            turns = await wal.get_unprocessed_turns(limit=1000)
            if not turns:
                _log.debug("catch_up_empty")
                return

            _log.info("catch_up_started", count=len(turns))
            
            processed_ids = []
            for turn in turns:
                try:
                    content = turn.get("content", "").strip()
                    if not content:
                        processed_ids.append(turn["id"])
                        continue
                        
                    turn_id = turn["id"]
                    metadata = turn.get("metadata")
                    ts = parse_turn_timestamp(turn.get("timestamp"))
                    
                    # 1. Privacy Sensitivity Classification
                    is_local = classify_sensitivity(content, metadata)
                    
                    # 2. Embed into Vector Engine
                    await vector_engine.add_memory(
                        turn_id=turn_id,
                        text=content,
                        is_local_only=is_local,
                        timestamp=ts
                    )
                    
                    # 3. Entity & Relation Extraction
                    lower_content = content.lower()
                    
                    # Health / Allergy Relations (Permanently Pinned Core Memory)
                    if "allergic to" in lower_content:
                        parts = lower_content.split("allergic to")
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            await graph.add_edge("user", "allergic_to", parts[1].strip().strip(".,!?"), weight=1.0, is_pinned=True)
                    elif "allergy" in lower_content and "to" in lower_content:
                        match = re.search(r"allergy to ([a-z0-9\s]+)", lower_content)
                        if match:
                            await graph.add_edge("user", "allergic_to", match.group(1).strip().strip(".,!?"), weight=1.0, is_pinned=True)

                    # Technology / Asset Usage Relations
                    if "uses" in lower_content:
                        parts = lower_content.split("uses")
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            await graph.add_edge(parts[0].strip(), "uses", parts[1].strip().strip(".,!?"), weight=0.6)
                            
                    # Ownership Relations
                    if "owns" in lower_content:
                        parts = lower_content.split("owns")
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            await graph.add_edge(parts[0].strip(), "owns", parts[1].strip().strip(".,!?"), weight=0.8)

                    # Family / Relationship Relations (Permanently Pinned Core Memory)
                    if "wife" in lower_content or "husband" in lower_content:
                        match = re.search(r"([a-z]+) is my (wife|husband)|my (wife|husband) is ([a-z]+)", lower_content)
                        if match:
                            name = match.group(1) or match.group(4)
                            rel = match.group(2) or match.group(3)
                            if name and rel:
                                await graph.add_edge("user", rel, name.strip(), weight=1.0, is_pinned=True)

                    processed_ids.append(turn_id)
                except Exception as turn_err:
                    _log.error("catch_up_turn_error", turn_id=turn.get("id"), error=str(turn_err))
                
            # Mark processed turns in bulk
            if processed_ids:
                await wal.mark_turns_processed(processed_ids)
                
            _log.info("catch_up_completed", processed_count=len(processed_ids))
            
        except Exception as e:
            _log.error("catch_up_failed", error=str(e))

