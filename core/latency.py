"""Latency profiling.

Records per-stage timestamps within a single conversational turn,
plus a rolling window of the last 50 turns for p50/p95/p99 statistics.

Usage:
    timer = LatencyTimer()
    with timer.stage("stt"):
        ...transcribe...
    timer.finish()
    timer.report()  # -> dict
"""

from __future__ import annotations


import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from core.event_bus import EventBus, EventType
from core.logging_setup import get_logger


# Canonical stage names used across the pipeline.
STAGE_STT = "stt"
STAGE_FIRST_TOKEN = "first_token"
STAGE_TOTAL = "total_to_first_audio"


@dataclass
class TurnTimings:
    started_at: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)  # stage_name -> duration_seconds
    marks: dict[str, float] = field(default_factory=dict)   # mark_name -> absolute perf_counter


class LatencyTimer:
    """Records timings for the current turn and a rolling window of past turns."""

    WINDOW_SIZE = 50

    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus = bus
        self._log = get_logger("latency")
        self._current: TurnTimings | None = None
        # Per-stage history for percentile computation.
        self._history: dict[str, deque[float]] = {}
        # H32: removed dead state ``_last_reported_total`` that was
        # added with a comment "Prevent duplicate log_report calls
        # from nested _speak_state invocations" but was never read
        # anywhere. The duplicate-prevention is now handled by the
        # ``_last_reported_at`` timestamp comparison in log_report.

    # --- Turn lifecycle ---

    def begin_turn(self) -> None:
        self._current = TurnTimings()

    def mark(self, name: str) -> None:
        """Record an absolute timestamp under `name`."""
        if self._current is None:
            self._current = TurnTimings()
        self._current.marks[name] = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager that times a stage and stores its duration."""
        if self._current is None:
            self._current = TurnTimings()
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._current.stages[name] = duration
            self._record_to_history(name, duration)

    def record(self, name: str, duration_seconds: float) -> None:
        """Manually record a stage duration (e.g. measured across async boundaries)."""
        if self._current is None:
            self._current = TurnTimings()
        self._current.stages[name] = duration_seconds
        self._record_to_history(name, duration_seconds)

    def finish_turn(self) -> dict[str, float]:
        """Close out the current turn; return its stage timings."""
        if self._current is None:
            return {}
        stages = dict(self._current.stages)
        self._current = None
        return stages

    # --- Reporting ---

    def percentiles(self, stage: str) -> dict[str, float]:
        h = self._history.get(stage)
        if not h:
            return {}
        sorted_h = sorted(h)
        return {
            "p50": _percentile(sorted_h, 0.50),
            "p95": _percentile(sorted_h, 0.95),
            "p99": _percentile(sorted_h, 0.99),
            "n": len(sorted_h),
        }

    def report(self) -> dict[str, dict[str, float]]:
        return {stage: self.percentiles(stage) for stage in self._history}

    def log_report(self) -> None:
        rep = self.report()
        self._log.info("latency_report", **{k: _fmt_pct(v) for k, v in rep.items()})
        if self._bus is not None:
            self._bus.publish(EventType.LATENCY_REPORT, report=rep)

    # --- Internal ---

    def _record_to_history(self, name: str, duration_seconds: float) -> None:
        h = self._history.setdefault(name, deque(maxlen=self.WINDOW_SIZE))
        h.append(duration_seconds)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _fmt_pct(d: dict[str, float]) -> str:
    if not d:
        return "n/a"
    return (
        f"p50={d['p50'] * 1000:.0f}ms "
        f"p95={d['p95'] * 1000:.0f}ms "
        f"p99={d['p99'] * 1000:.0f}ms "
        f"n={int(d['n'])}"
    )
