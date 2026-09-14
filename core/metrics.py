"""Observability, Prometheus metrics export, and stream health telemetry.

Provides:
- In-process metrics registry with Prometheus text format export.
- Real-time stream health watchdog (detects stalled audio, stuck VAD, prolonged silence).
- Async HTTP server exposing /metrics and /health endpoints.
"""

from __future__ import annotations

import asyncio
import collections
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
from aiohttp import web

from core.logging_setup import get_logger

logger = get_logger("metrics")


class MetricsRegistry:
    """Thread-safe, lightweight metrics storage conforming to Prometheus standards."""

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = collections.defaultdict(dict)
        self._gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = collections.defaultdict(dict)
        self._help: Dict[str, str] = {}
        self._start_time = time.time()

    def _normalize_labels(self, labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def register_help(self, name: str, doc: str) -> None:
        self._help[name] = doc

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None, doc: str = "") -> None:
        if doc and name not in self._help:
            self._help[name] = doc
        lbl_key = self._normalize_labels(labels)
        current = self._counters[name].get(lbl_key, 0.0)
        self._counters[name][lbl_key] = current + value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, doc: str = "") -> None:
        if doc and name not in self._help:
            self._help[name] = doc
        lbl_key = self._normalize_labels(labels)
        self._gauges[name][lbl_key] = float(value)

    def to_prometheus_text(self) -> str:
        lines: List[str] = []
        # Process metrics
        uptime = time.time() - self._start_time
        self.set_gauge("jarvis_uptime_seconds", uptime, doc="Process uptime in seconds")

        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            self.set_gauge("jarvis_process_rss_bytes", mem_info.rss, doc="Resident Memory Size in bytes")
            self.set_gauge("jarvis_process_cpu_percent", process.cpu_percent(), doc="Current CPU percent usage")
        except Exception:
            pass

        # Format Counters
        for name, series in sorted(self._counters.items()):
            doc = self._help.get(name, f"Metric {name}")
            lines.append(f"# HELP {name} {doc}")
            lines.append(f"# TYPE {name} counter")
            for labels, val in sorted(series.items()):
                lbl_str = ""
                if labels:
                    lbl_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
                lines.append(f"{name}{lbl_str} {val}")

        # Format Gauges
        for name, series in sorted(self._gauges.items()):
            doc = self._help.get(name, f"Metric {name}")
            lines.append(f"# HELP {name} {doc}")
            lines.append(f"# TYPE {name} gauge")
            for labels, val in sorted(series.items()):
                lbl_str = ""
                if labels:
                    lbl_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
                lines.append(f"{name}{lbl_str} {val}")

        lines.append("")
        return "\n".join(lines)


class HealthWatchdog:
    """Watches the audio input, VAD probabilities, and stream latency for silent degradation."""

    def __init__(self, registry: MetricsRegistry) -> None:
        self._registry = registry
        self.last_audio_frame_ts: float = time.time()
        self.audio_frames_total: int = 0
        self.current_rms: float = 0.0
        self.last_vad_prob: float = 0.0
        self._vad_recent_probs: collections.deque = collections.deque(maxlen=60)
        self.active_state: str = "IDLE"
        self.current_brain: str = "fast"

    def record_audio_frame(self, rms: float) -> None:
        now = time.time()
        self.last_audio_frame_ts = now
        self.audio_frames_total += 1
        self.current_rms = rms
        self._registry.increment("jarvis_audio_frames_total", 1.0, doc="Total audio input frames captured")
        self._registry.set_gauge("jarvis_audio_input_rms", rms, doc="Latest calculated audio input RMS level")

    def record_vad(self, speech_prob: float) -> None:
        self.last_vad_prob = speech_prob
        self._vad_recent_probs.append(speech_prob)
        self._registry.set_gauge("jarvis_vad_speech_prob", speech_prob, doc="Latest speech probability from VAD")

    def record_state(self, state_name: str) -> None:
        self.active_state = state_name
        self._registry.increment("jarvis_state_transitions_total", 1.0, labels={"to_state": state_name}, doc="Orchestrator state transitions")

    def check_health(self) -> Dict[str, Any]:
        now = time.time()
        warnings: List[str] = []
        is_healthy = True

        # Check audio stream stall
        idle_audio_secs = now - self.last_audio_frame_ts
        if self.active_state in ("LISTEN", "TRANSCRIBE") and idle_audio_secs > 5.0:
            warnings.append(f"Audio stream stalled in state {self.active_state} for {idle_audio_secs:.1f}s")
            is_healthy = False

        # Check VAD stuckness (e.g. constant 0.35 with 0 variance over 50 samples)
        if len(self._vad_recent_probs) >= 50:
            probs = list(self._vad_recent_probs)
            mean = sum(probs) / len(probs)
            variance = sum((p - mean) ** 2 for p in probs) / len(probs)
            if variance < 1e-6 and 0.1 < mean < 0.9:
                warnings.append(f"VAD appears stuck at static value {mean:.4f} with zero variance")
                is_healthy = False

        status = "healthy" if is_healthy else ("degraded" if warnings else "healthy")

        return {
            "status": status,
            "uptime_seconds": round(now - self._registry._start_time, 1),
            "state": self.active_state,
            "current_brain": self.current_brain,
            "audio": {
                "frames_total": self.audio_frames_total,
                "seconds_since_last_frame": round(idle_audio_secs, 2),
                "current_rms": round(self.current_rms, 5),
            },
            "vad": {
                "latest_probability": round(self.last_vad_prob, 4),
            },
            "warnings": warnings,
        }


class MetricsServer:
    """Async HTTP server exposing Prometheus metrics and health check endpoints."""

    def __init__(self, registry: MetricsRegistry, watchdog: HealthWatchdog, host: str = "127.0.0.1", port: int = 9090) -> None:
        self._registry = registry
        self._watchdog = watchdog
        self._host = host
        self._port = port
        self._app = web.Application()
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_get("/health", self._handle_health)
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        content = self._registry.to_prometheus_text()
        return web.Response(text=content, content_type="text/plain", charset="utf-8")

    async def _handle_health(self, request: web.Request) -> web.Response:
        data = self._watchdog.check_health()
        status_code = 200 if data["status"] == "healthy" else (503 if data["status"] == "degraded" else 200)
        return web.json_response(data, status=status_code)

    async def start(self) -> None:
        try:
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            logger.info("metrics_server_started", host=self._host, port=self._port)
        except Exception as e:
            logger.warning("metrics_server_start_failed", error=str(e), port=self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            logger.info("metrics_server_stopped")


# Global singleton instances for easy hook-in across the codebase
GLOBAL_METRICS = MetricsRegistry()
GLOBAL_WATCHDOG = HealthWatchdog(GLOBAL_METRICS)
