"""Brain Router — Cognitive Triad dispatcher with tool support.

Routes each user request based on network availability and complexity:

  - ONLINE  → Gemini Flash as default priority.
            → Classify request difficulty: if high difficulty, route to Gemini Pro.
            → Fallback to local Qwen3:4b if Gemini fails.
  - OFFLINE → Qwen3:4b (local primary model via Ollama).
  - FAILSAFE → SmolLM3:135m (takes over silently on local model crash/VRAM OOM).

Phase 2 additions:
  - Accepts tool_declarations for Gemini native function calling.
  - Accepts tool_schema_text for Ollama prompt injection.
  - Parses Ollama's text output for JSON tool-call blocks.
  - Yields ToolCallRequest objects when a tool call is detected.
"""

from __future__ import annotations

import json
import re
import socket
from enum import Enum
from typing import Any, AsyncIterator

from core.config import BrainsConfig, Config, RoutingMode
from core.errors import GeminiError, GPUOutOfMemoryError, OllamaUnavailableError
from core.llm import LLMClient
from core.logging_setup import get_logger
from core.types import ToolCallRequest
from core.openai_client import OpenAIClient
from core.event_bus import EventType


# RoutingMode is the public enum from core.config. We re-export the four
# valid string values here as constants so the brain router's own docstrings
# stay readable without an import-side leak.
try:
    from core.config import RoutingMode  # type: ignore
except ImportError:  # pragma: no cover - defensive
    RoutingMode = None  # type: ignore


from core.brain_adapter import (
    BaseBrainAdapter,
    OllamaBrainAdapter,
    OpenAICompatibleBrainAdapter,
)


class BrainTier(str, Enum):
    """Abstract cognitive tiers for request routing."""
    FAST = "fast"
    COMPLEX = "complex"
    LOCAL = "local"
    FAILSAFE = "failsafe"


class BrainType(str, Enum):
    GEMINI_FLASH = "gemini_flash"
    GEMINI_PRO = "gemini_pro"
    QWEN = "qwen"
    SMOLLM = "smollm"
    # Modern Tier aliases
    FAST = "gemini_flash"
    COMPLEX = "gemini_pro"
    LOCAL = "qwen"
    FAILSAFE = "smollm"


# Marker string to detect a tool_call block in Ollama text output
_TOOL_CALL_MARKER = '"tool_call"'

# H2: hard cap on the tool-call buffer to prevent unbounded growth if
# the model emits a partial JSON that never closes. Without this, a
# runaway model could OOM the orchestrator. 50 KB is far more than any
# reasonable tool-call JSON; the cap is only there as a safety net.
_MAX_TOOL_CALL_BUFFER = 50 * 1024

# After a tool call is yielded, only buffer for a short trailing-text
# window before giving up. The model is supposed to stop after a
# tool call, so anything beyond 200 chars of trailing text is suspect
# and we flush it as plain text to avoid growing the buffer forever.
_TOOL_CALL_TRAILING_WINDOW = 200


class BrainRouter:
    """Routes user requests to the correct brain and streams the response.

    Parameters
    ----------
    cfg : Config
        Full application config (reads brains section).
    llm : LLMClient
        Shared Ollama client — used for fallback local generation.
    gemini_client : GeminiClient | None
        Cloud brain client.
    bus : EventBus
        For emitting BRAIN_CHANGE events.
    """

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        bus: EventBus,
        gemini_client=None,
    ) -> None:
        self._cfg = cfg
        self._llm = llm
        self._bus = bus
        self._gemini = gemini_client
        self._log = get_logger("brain_router")
        self._active_brain = BrainType.GEMINI_FLASH


        # Adapter Registry
        self._cloud_adapter = OpenAICompatibleBrainAdapter(gemini_client) if gemini_client else None
        self._local_adapter = OllamaBrainAdapter(llm, cfg) if llm else None
        self._adapters: dict[str, BaseBrainAdapter] = {}
        if self._cloud_adapter:
            self._adapters["cloud"] = self._cloud_adapter
            self._adapters["fast"] = self._cloud_adapter
            self._adapters["complex"] = self._cloud_adapter
        if self._local_adapter:
            self._adapters["local"] = self._local_adapter
            self._adapters["failsafe"] = self._local_adapter

    def register_adapter(self, name: str, adapter: BaseBrainAdapter) -> None:
        """Register a custom brain adapter dynamically."""
        self._adapters[name] = adapter

    def get_adapter(self, name: str) -> BaseBrainAdapter | None:
        """Retrieve an adapter by tier or name."""
        return self._adapters.get(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_and_generate(
        self,
        text: str,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        tool_declarations: list[dict[str, Any]] | None = None,
        tool_schema_text: str | None = None,
        force_brain: BrainType | None = None,
        bypass_tools: bool = False,
    ) -> AsyncIterator[str | ToolCallRequest]:
        """Route to the correct brain based on priority and yield response tokens.

        Parameters
        ----------
        text : str
            The user's latest message (used for classification + fallback).
        messages : list[dict[str, str]] | None
            Full conversation history including latest user message.
        system_prompt : str | None
            System instruction injected into all brain calls.
        tool_declarations : list[dict] | None
            Gemini-format tool function declarations (Phase 2).
        tool_schema_text : str | None
            Text-based tool schema for Ollama prompt injection (Phase 2).
        force_brain : BrainType | None
            If set, skip routing and stream from this brain directly. Used
            by the H6 OOM fallback path so the orchestrator can hand off
            to the SmolLM failsafe without re-entering the routing
            decision. ``bypass_tools`` is automatically ``True`` for
            ``SMOLLM`` since the tiny model cannot reliably use tools.
        bypass_tools : bool
            If True, the local brain streams raw tokens without
            tool-call parsing. Set by the OOM fallback path because
            SmolLM cannot reliably emit structured tool calls.

        Yields
        ------
        str | ToolCallRequest
            Individual text tokens or ToolCallRequest objects from the selected brain.
        """
        # 0. Honour an explicit force_brain (H6 OOM fallback path).
        if force_brain is not None:
            async for item in self._stream_force_brain(
                force_brain, text, messages, system_prompt,
                tool_declarations, tool_schema_text, bypass_tools,
            ):
                yield item
            return

        # 1. Inspect config.
        routing_mode = self._cfg.brains.routing_mode
        specialist_cfg = self._cfg.brains.brain2_specialist
        gemini_available = specialist_cfg.enabled and self._gemini is not None

        # 2. Decide whether this request should hit the cloud.
        use_cloud, decision_reason = await self._should_use_cloud(
            routing_mode, gemini_available, text,
        )
        self._log.info(
            "routing_decision",
            mode=routing_mode,
            use_cloud=use_cloud,
            reason=decision_reason,
            text=text[:80],
        )

        # Case A: cloud (Gemini/Grok via Oracle/OpenAI)
        quota_blocked = self._gemini is not None and getattr(self._gemini, "is_quota_blocked", False)
        if use_cloud and quota_blocked:
            remaining = getattr(self._gemini, "quota_cooldown_remaining_s", 0.0)
            self._log.info(
                "routing_skip_cloud_quota_cooldown",
                cooldown_remaining_s=round(remaining, 1),
                text=text[:80],
            )
        if use_cloud and not quota_blocked:
            difficulty = await self._classify_difficulty(text)
            model_type = "pro" if difficulty == "PRO" else "flash"
            self._log.info(
                "routed_to_cloud",
                engine="complex" if model_type == "pro" else "fast",
                model_type=model_type,
                text=text[:80],
            )
            self._switch_brain(
                BrainType.GEMINI_PRO if model_type == "pro" else BrainType.GEMINI_FLASH
            )

            # Build history text for cloud
            history_for_gemini: list[dict[str, str]] | None = None
            if messages and len(messages) > 1:
                history_for_gemini = messages[:-1]

            try:
                # Primary attempt with selected engine
                if tool_declarations:
                    async for item in self._gemini.stream_with_tools(
                        prompt=text,
                        system_prompt=system_prompt,
                        history=history_for_gemini,
                        tool_declarations=tool_declarations,
                        model_type=model_type,
                    ):
                        yield item
                else:
                    async for token in self._gemini.stream(
                        prompt=text,
                        system_prompt=system_prompt,
                        history=history_for_gemini,
                        model_type=model_type,
                    ):
                        yield token
                return
            except Exception as e:
                # Tier 1 Fallback: If Complex Engine failed, smoothly try Fast Engine
                if model_type == "pro":
                    self._log.warning(
                        "complex_engine_failed_falling_back_to_fast",
                        error=str(e),
                        text=text[:80],
                    )
                    self._switch_brain(BrainType.GEMINI_FLASH)
                    try:
                        if tool_declarations:
                            async for item in self._gemini.stream_with_tools(
                                prompt=text,
                                system_prompt=system_prompt,
                                history=history_for_gemini,
                                tool_declarations=tool_declarations,
                                model_type="flash",
                            ):
                                yield item
                        else:
                            async for token in self._gemini.stream(
                                prompt=text,
                                system_prompt=system_prompt,
                                history=history_for_gemini,
                                model_type="flash",
                            ):
                                yield token
                        return
                    except Exception as e_fast:
                        self._log.warning("fast_engine_also_failed", error=str(e_fast))
                        e = e_fast

                # Cloud failure handling -> Tier 2 Fallback to Local Qwen
                if not self._cfg.brains.cloud_fallback_enabled:
                    self._log.warning(
                        "cloud_failure_no_fallback_configured",
                        error=str(e),
                    )
                    return
                self._log.warning(
                    "cloud_engines_failed_falling_back_to_local_qwen",
                    error=str(e),
                )
                # Fall through to local Qwen.

        # Case B: local primary (Qwen)
        self._switch_brain(BrainType.QWEN)
        self._log.info(
            "routing_to_local_primary",
            model=self._cfg.brains.brain1_primary.name,
        )

        try:
            # Always bypass tools for local Qwen — small models (<7B)
            # hallucinate tool calls from garbage transcripts. Tools are
            # only reliable with Gemini's native function calling. When
            # Gemini is unavailable, Qwen answers conversationally only.
            async for item in self._local_stream_with_tools(
                text, messages, system_prompt, tool_schema_text,
                model_name=self._cfg.brains.brain1_primary.name,
                bypass_tools=True,
            ):
                yield item
            return
        except (GPUOutOfMemoryError, OllamaUnavailableError) as e:
            self._log.warning(
                "local_primary_failed_silently_falling_back_to_failsafe",
                error=str(e),
            )
            # Switch silently to SmolLM failsafe
            self._switch_brain(BrainType.SMOLLM)

            async for item in self._local_stream_with_tools(
                text, messages, system_prompt, None,  # No tools for failsafe.
                model_name=self._cfg.brains.brain3_failsafe.name,
                bypass_tools=True,
            ):
                yield item

    async def _should_use_cloud(
        self,
        routing_mode: str,
        gemini_available: bool,
        text: str,
    ) -> tuple[bool, str]:
        """Decide whether this turn should hit the cloud brain."""
        if routing_mode == RoutingMode.LOCAL_ONLY:
            return (False, "mode_local_only_no_cloud")

        # Cloud-eligible modes (local_first, cloud_first, cloud_only).
        if not gemini_available:
            return (False, "gemini_not_configured")
        if not self._network_ok():
            return (False, "network_offline")

        if routing_mode in (RoutingMode.CLOUD_FIRST, RoutingMode.CLOUD_ONLY):
            # Short-circuit: skip the Qwen complexity classifier entirely.
            # Running a local Qwen inference just to confirm "yes use cloud"
            # adds 2-4s of latency on every single turn. cloud_only / cloud_first
            # means we already decided — go straight to Gemini.
            return (True, f"mode_{routing_mode}")

        # local_first: classifier decides.
        # Hard gate: very short or punctuation-only transcripts are almost
        # certainly STT garbage (mic noise, breath, partial word). Never
        # send these to the cloud — they burn quota and confuse the LLM.
        stripped = text.strip().strip(".,!?;:-")
        if len(stripped) < 4:
            return (False, "too_short_likely_noise")

        if _is_obviously_complex(text):
            return (True, "fast_path_obviously_complex")
        complexity = await self._classify_complexity_local(text)
        if complexity == "COMPLEX":
            return (True, "classifier_complex")
        return (False, "classifier_simple")

    async def _stream_force_brain(
        self,
        brain: BrainType,
        text: str,
        messages: list[dict[str, str]] | None,
        system_prompt: str | None,
        tool_declarations: list[dict[str, Any]] | None,
        tool_schema_text: str | None,
        bypass_tools: bool,
    ) -> AsyncIterator[str | ToolCallRequest]:
        """Stream from a specific brain, skipping the routing decision.

        Used by the H6 OOM fallback path. Local brains always have
        ``bypass_tools=True`` because the tiny SmolLM and the small
        fallback Qwen are not reliable tool callers; the orchestrator
        will retry without tools on the next turn.
        """
        self._switch_brain(brain)

        if brain in (BrainType.SMOLLM, BrainType.QWEN):
            model_name = (
                self._cfg.brains.brain3_failsafe.name
                if brain == BrainType.SMOLLM
                else self._cfg.brains.brain1_primary.name
            )
            # SmolLM is too small for reliable tool calling, so it always bypasses tools.
            # Qwen (BrainType.QWEN) supports tool calls via tool_schema_text when not explicitly bypassed.
            local_bypass = True if brain == BrainType.SMOLLM else bypass_tools
            effective_schema = None if local_bypass else tool_schema_text
            async for item in self._local_stream_with_tools(
                text, messages, system_prompt, effective_schema,
                model_name=model_name,
                bypass_tools=local_bypass,
            ):
                yield item
            return

        # Gemini (Flash or Pro)
        model_type = "pro" if brain == BrainType.GEMINI_PRO else "flash"
        history_for_gemini: list[dict[str, str]] | None = None
        if messages and len(messages) > 1:
            history_for_gemini = messages[:-1]
        if tool_declarations and not bypass_tools:
            async for item in self._gemini.stream_with_tools(
                prompt=text,
                system_prompt=system_prompt,
                history=history_for_gemini,
                tool_declarations=tool_declarations,
                model_type=model_type,
            ):
                yield item
        else:
            async for token in self._gemini.stream(
                prompt=text,
                system_prompt=system_prompt,
                history=history_for_gemini,
                model_type=model_type,
            ):
                yield token

    @property
    def active_brain(self) -> BrainType:
        return self._active_brain

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def _load_complexity_classifier_prompt(self) -> str:
        path = self._cfg.resolve(self._cfg.prompts.complexity_classifier)
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self._log.warning("complexity_classifier_prompt_not_found", path=str(path))
            return "Respond with exactly one word: SIMPLE or COMPLEX."

    async def _classify_complexity_local(self, text: str) -> str:
        """Use local Qwen to classify SIMPLE vs COMPLEX."""
        prompt = await self._load_complexity_classifier_prompt()
        try:
            response_text = ""
            async for token in self._llm.stream(
                model=self._cfg.brains.brain1_primary.name,
                prompt=text,
                system_prompt=prompt,
            ):
                response_text += token
                if len(response_text) > 20: # fail fast
                    break
            
            clean_result = response_text.strip().upper()
            if "COMPLEX" in clean_result:
                self._log.info("complexity_classified", result="COMPLEX", text=text[:60])
                return "COMPLEX"
            else:
                self._log.debug("complexity_classified", result="SIMPLE", text=text[:60])
                return "SIMPLE"
        except Exception as e:
            self._log.warning("complexity_classification_failed", error=str(e))
            return "SIMPLE" # Fallback to simple

    async def _classify_difficulty(self, text: str) -> str:
        """Return 'FLASH' or 'PRO'. Instant classification using heuristics.

        2026-06-20: respects ``cfg.brains.brain2_specialist.use_pro``.
        When the user lacks Pro-tier Vertex AI access, every "PRO"
        request would either fail (no auth) or take 3-5x longer
        than Flash on a paid Pro endpoint. The user's logic: pin
        everything to Flash. The Pro path is preserved in code
        (so it can be re-enabled with a config flip) but the
        classifier short-circuits to FLASH when ``use_pro=False``.
        """
        clean_text = text.strip().lower()

        # 2026-06-20: kill switch. When Pro is disabled in config
        # (no access, cost cap, or just wanting Flash-only), never
        # return PRO. The Pro brain code path is still alive in
        # ``_stream_force_brain`` and the Gemini client, so flipping
        # this back to true restores full behaviour.
        if not getattr(self._cfg.brains.brain2_specialist, "use_pro", True):
            self._log.debug("difficulty_classified", result="FLASH", text=text[:60], reason="pro_disabled")
            return "FLASH"

        # Tool results or tool errors should ALWAYS stay on Flash
        if clean_text.startswith("tool '") or clean_text.startswith("tool failed") or clean_text.startswith("tool "):
            self._log.debug("difficulty_classified", result="FLASH", text=text[:60], reason="tool_result")
            return "FLASH"

        # Keywords that trigger the Complex / Smart brain
        pro_triggers = [
            "complex logic", "high difficulty", "system architecture", "deep logic",
            "mathematical proof", "expert mode", "deep analysis",
            "pro brain", "use pro", "write code", "implement algorithm", "debug algorithm"
        ]

        if any(trigger in clean_text for trigger in pro_triggers):
            self._log.info("difficulty_classified", result="PRO", text=text[:60])
            return "PRO"

        self._log.debug("difficulty_classified", result="FLASH", text=text[:60])
        return "FLASH"

    # ------------------------------------------------------------------
    # Local streams with tool-call parsing
    # ------------------------------------------------------------------

    async def _local_stream_with_tools(
        self,
        text: str,
        messages: list[dict[str, str]] | None,
        system_prompt: str | None,
        tool_schema_text: str | None,
        model_name: str,
        bypass_tools: bool = False,
    ) -> AsyncIterator[str | ToolCallRequest]:
        """Stream from a local Ollama model, parsing tool-call JSON blocks.

        If tool_schema_text is provided, it's appended to the system prompt
        so the local model knows what tools are available and the expected
        JSON format. The output is then parsed for tool_call JSON blocks.

        ``bypass_tools=True`` short-circuits tool parsing and the
        tool-schema injection. Used by the H6 OOM fallback path
        (SmolLM and the small fallback Qwen cannot reliably emit
        structured tool calls).
        """
        # If no tools, pass through directly
        if bypass_tools or not tool_schema_text:
            if messages:
                raw_stream = self._llm.chat_stream(
                    model=model_name,
                    messages=messages,
                    system_prompt=system_prompt,
                )
            else:
                raw_stream = self._llm.stream(
                    model=model_name,
                    prompt=text,
                    system_prompt=system_prompt,
                )
            async for token in raw_stream:
                yield token
            return

        # Inject tool schema into system prompt
        effective_prompt = system_prompt or ""
        effective_prompt = f"{effective_prompt}\n\n{tool_schema_text}"

        # Get the raw token stream from Ollama with low temperature and suffix reminder
        if messages:
            # Copy messages list to avoid mutating the original history
            modified_messages = [dict(msg) for msg in messages]
            # Find the last user message and append a reminder suffix
            for i in range(len(modified_messages) - 1, -1, -1):
                if modified_messages[i]["role"] == "user":
                    modified_messages[i]["content"] += (
                        "\n\n[REMINDER: If you need to use a tool, respond with the tool call JSON on its own line:\n"
                        '{"tool_call": {"name": "tool_name", "arguments": {...}}}\n'
                        "Always invoke the tool immediately if the user wants to send a message or perform an action.]"
                    )
                    break
            raw_stream = self._llm.chat_stream(
                model=model_name,
                messages=modified_messages,
                system_prompt=effective_prompt,
                temperature=0.0,
            )
        else:
            prompt_with_reminder = text + (
                "\n\n[REMINDER: If you need to use a tool, respond with the tool call JSON on its own line:\n"
                '{"tool_call": {"name": "tool_name", "arguments": {...}}}\n'
                "Always invoke the tool immediately if the user wants to send a message or perform an action.]"
            )
            raw_stream = self._llm.stream(
                model=model_name,
                prompt=prompt_with_reminder,
                system_prompt=effective_prompt,
                temperature=0.0,
            )

        # With tools: accumulate tokens and detect tool_call JSON
        # H2: this loop has two new safety guards:
        #   1. A hard cap (_MAX_TOOL_CALL_BUFFER = 50 KB). If the
        #      model emits an unterminated JSON, we used to buffer
        #      forever and eventually OOM. We now flush and reset.
        #   2. A trailing-text window (_TOOL_CALL_TRAILING_WINDOW = 200).
        #      After a tool call is yielded, the model should stop.
        #      If it keeps emitting text, we only buffer for 200
        #      chars before flushing as plain text. Without this, a
        #      second tool call in the same response could be
        #      dropped because we keep waiting for the first one to
        #      close.
        buffer = ""
        in_tool_call = False  # True after we've yielded a tool call
        async for token in raw_stream:
            buffer += token

            # H2 (1): hard cap. If the buffer exceeds the cap without
            # a recognisable tool call, flush as plain text and
            # reset. This prevents OOM on a runaway model.
            if len(buffer) > _MAX_TOOL_CALL_BUFFER:
                self._log.warning(
                    "tool_call_buffer_overflow",
                    size=len(buffer),
                    cap=_MAX_TOOL_CALL_BUFFER,
                )
                yield buffer
                buffer = ""
                in_tool_call = False
                continue

            # Check if buffer contains a complete tool_call JSON
            tool_call, json_start, json_end = _try_extract_tool_call(buffer)
            if tool_call is not None:
                # Emit any text before the tool call
                text_before = buffer[:json_start].strip()
                if text_before:
                    yield text_before
                # Yield the tool call request
                yield tool_call
                # Keep any text after the JSON block
                buffer = buffer[json_end:]
                in_tool_call = True
            else:
                if in_tool_call:
                    # H2 (2): trailing-text window. The model is
                    # supposed to stop after a tool call. If it
                    # keeps emitting text, only buffer for the
                    # short trailing window before flushing.
                    if len(buffer) > _TOOL_CALL_TRAILING_WINDOW:
                        yield buffer
                        buffer = ""
                        in_tool_call = False
                else:
                    # No tool call yet — wait for enough buffer to
                    # be sure no tool call is in progress.
                    if len(buffer) > 200 and _TOOL_CALL_MARKER not in buffer:
                        # Safe to flush — no tool call in progress
                        yield buffer
                        buffer = ""

        # Flush remaining buffer
        remaining = buffer.strip()
        if remaining:
            # One final check for tool call
            tool_call, json_start, json_end = _try_extract_tool_call(remaining)
            if tool_call is not None:
                text_before = remaining[:json_start].strip()
                if text_before:
                    yield text_before
                yield tool_call

                # yield any remaining text
                text_after = remaining[json_end:].strip()
                if text_after:
                    yield text_after
            else:
                yield remaining

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _switch_brain(self, brain: BrainType) -> None:
        """Switch active brain and emit BRAIN_CHANGE event if changed."""
        if brain != self._active_brain:
            prev = self._active_brain
            self._active_brain = brain
            self._log.info("brain_switch", from_brain=prev.value, to_brain=brain.value)
            self._bus.publish(
                EventType.BRAIN_CHANGE,
                brain=brain.value,
                prev_brain=prev.value,
            )

    @staticmethod
    def _network_ok() -> bool:
        """Fast TCP probe to check internet connectivity (<1s timeout)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(("8.8.8.8", 53))
            s.close()
            return True
        except (socket.error, OSError):
            return False


# ---------------------------------------------------------------------------
# Helpers — Ollama tool-call JSON parsing
# ---------------------------------------------------------------------------

def _try_extract_tool_call(text: str) -> tuple[ToolCallRequest | None, int, int]:
    """Try to extract a tool_call JSON from text.

    Expected format: {"tool_call": {"name": "tool_name", "arguments": {...}}}
    Returns (ToolCallRequest, start_index, end_index) if found, (None, -1, -1) otherwise.
    """
    marker_idx = text.find(_TOOL_CALL_MARKER)
    if marker_idx == -1:
        return None, -1, -1

    # Find the opening brace before the marker
    start_idx = text.rfind('{', 0, marker_idx)
    if start_idx == -1:
        return None, -1, -1

    try:
        # Use JSONDecoder to parse the JSON object, handling nested braces correctly
        decoder = json.JSONDecoder()
        data, end_idx_relative = decoder.raw_decode(text[start_idx:])
        end_idx = start_idx + end_idx_relative
        
        tool_call = data.get("tool_call", {})
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if not name:
            return None, -1, -1

        return ToolCallRequest(name=name, arguments=arguments), start_idx, end_idx
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        return None, -1, -1


def _is_obviously_complex(text: str) -> bool:
    """Fast heuristic: return True only if the request is likely COMPLEX.

    Called before the LLM classifier to avoid a full Ollama inference
    round-trip for simple greetings, status queries, and short commands.
    Falls through to the LLM for anything ambiguous.
    """
    t = text.strip().lower()

    # Trigger scan runs first so short-but-specific queries like
    # "find the fifa match schedule" (6 words) are not short-circuited
    # to SIMPLE before the trigger list has a chance to match.
    for trigger in _COMPLEX_TRIGGERS:
        if trigger in t:
            return True

    # Short utterances with no matching trigger are almost always simple.
    if len(t.split()) <= 6:
        return False

    return False


# 2026-06-20: hoisted the trigger list to a module-level
# constant. The function now does an early return on first match
# (faster on hot path) and the trigger list is easier to test and
# extend without re-reading the function body.
_COMPLEX_TRIGGERS = (
    "write a",
    "generate a",
    "create a",
    "build a",
    "design a",
    "analyse",
    "analyze",
    "research",
    "explain in detail",
    "step by step",
    "multiple files",
    "entire codebase",
    "architecture",
    "deep dive",
    "comprehensive",
    "essay",
    "report on",
    "summarize the",
    "mathematical",
    "proof",
    "algorithm",
    "refactor",
    "debug this",
    "fix all",
    # Tool-heavy / Web Search triggers. 2026-06-20:
    # expanded so sports schedules, live scores, real-time
    # info, and any explicit "search" verb route to Gemini
    # Flash (much better at tool calling than the 3B Qwen).
    "search the web",
    "google",
    "look up",
    "find out",
    "tell me about",
    "what is the latest",
    "what's the latest",
    "who won",
    "what happened",
    "news about",
    "world cup",
    # Sports / schedules / real-time
    "fifa",
    "match schedule",
    "match fixtures",
    "fixtures",
    "kickoff",
    "kick-off",
    "live score",
    "today's score",
    "tonight's score",
    "premier league",
    "champions league",
    "la liga",
    "serie a",
    "bundesliga",
    "ipl",
    "t20",
    "cricket",
    "espn",
    "cricbuzz",
    "match",
    "score",
    "tournament",
    # Weather / prices / traffic
    "weather",
    "forecast",
    "temperature",
    "stock price",
    "share price",
    "exchange rate",
    "dollar rate",
    "traffic",
    # Time-sensitive nouns
    "right now",
    "as of today",
    "this week",
)
