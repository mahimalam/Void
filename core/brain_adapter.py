"""Brain Adapter interface and concrete implementations.

Provides a unified abstraction over cloud OpenAI-compatible gateways (Kilo, OCI,
OpenRouter) and local Ollama inference. This decouples the brain router and orchestrator
from provider-specific HTTP client APIs and parameter formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from core.logging_setup import get_logger
from core.types import ToolCallRequest


class BaseBrainAdapter(ABC):
    """Abstract base class for all LLM brain adapters."""

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        model_type: str = "fast",
    ) -> AsyncIterator[str]:
        """Stream token-by-token plain text response."""
        ...

    @abstractmethod
    async def stream_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        tool_declarations: list[dict[str, Any]] | None = None,
        model_type: str = "fast",
    ) -> AsyncIterator[str | ToolCallRequest]:
        """Stream response yielding text tokens and/or ToolCallRequest objects."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this brain provider is configured and reachable."""
        ...

    async def close(self) -> None:
        """Release any underlying network connections or resources."""
        pass


class OpenAICompatibleBrainAdapter(BaseBrainAdapter):
    """Adapter for OpenAI-compatible endpoints (Kilo Gateway, Oracle OCI, OpenRouter)."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._log = get_logger("brain_adapter.openai")

    def is_available(self) -> bool:
        if not self._client:
            return False
        if hasattr(self._client, "is_available"):
            return self._client.is_available()
        return True

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        model_type: str = "fast",
    ) -> AsyncIterator[str]:
        if not self._client:
            return
        # Normalize model_type: "pro" or "complex" -> "pro", "flash" or "fast" -> "flash"
        normalized_type = "pro" if model_type in ("pro", "complex") else "flash"
        async for token in self._client.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            model_type=normalized_type,
        ):
            yield token

    async def stream_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        tool_declarations: list[dict[str, Any]] | None = None,
        model_type: str = "fast",
    ) -> AsyncIterator[str | ToolCallRequest]:
        if not self._client:
            return
        normalized_type = "pro" if model_type in ("pro", "complex") else "flash"
        async for item in self._client.stream_with_tools(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            tool_declarations=tool_declarations,
            model_type=normalized_type,
        ):
            yield item

    async def close(self) -> None:
        if self._client and hasattr(self._client, "close"):
            await self._client.close()


class OllamaBrainAdapter(BaseBrainAdapter):
    """Adapter for local Ollama server."""

    def __init__(self, client: Any, cfg: Any = None) -> None:
        self._client = client
        self._cfg = cfg
        self._log = get_logger("brain_adapter.ollama")

    def is_available(self) -> bool:
        return self._client is not None

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        model_type: str = "fast",
    ) -> AsyncIterator[str]:
        if not self._client:
            return
        model_name = self._resolve_model(model_type)
        async for token in self._client.stream(
            model=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
        ):
            yield token

    async def stream_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        tool_declarations: list[dict[str, Any]] | None = None,
        model_type: str = "fast",
    ) -> AsyncIterator[str | ToolCallRequest]:
        # For local Ollama, we yield plain text stream (tools bypassed on <7B models for reliability)
        async for token in self.stream(prompt, system_prompt, history, model_type):
            yield token

    def _resolve_model(self, model_type: str) -> str:
        if self._cfg and hasattr(self._cfg, "brains"):
            if model_type in ("failsafe", "smollm"):
                return self._cfg.brains.brain3_failsafe.name
            return self._cfg.brains.brain1_primary.name
        return "qwen3:4b-instruct" if model_type not in ("failsafe", "smollm") else "smollm3:135m-instruct-q4_K_M"
