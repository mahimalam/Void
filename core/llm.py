"""Ollama LLM client.

Streams token-by-token responses from a local Ollama-hosted model.
Supports primary → fallback auto-switching on CUDA OOM.

Two streaming APIs:
  stream()      — /api/generate (single prompt + system)
  chat_stream() — /api/chat (message history list, for conversation context)

Usage:
    llm = LLMClient(cfg)
    async for token in llm.stream(prompt, system_prompt):
        ...accumulate tokens...
    async for token in llm.chat_stream(model, messages, system_prompt):
        ...accumulate tokens...
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx

from core.config import OllamaConfig
from core.errors import OllamaUnavailableError, GPUOutOfMemoryError
from core.logging_setup import get_logger


class LLMClient:
    """Async Ollama client with streaming and model probing."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self._cfg = cfg
        self._log = get_logger("llm")
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._cfg.host,
                timeout=httpx.Timeout(self._cfg.request_timeout_seconds),
            )
        return self._client

    # --- Health / probe ---

    async def probe_model(self, model_name: str) -> bool:
        """Check if the model is available (loaded or loadable)."""
        try:
            client = await self._ensure_client()
            resp = await client.post("/api/generate", json={
                "model": model_name,
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": self._cfg.startup_probe_tokens},
            })
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """Return list of locally available models."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except Exception as e:
            self._log.error("list_models_failed", error=str(e))
            return []

    # --- Streaming ---

    async def stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from the model, one at a time.

        Yields individual text tokens as they arrive.
        Raises OllamaUnavailableError or GPUOutOfMemoryError on failure.
        """
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": self._cfg.num_ctx,
                "temperature": temperature if temperature is not None else self._cfg.temperature,
                "keep_alive": self._cfg.keep_alive,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with client.stream("POST", "/api/generate", json=payload) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    self._log.error(
                        "ollama_error",
                        status=resp.status_code,
                        body=error_text.decode(errors="replace")[:500],
                    )
                    if resp.status_code == 404:
                        raise OllamaUnavailableError(
                            f"Model '{model}' not found on Ollama server",
                        )
                    raise OllamaUnavailableError(
                        f"Ollama returned status {resp.status_code}",
                    )

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        error_msg = data["error"]
                        if "out of memory" in error_msg.lower():
                            raise GPUOutOfMemoryError(
                                f"GPU OOM on model {model}: {error_msg}",
                            )
                        raise OllamaUnavailableError(error_msg)

                    token = data.get("response", "")
                    if token:
                        yield token

                    if data.get("done", False):
                        return

        except (GeneratorExit, asyncio.CancelledError, KeyboardInterrupt):
            return  # suppress — let httpx cleanup in finally
        except httpx.ConnectError as e:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self._cfg.host}: {e}",
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaUnavailableError(
                f"Ollama request timed out after {self._cfg.request_timeout_seconds}s",
            ) from e

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream using /api/chat with conversation message history.

        Parameters
        ----------
        model : str
            Ollama model name (e.g. 'qwen3:4b-instruct').
        messages : list[dict[str, str]]
            Full message history including the latest user message.
            Each dict has 'role' ('user' or 'assistant') and 'content'.
        system_prompt : str | None
            Injected as the 'system' role message prepended to messages.
        temperature : float | None
            Override model default temperature.

        Yields
        ------
        str
            Individual text tokens.
        """
        client = await self._ensure_client()

        # Prepend system message if given
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": all_messages,
            "stream": True,
            "options": {
                "num_ctx": self._cfg.num_ctx,
                "temperature": temperature if temperature is not None else self._cfg.temperature,
                "keep_alive": self._cfg.keep_alive,
            },
        }

        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    self._log.error(
                        "ollama_chat_error",
                        status=resp.status_code,
                        body=error_text.decode(errors="replace")[:500],
                    )
                    raise OllamaUnavailableError(
                        f"Ollama /api/chat returned status {resp.status_code}",
                    )

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        error_msg = data["error"]
                        if "out of memory" in error_msg.lower():
                            raise GPUOutOfMemoryError(
                                f"GPU OOM on model {model}: {error_msg}",
                            )
                        raise OllamaUnavailableError(error_msg)

                    # /api/chat returns tokens in message.content
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

                    if data.get("done", False):
                        return

        except (GeneratorExit, asyncio.CancelledError, KeyboardInterrupt):
            return
        except httpx.ConnectError as e:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self._cfg.host}: {e}",
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaUnavailableError(
                f"Ollama chat request timed out after {self._cfg.request_timeout_seconds}s",
            ) from e

    # --- Cleanup ---

    async def cleanup(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
