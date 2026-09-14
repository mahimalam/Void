"""OpenAI-compatible client for free AI gateways (Kilo Gateway / OpenAI-compatible).

Streams responses directly from Kilo Gateway or any OpenAI-compatible endpoint.
Implements in-tier multi-model failover so free-model rate-limits or timeouts
seamlessly rollover to the next candidate model without dropping the voice loop.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator
import httpx

from core.config import GeminiConfig
from core.errors import GeminiError
from core.logging_setup import get_logger
from core.types import ToolCallRequest

# ── Free Model Priority Pools (from Kilo Gateway) ───────────────────────────
DEFAULT_FAST_MODELS = [
    "nex-agi/nex-n2.5-mini:free",
    "cohere/north-mini-code:free",
    "kilo-auto/free",
    "liquid/lfm-2.5-2.6b:free",
    "stepfun/step-3.7-flash:free",
    "inclusionai/ling-3.0-flash-vl:free",
]

DEFAULT_COMPLEX_MODELS = [
    "cohere/north-mini-code:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nex-agi/nex-n2.5-pro:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]


class OpenAIClient:
    """OpenAI-compatible streaming client with native multi-model failover.

    Uses a single persistent httpx.AsyncClient for all requests so that
    TLS connections are reused across turns instead of re-handshaking on
    every call (saves 100-300 ms per voice turn).
    Call ``await client.close()`` on shutdown to release the connection pool.
    """

    def __init__(self, cfg: GeminiConfig | None = None) -> None:
        self._cfg = cfg or GeminiConfig()
        self._log = get_logger("openai_client")
        self._initialized = False
        self._quota_blocked_until: float = 0.0
        self.preferred_model_type: str = "flash"

        # Persistent HTTP client — created lazily on first request, reused
        self._http_client: httpx.AsyncClient | None = None

        # Determine base URL: config base_url -> env FREE_GATEWAY_URL -> default https://api.kilo.ai/api/gateway/v1
        base_url = getattr(self._cfg, "base_url", None)
        if not base_url or "127.0.0.1:3001" in base_url or "localhost:3001" in base_url:
            base_url = (
                os.environ.get("FREE_GATEWAY_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.kilo.ai/api/gateway/v1"
            )
        self._base_url = base_url.rstrip("/")

        # API key (optional for keyless gateways like Kilo)
        self._api_key = (
            self._cfg.api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("FREE_GATEWAY_API_KEY")
            or ""
        )
        self._compartment_id = os.environ.get("ORACLE_COMPARTMENT_ID")
        self._region = os.environ.get("ORACLE_REGION") or "us-ashburn-1"

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=float(self._cfg.timeout_seconds), write=10.0, pool=5.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            self._log.debug("openai_http_client_created", base_url=self._base_url)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
            self._log.debug("openai_http_client_closed")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._ensure_initialized()

    @property
    def is_quota_blocked(self) -> bool:
        import time
        return time.monotonic() < self._quota_blocked_until

    @property
    def quota_cooldown_remaining_s(self) -> float:
        import time
        return max(0.0, self._quota_blocked_until - time.monotonic())

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # Keyless gateways do not strictly require an API key
        self._initialized = True

    def _resolve_model_candidates(self, model_type: str) -> list[str]:
        """Resolve an ordered list of candidate models to try."""
        if model_type in ("flash", "fast", "simple"):
            preferred = getattr(self._cfg, "model_flash", None)
            candidates = list(DEFAULT_FAST_MODELS)
            if preferred and preferred in candidates:
                candidates.remove(preferred)
                candidates.insert(0, preferred)
            elif preferred:
                candidates.insert(0, preferred)
            return candidates

        elif model_type in ("pro", "smart", "complex"):
            preferred = getattr(self._cfg, "model_pro", None)
            candidates = list(DEFAULT_COMPLEX_MODELS)
            if preferred and preferred in candidates:
                candidates.remove(preferred)
                candidates.insert(0, preferred)
            elif preferred:
                candidates.insert(0, preferred)
            return candidates

        # Specific model name provided
        return [model_type]

    # ------------------------------------------------------------------
    # Streaming text with in-tier candidate failover
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        model_type: str = "flash",
    ) -> AsyncIterator[str]:
        self._ensure_initialized()

        if self.is_quota_blocked:
            remaining = self.quota_cooldown_remaining_s
            raise GeminiError(
                f"Cloud brain quota exhausted or rate limited. Try again in {int(remaining)}s.",
                spoken="",
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        candidates = self._resolve_model_candidates(model_type)
        client = self._get_http_client()
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_err = None
        for model_name in candidates:
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 1024,
            }

            tokens_yielded = 0
            try:
                self._log.debug("trying_cloud_model", model=model_name, engine=model_type)
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(connect=3.5, read=7.0, write=5.0, pool=3.0),
                ) as response:
                    if response.status_code != 200:
                        self._log.warning(
                            "cloud_candidate_http_error",
                            model=model_name,
                            status=response.status_code,
                        )
                        continue

                    in_think = False
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices")
                                if not choices:
                                    continue
                                delta = choices[0].get("delta") or {}
                                content = delta.get("content") or ""
                                if not content:
                                    continue
                                if "<think>" in content:
                                    in_think = True
                                    content = content.split("<think>", 1)[0]
                                if in_think:
                                    if "</think>" in content:
                                        in_think = False
                                        content = content.split("</think>", 1)[1]
                                    else:
                                        content = ""
                                if content:
                                    tokens_yielded += 1
                                    yield content
                            except (json.JSONDecodeError, IndexError, AttributeError):
                                pass

                # If candidate responded and yielded tokens, successfully finish
                if tokens_yielded > 0:
                    return

            except Exception as e:
                self._log.warning(
                    "cloud_candidate_stream_failed",
                    model=model_name,
                    error=str(e),
                )
                last_err = e
                continue

        # If all candidates in this tier failed
        self._log.error("all_cloud_candidates_exhausted", engine=model_type, last_error=str(last_err))
        raise GeminiError(
            f"All {model_type} cloud candidates failed: {last_err}",
            spoken="Cloud brain returned an error.",
        )

    # ------------------------------------------------------------------
    # Streaming with tool calls
    # ------------------------------------------------------------------

    async def stream_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        tool_declarations: list[dict[str, Any]] | None = None,
        model_type: str = "flash",
    ) -> AsyncIterator[str | ToolCallRequest]:

        if not tool_declarations:
            async for token in self.stream(prompt, system_prompt, history, model_type):
                yield token
            return

        self._ensure_initialized()

        if self.is_quota_blocked:
            remaining = self.quota_cooldown_remaining_s
            raise GeminiError(
                f"Cloud brain quota exhausted or rate limited. Try again in {int(remaining)}s.",
                spoken="",
            )

        tools = []
        for decl in tool_declarations:
            tools.append({
                "type": "function",
                "function": {
                    "name": decl["name"],
                    "description": decl.get("description", ""),
                    "parameters": decl.get("parameters", {"type": "object", "properties": {}}),
                },
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        candidates = self._resolve_model_candidates(model_type)
        client = self._get_http_client()
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_err = None
        for model_name in candidates:
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "tools": tools,
                "temperature": 0.7,
                "max_tokens": 1024,
            }

            tokens_yielded = 0
            tool_calls_buffer: dict[int, dict[str, str]] = {}

            try:
                self._log.debug("trying_cloud_tool_model", model=model_name, engine=model_type)
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(connect=3.5, read=7.0, write=5.0, pool=3.0),
                ) as response:
                    if response.status_code != 200:
                        self._log.warning(
                            "cloud_tool_candidate_http_error",
                            model=model_name,
                            status=response.status_code,
                        )
                        continue

                    in_think = False
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices")
                                if not choices:
                                    continue
                                delta = choices[0].get("delta") or {}

                                content = delta.get("content") or ""
                                if content:
                                    if "<think>" in content:
                                        in_think = True
                                        content = content.split("<think>", 1)[0]
                                    if in_think:
                                        if "</think>" in content:
                                            in_think = False
                                            content = content.split("</think>", 1)[1]
                                        else:
                                            content = ""
                                    if content:
                                        tokens_yielded += 1
                                        yield content

                                if "tool_calls" in delta and delta["tool_calls"]:
                                    for tc in delta["tool_calls"]:
                                        idx = tc.get("index", 0)
                                        if idx not in tool_calls_buffer:
                                            tool_calls_buffer[idx] = {"name": "", "arguments": ""}
                                        if "function" in tc:
                                            if "name" in tc["function"]:
                                                tool_calls_buffer[idx]["name"] += tc["function"]["name"]
                                            if "arguments" in tc["function"]:
                                                tool_calls_buffer[idx]["arguments"] += tc["function"]["arguments"]
                            except (json.JSONDecodeError, IndexError, AttributeError):
                                pass

                    # After stream, yield accumulated tool calls
                    if tool_calls_buffer:
                        for tc in tool_calls_buffer.values():
                            try:
                                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                                yield ToolCallRequest(name=tc["name"], arguments=args)
                                tokens_yielded += 1
                            except json.JSONDecodeError:
                                self._log.warning("openai_tool_args_parse_failed", args=tc["arguments"])

                    if tokens_yielded > 0:
                        return

            except Exception as e:
                self._log.warning(
                    "cloud_tool_candidate_failed",
                    model=model_name,
                    error=str(e),
                )
                last_err = e
                continue

        self._log.error("all_cloud_tool_candidates_exhausted", engine=model_type, last_error=str(last_err))
        raise GeminiError(
            f"All {model_type} cloud tool candidates failed: {last_err}",
            spoken="Cloud brain returned an error.",
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> list[float]:
        """Get an embedding vector for the text using Oracle OCI."""
        self._ensure_initialized()

        model_name = getattr(self._cfg, "model_embed", "cohere.embed-v4.0")
        embed_url = f"https://inference.generativeai.{self._region}.oci.oraclecloud.com/20231130/actions/embedText"

        payload = {
            "inputs": [text],
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model_name,
            },
            "compartmentId": self._compartment_id,
        }

        client = self._get_http_client()
        try:
            response = await client.post(
                embed_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            raise GeminiError("No embedding returned from Oracle API.")
        except GeminiError:
            raise
        except Exception as e:
            self._log.error("oracle_embed_error", error=str(e))
            raise GeminiError(f"Embedding failed: {e}") from e
