"""Streaming — token buffer → sentence boundary detection → TTS chunk splitter.

Accumulates tokens from the LLM stream, detects sentence boundaries
(., ?, !, ...), and yields complete sentence chunks for TTS synthesis.
This enables JARVIS to start speaking while the LLM is still generating.

Usage:
    splitter = SentenceSplitter(cfg)
    async for chunk in splitter.process(token_stream):
        await tts.speak(chunk)
"""

from __future__ import annotations

import re
from typing import AsyncIterator

from core.config import StreamingConfig


class SentenceSplitter:
    """Accumulates tokens and splits on sentence boundaries for streaming TTS."""

    def __init__(self, cfg: StreamingConfig) -> None:
        self._cfg = cfg
        self._buffer = ""
        self._terminator_pattern = re.compile(cfg.sentence_terminators)

    async def process(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """Process a stream of tokens, yielding complete sentence chunks.

        Args:
            token_stream: Async iterator yielding individual tokens from the LLM.

        Yields:
            Complete sentence strings ready for TTS synthesis.
        """
        async for token in token_stream:
            if not token:
                continue

            self._buffer += token

            # Check if buffer contains a complete sentence
            while len(self._buffer) >= self._cfg.max_chunk_chars or (
                len(self._buffer) >= self._cfg.min_chunk_chars
                and self._terminator_pattern.search(self._buffer)
            ):
                # Find the last sentence boundary within the buffer
                chunk = self._pop_sentence()
                if chunk:
                    chunk = chunk.strip()
                    if chunk:
                        yield chunk

        # Flush remaining buffer
        remaining = self._buffer.strip()
        if remaining:
            yield remaining
        self._buffer = ""

    def _pop_sentence(self) -> str:
        """Extract the first complete sentence from the buffer.

        Finds sentence boundaries (., ?, !, ...) at or after
        min_chunk_chars. If no boundary is found but the buffer exceeds
        max_chunk_chars, hard-splits to prevent infinite loops.
        Returns the sentence including its terminator (or the hard-split
        chunk), and removes it from the buffer.
        """
        if len(self._buffer) < self._cfg.min_chunk_chars:
            return ""

        matches = list(self._terminator_pattern.finditer(self._buffer))

        best_idx = -1
        for m in matches:
            term_end = m.end()
            if term_end <= self._cfg.max_chunk_chars:
                best_idx = term_end

        if best_idx == -1:
            if len(self._buffer) >= self._cfg.max_chunk_chars:
                # H39: hard-split at the cap. The previous code
                # returned the first ``max_chunk_chars`` characters
                # verbatim, which could land mid-word ("Speak" vs
                # "ing flu..."). Back up to the last whitespace
                # within the cap so the chunk ends on a clean
                # word boundary. If no whitespace is found inside
                # the cap, fall back to the hard split (better to
                # speak half a word than to silently drop text).
                hard_end = self._cfg.max_chunk_chars
                window = self._buffer[:hard_end]
                last_space = window.rfind(" ")
                if last_space > 0:
                    # Include the trailing space in the chunk so
                    # the spoken boundary has a natural pause and
                    # the remaining buffer starts with the next
                    # word, not another space.
                    chunk = self._buffer[:last_space + 1]
                    self._buffer = self._buffer[last_space + 1:]
                    return chunk
                chunk = self._buffer[:hard_end]
                self._buffer = self._buffer[hard_end:]
                return chunk
            return ""

        chunk = self._buffer[:best_idx]
        self._buffer = self._buffer[best_idx:]
        return chunk

    def reset(self) -> None:
        """Clear the buffer without returning content.

        L1: the previous ``flush()`` method (which both cleared
        the buffer and returned the cleared text) was unused —
        the orchestrator's stream consumption loop
        (``_speak_state``) drains the splitter via the
        async iterator, never via the explicit ``flush()``
        call. The function was a leak vector for partial
        sentences if anyone ever called it at the wrong time
        (they would get a half-spoken fragment). Removed;
        callers can use ``reset()`` to drop state without
        returning content. The test suite still uses
        ``reset()`` to clear the buffer between cases.
        """
        self._buffer = ""
