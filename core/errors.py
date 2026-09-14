"""Typed exceptions for every external boundary in the voice loop.

Every catch-all in the orchestrator distinguishes:
  - Recoverable: log + spoken apology + return to IDLE.
  - Unrecoverable: log + spoken apology + return to IDLE (Phase 0 treats
    every error as recoverable; the voice loop never crashes).

Raising a typed exception (instead of a bare Exception) makes it possible
to attach a user-facing spoken_message that the orchestrator can read out.
"""

from __future__ import annotations


class JarvisError(Exception):
    """Base class. `spoken_message` is what JARVIS will say to the user.

    M3: ``spoken_message`` used to be a *class* attribute. Subclasses
    overrode it via assignment, and the base ``__init__`` then had to
    check ``spoken is not None`` to avoid clobbering the subclass
    value. The class-attribute pattern is also fragile: every
    instance shares the same string, so any future code that ever
    *mutated* ``self.spoken_message`` (e.g. an "enrich with context"
    helper) would silently leak that change across every other
    instance. ``spoken_message`` is now an instance attribute,
    initialised in ``__init__`` from the class default. Subclasses
    keep the public override-via-class-attribute pattern by storing
    their default in a ``_spoken_default`` class attribute that
    ``__init__`` reads once.
    """

    # Subclasses may set this to override the default spoken string.
    # It is a class attribute so it is shared across all instances of
    # the subclass, which is what we want for a constant default.
    _spoken_default: str = "Something went wrong."

    def __init__(self, technical: str, spoken: str | None = None) -> None:
        super().__init__(technical)
        # Read the per-class default via ``type(self)`` so subclasses
        # that override ``_spoken_default`` get the right value.
        self.spoken_message: str = (
            spoken if spoken is not None else type(self)._spoken_default
        )


class ConfigError(JarvisError):
    _spoken_default = "My configuration is invalid. Check the log."


class AudioDeviceError(JarvisError):
    _spoken_default = "I can't reach the microphone or speakers."


class WakeWordError(JarvisError):
    _spoken_default = "Wake word system failed to start."


class VADError(JarvisError):
    _spoken_default = "Voice activity detection failed."


class STTError(JarvisError):
    _spoken_default = "I couldn't transcribe what you said."


class TTSError(JarvisError):
    _spoken_default = "My voice synthesis stopped working."


class LLMError(JarvisError):
    _spoken_default = "My brain is unreachable right now."


class OllamaUnavailableError(LLMError):
    _spoken_default = "Ollama is not responding. Make sure it is running."


class GPUOutOfMemoryError(LLMError):
    _spoken_default = "I'm out of GPU memory. Falling back to a smaller model."


class GeminiError(LLMError):
    _spoken_default = "I couldn't reach the cloud brain."


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------

class ToolExecutionError(JarvisError):
    """Base class for tool execution failures."""
    _spoken_default = "That tool ran into a problem."
