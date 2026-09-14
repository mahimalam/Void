"""Silero VAD — Voice Activity Detection.

Detects when speech starts and ends in the audio stream.
Also provides barge-in detection while JARVIS is speaking.

Usage:
    vad = VADProcessor(cfg)
    # For end-of-utterance:
    speech = await vad.collect_utterance(audio_input)
    # For barge-in:
    if vad.detect_barge_in(frame):
        ...interrupt TTS...
"""

from __future__ import annotations

from collections import deque
import time
from typing import AsyncIterator

import numpy as np
import torch

from core.config import VADConfig
from core.errors import VADError
from core.logging_setup import get_logger

# Type alias for the VAD model
VADModel = object


class VADProcessor:
    """Silero VAD — speech / silence / barge-in detection."""

    def __init__(self, cfg: VADConfig) -> None:
        self._cfg = cfg
        self._log = get_logger("vad")
        self._model: VADModel | None = None
        self._sample_rate: int = 16000  # Silero expects 16 kHz
        self._frame_ms: int = 80  # each audio frame is 1280 samples @ 16kHz = 80ms

        # Internal state for utterance collection
        self._speech_buffer: list[np.ndarray] = []
        self._silence_frames: int = 0
        self._in_speech: bool = False
        self._utterance_start: float | None = None

        # Adaptive threshold override (set by orchestrator based on noise floor)
        self._dynamic_threshold: float | None = None

    async def load_model(self) -> None:
        """Download/load Silero VAD model (lazy)."""
        if self._model is not None:
            return
        try:
            import silero_vad

            self._model = silero_vad.load_silero_vad()
            self._model.reset_states()
            self._log.info("vad_model_loaded")
        except Exception as e:
            raise VADError(f"Failed to load Silero VAD: {e}") from e

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def _effective_threshold(self) -> float:
        return self._dynamic_threshold if self._dynamic_threshold is not None else self._cfg.threshold

    def set_threshold(self, threshold: float) -> None:
        """Override the VAD threshold dynamically (based on ambient noise floor)."""
        self._dynamic_threshold = threshold

    def reset_threshold(self) -> None:
        """Revert to the configured default threshold."""
        self._dynamic_threshold = None

    # --- Frame-level speech probability ---

    @staticmethod
    def _normalize(samples: np.ndarray) -> np.ndarray:
        """Normalize float32 audio so peak amplitude hits 0.9."""
        peak = np.max(np.abs(samples))
        if peak < 1e-6:
            return samples
        scale = 0.9 / peak
        # Cap at 10x to avoid blowing up absolute pure silence, but allow 
        # quiet laptop microphones to be boosted sufficiently for Silero VAD.
        scale = min(scale, 10.0)
        return samples * scale

    def speech_probability(self, frame: np.ndarray) -> float:
        """Return probability (0-1) that `frame` contains speech."""
        if self._model is None:
            return 0.0
        try:
            samples = frame if frame.ndim == 1 else frame.squeeze()
            samples = self._normalize(samples)  # Normalize to boost low mic volumes
            chunk_size = 512
            chunks = []
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                chunks.append(chunk)
            best = 0.0
            for chunk in chunks:
                t = torch.from_numpy(chunk).float()
                p = self._model(t, self._sample_rate).item()
                if p > best:
                    best = p
            prob = float(best)
            try:
                from core.metrics import GLOBAL_WATCHDOG
                GLOBAL_WATCHDOG.record_vad(prob)
            except Exception:
                pass
            return prob
        except Exception as e:
            self._log.warning("vad_speech_prob_failed", error=str(e))
            return 0.0



    def is_speech(self, frame: np.ndarray) -> bool:
        """Quick check — is this frame speech? Uses adaptive threshold if set."""
        return self.speech_probability(frame) >= self._effective_threshold

    # --- End-of-utterance detection ---

    async def collect_utterance(
        self,
        get_frame: Callable[[], Coroutine[Any, Any, AudioFrame | None]],
        max_duration_s: float = 30.0,
        max_consecutive_none: int = 20,
        initial_silence_timeout_s: float = 3.5,
    ) -> np.ndarray | None:
        """Collect audio frames from an async source until silence is detected.

        Returns the full utterance as a numpy array, or None if interrupted
        or if max_consecutive_none (device failure) is exceeded.

        H26: the previous implementation prepended ``speech_pad_ms``
        of ZERO audio BEFORE the first speech frame. That dropped
        the first ~250 ms of the user's utterance (the soft / quiet
        attack of "h", "w", "y" sounds that VAD often misses). The
        fix moves the padding to POST-speech: the leading edge of
        the user's audio is preserved verbatim, and the trailing
        padding still gives the STT model a clean end-of-utterance.
        """
        self.reset()
        self._utterance_start = time.monotonic()
        padding_frames = max(1, int(self._cfg.speech_pad_ms / self._frame_ms))
        consecutive_none = 0
        pre_buffer: deque[np.ndarray] = deque(maxlen=4)  # ~320ms rolling buffer before speech onset
        speech_frames_count = 0

        while True:
            frame = await get_frame()
            if frame is None:
                consecutive_none += 1
                if max_consecutive_none > 0 and consecutive_none >= max_consecutive_none:
                    self._log.info("utterance_aborted_device_failure", consecutive=consecutive_none)
                    break
                continue
            consecutive_none = 0
            frame_data = frame.samples if hasattr(frame, 'samples') else frame

            if not self._in_speech and initial_silence_timeout_s > 0:
                if time.monotonic() - self._utterance_start > initial_silence_timeout_s:
                    self._log.info("utterance_initial_silence_timeout", timeout_s=initial_silence_timeout_s)
                    break

            if time.monotonic() - self._utterance_start > max_duration_s:
                self._log.info("utterance_max_duration_reached")
                break

            prob = self.speech_probability(frame_data)
            is_speech = prob >= self._effective_threshold

            if is_speech:
                if not self._in_speech:
                    self._in_speech = True
                    # Prepend pre-speech frames so the soft onset of words is never clipped
                    for pf in pre_buffer:
                        self._speech_buffer.append(pf)

                self._speech_buffer.append(frame_data)
                self._silence_frames = 0
                speech_frames_count += 1
            else:
                if not self._in_speech:
                    pre_buffer.append(frame_data)
                else:
                    self._silence_frames += 1
                    # Keep trailing padding for clean acoustic tail
                    if self._silence_frames <= padding_frames:
                        self._speech_buffer.append(frame_data)

                    # End-of-utterance: require sustained silence
                    silence_ms = self._silence_frames * self._frame_ms
                    if silence_ms >= self._cfg.min_silence_ms:
                        break

        # Discard if empty or fewer than 3 frames (< 240ms) of real speech (mic taps / pops)
        if not self._speech_buffer or speech_frames_count < 3:
            self.reset()
            return None

        utterance = np.concatenate(self._speech_buffer)
        self.reset()
        return utterance

    # --- State management ---

    # L9: the previous ``detect_barge_in`` method has been
    # removed. It was DEPRECATED in Phase 0 (per its own
    # docstring) because VAD-based barge-in requires AEC to avoid
    # echo loops, and the live system uses OpenWakeWord for
    # barge-in via the orchestrator's ``_barge_in_monitor``. The
    # H27 zero-gap-300ms fix was applied to it but the method
    # was still never called. Keeping dead code with a fix is
    # worse than removing it: the next person to re-enable
    # VAD-based barge-in will write it fresh, with the right
    # short-silence tolerance, on top of the current
    # speech_probability / is_speech helpers.

    def reset(self) -> None:
        self._speech_buffer.clear()
        self._silence_frames = 0
        self._in_speech = False
        self._utterance_start = None
        if self._model is not None and hasattr(self._model, "reset_states"):
            try:
                self._model.reset_states()
            except Exception:
                pass

    async def cleanup(self) -> None:
        self._model = None
        self.reset()
