"""Deepgram Aura TTS client — cloud quality voice synthesis.

Deepgram Aura-2 produces genuinely cinematic voice quality comparable to
ElevenLabs, with ~150ms first-chunk latency. The API is simple: POST text,
receive raw PCM audio (audio/l16;rate=24000).

Wake-phrase voice switching:
  "Jarvis"     → aura-2-andromeda-en  (professional, authoritative)
  "Hey Jarvis" → aura-2-amalthea-en   (warm, expressive)

The client resamples 24000 Hz → 22050 Hz to match AudioOutput's pipeline.

Free credit: $200 on signup (~months of JARVIS use).
After that: $0.0150 per 1000 chars (~$0.05/day for heavy use).

Usage:
    client = DeepgramTTSClient(api_key="...", cfg=tts_cfg)
    client.set_active_voice("normal")   # or "sexy"
    audio_np = await client.synthesize("Online and ready, sir.")
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

import httpx
import numpy as np

from core.config import TTSConfig
from core.errors import TTSError
from core.logging_setup import get_logger

# Deepgram Aura-2 TTS endpoint
_DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

# Audio format returned by Deepgram: raw signed 16-bit PCM, 24000 Hz, mono
_DEEPGRAM_SAMPLE_RATE = 24000

# Voice models — matched to wake phrase (Aura-1 models synthesize in ~370ms vs 2000ms+ for Aura-2)
_VOICE_NORMAL = "aura-orion-en"     # "Jarvis" — deep, authoritative male
_VOICE_SEXY   = "aura-asteria-en"   # "Hey Jarvis" — warm, natural female

# Timeout for the Deepgram API call
_TIMEOUT_S = 8.0


def _decode_audio_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode audio bytes (MP3 or WAV/PCM) to float32 numpy array at target_sr."""
    if not raw_bytes:
        return np.array([], dtype=np.float32)

    try:
        import io
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != target_sr:
            n_src = len(audio)
            n_dst = max(1, int(n_src * target_sr / sr))
            src_idx = np.linspace(0, n_src - 1, n_dst)
            lo = np.floor(src_idx).astype(int)
            hi = np.minimum(lo + 1, n_src - 1)
            frac = (src_idx - lo).astype(np.float32)
            audio = audio[lo] * (1.0 - frac) + audio[hi] * frac
        return audio.astype(np.float32)
    except Exception:
        return _pcm16_to_float32(raw_bytes, 24000, target_sr)


def _pcm16_to_float32(raw: bytes, src_rate: int, dst_rate: int) -> np.ndarray:
    """Convert raw signed 16-bit PCM bytes to float32 numpy array.

    Resamples from src_rate (24000 Hz) to dst_rate (22050 Hz) using
    linear interpolation — fast, no extra dependencies.
    """
    if not raw:
        return np.array([], dtype=np.float32)

    # Decode int16 → float32 [-1, 1]
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample if needed
    if src_rate != dst_rate:
        n_src = len(samples)
        n_dst = max(1, int(n_src * dst_rate / src_rate))
        src_idx = np.linspace(0, n_src - 1, n_dst)
        lo = np.floor(src_idx).astype(int)
        hi = np.minimum(lo + 1, n_src - 1)
        frac = (src_idx - lo).astype(np.float32)
        samples = samples[lo] * (1.0 - frac) + samples[hi] * frac

    return samples.astype(np.float32)


class DeepgramTTSClient:
    """Async Deepgram Aura-2 TTS client with wake-phrase voice switching.

    Supports two voices selected by the orchestrator based on which
    wake phrase was detected:
      - normal mode ("Jarvis")     → aura-2-andromeda-en
      - expressive mode ("Hey Jarvis") → aura-2-amalthea-en

    Falls back to Piper (via TTSError) on any API failure so the voice
    loop never stalls.
    """

    def __init__(self, api_key: str, cfg: TTSConfig) -> None:
        self._api_key = api_key
        self._cfg = cfg
        self._log = get_logger("deepgram_tts")
        # Active voice — switched by orchestrator on wake detection
        self._active_model = _VOICE_NORMAL
        self._active_mode = "normal"
        # Failure tracking for backoff
        self._consecutive_failures = 0
        self._backoff_until: float = 0.0
        # Session character counter (informational)
        self._chars_session = 0
        self._http_client: Any | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=_TIMEOUT_S,
                limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=60.0),
            )
        return self._http_client

    async def warm_up(self) -> None:
        if self._api_key:
            try:
                client = self._get_http_client()
                await client.get("https://api.deepgram.com", timeout=2.5)
            except Exception:
                pass

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def is_backed_off(self) -> bool:
        """True if we're in a transient error backoff window."""
        return time.monotonic() < self._backoff_until

    def set_active_voice(self, mode: str) -> None:
        """Switch voice based on wake phrase detection.

        Args:
            mode: "normal" for "Jarvis", "sexy" for "Hey Jarvis".
        """
        if mode == "sexy":
            new_model = _VOICE_SEXY
        else:
            new_model = _VOICE_NORMAL

        if new_model != self._active_model:
            self._log.info(
                "deepgram_voice_switched",
                from_mode=self._active_mode,
                to_mode=mode,
                model=new_model,
            )
            self._active_model = new_model
            self._active_mode = mode

    def _mark_failure(self, is_quota: bool = False) -> None:
        self._consecutive_failures += 1
        if is_quota:
            # Credit exhausted — back off for the session
            self._backoff_until = time.monotonic() + 3600.0
            self._log.warning(
                "deepgram_credit_exhausted",
                chars_session=self._chars_session,
                hint="Top up at deepgram.com or switch to Edge TTS fallback.",
            )
        elif self._consecutive_failures >= 3:
            self._backoff_until = time.monotonic() + 15.0
            self._log.warning(
                "deepgram_consecutive_failures",
                consecutive=self._consecutive_failures,
                backoff_s=15,
            )

    def _mark_success(self) -> None:
        if self._consecutive_failures > 0:
            self._log.info(
                "deepgram_recovered",
                prior_failures=self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._backoff_until = 0.0

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text via Deepgram Aura-2 API.

        Returns float32 numpy array at cfg.sample_rate Hz.
        Raises TTSError on failure so TTSManager falls back to Piper.
        """
        text = text.strip()
        if not text:
            return np.array([], dtype=np.float32)

        if self.is_backed_off:
            remaining = round(self._backoff_until - time.monotonic(), 1)
            raise TTSError(
                f"Deepgram TTS in backoff for {remaining}s",
                spoken="",
            )

        t0 = time.perf_counter()
        try:
            client = self._get_http_client()
            response = await client.post(
                _DEEPGRAM_TTS_URL,
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                },
                params={
                    "model": self._active_model,
                    "encoding": "mp3",
                },
            )

            elapsed_ms = round((time.perf_counter() - t0) * 1000)

            if response.status_code == 401:
                self._mark_failure()
                raise TTSError(
                    "Deepgram API key invalid (401). Check DEEPGRAM_API_KEY in data/.env.",
                    spoken="Cloud voice key is invalid.",
                )

            if response.status_code == 402:
                self._mark_failure(is_quota=True)
                raise TTSError(
                    "Deepgram credit exhausted (402). Top up at deepgram.com.",
                    spoken="",  # silent — falls back to Piper
                )

            if response.status_code == 429:
                self._mark_failure()
                raise TTSError(
                    "Deepgram rate limited (429).",
                    spoken="",
                )

            if response.status_code != 200:
                self._mark_failure()
                raise TTSError(
                    f"Deepgram HTTP {response.status_code}: {response.text[:200]}",
                    spoken="",
                )

            raw_bytes = response.content
            if not raw_bytes:
                self._mark_failure()
                raise TTSError("Deepgram returned empty audio.", spoken="")

            # Fast decode MP3 → float32 numpy array (sample_rate: 22050 Hz)
            target_sr = self._cfg.sample_rate
            audio = await asyncio.to_thread(
                _decode_audio_bytes, raw_bytes, target_sr
            )

            self._chars_session += len(text)
            self._mark_success()
            self._log.info(
                "deepgram_tts_complete",
                chars=len(text),
                model=self._active_model,
                latency_ms=elapsed_ms,
                audio_samples=len(audio),
                chars_session=self._chars_session,
            )
            return audio

        except TTSError:
            raise
        except httpx.TimeoutException:
            self._mark_failure()
            raise TTSError(
                f"Deepgram TTS timed out after {_TIMEOUT_S}s.",
                spoken="",
            )
        except Exception as e:
            self._mark_failure()
            raise TTSError(
                f"Deepgram TTS failed: {e}",
                spoken="",
            ) from e
