"""Speech-to-Text — Deepgram Nova-3 transcription.

Phase 4: Pure cloud STT using Deepgram Nova-3.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import wave
from dataclasses import dataclass
from typing import Callable

import httpx
import numpy as np

from core.config import STTConfig
from core.errors import STTError
from core.logging_setup import get_logger

@dataclass
class TranscriptionResult:
    text: str
    avg_logprob: float
    no_speech_prob: float
    detected_lang: str | None
    used_translate_fallback: bool = False

class STTEngine:
    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg
        self._log = get_logger("stt")
        self._api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not self._api_key:
            env_path = os.path.join("data", ".env")
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if "=" in line and not line.startswith("#"):
                            k, v = line.strip().split("=", 1)
                            if k.strip() == "DEEPGRAM_API_KEY":
                                self._api_key = v.strip()
        
        if not self._api_key:
            self._log.warning("DEEPGRAM_API_KEY is missing! STT will fail.")
        
        self._consecutive_failures = 0
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=8.0,
                limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=60.0),
            )
        return self._http_client

    async def load_model(self) -> None:
        if self._api_key:
            try:
                client = self._get_http_client()
                await client.get("https://api.deepgram.com", timeout=2.5)
            except Exception:
                pass

    async def cleanup(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    @staticmethod
    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    async def transcribe(
        self,
        audio: np.ndarray,
        _on_partial: Callable[[str], None] | None = None,
    ) -> TranscriptionResult:
        if audio.size == 0:
            return TranscriptionResult("", 0.0, 1.0, None)

        if not self._api_key:
            raise STTError("DEEPGRAM_API_KEY not configured.", spoken="I cannot hear you, my STT key is missing.")

        wav_bytes = await asyncio.to_thread(self._numpy_to_wav_bytes, audio, 16000)
        
        t0 = time.perf_counter()
        try:
            client = self._get_http_client()
            response = await client.post(
                "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en",
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": "audio/wav"
                },
                content=wav_bytes,
            )
            
            response.raise_for_status()
            data = response.json()
            
            try:
                channels = data["results"]["channels"][0]
                alternatives = channels["alternatives"][0]
                text = alternatives["transcript"]
                confidence = alternatives["confidence"]
                import math
                avg_logprob = math.log(max(confidence, 0.001))
            except (KeyError, IndexError):
                text = ""
                avg_logprob = -2.0
                confidence = 0.0
                
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            
            self._log.info(
                "deepgram_stt_complete",
                chars=len(text),
                audio_s=round(len(audio)/16000, 2),
                latency_ms=elapsed_ms,
                confidence=round(confidence, 3)
            )
            
            audio_duration_s = len(audio) / 16000
            word_count = len(text.split()) if text else 0
            words_per_second = word_count / max(audio_duration_s, 0.1)
            
            text_lower = text.lower().strip().rstrip(".,!?")
            _FILLER_NOISE = {
                "uh", "um", "uh-huh", "uh huh", "mhm", "mm-hmm", "mm hmm",
                "huh", "mm", "hmm", "ah", "oh", "ha"
            }
            _SUBTITLE_HALLUCINATIONS = {
                "thank you for watching", "thanks for watching",
                "please subscribe", "subtitles by", "watching"
            }
            _ORPHAN_NOISE = {"the", "a", "an", "it", "he", "she", "we", "they"}
            
            if text_lower in _FILLER_NOISE or text_lower in _SUBTITLE_HALLUCINATIONS:
                text = ""
                avg_logprob = -2.0
            elif text_lower in _ORPHAN_NOISE and word_count == 1:
                avg_logprob = -2.0
            elif confidence < 0.25 and word_count <= 2:
                avg_logprob = -2.0
            elif audio_duration_s > 1.2 and words_per_second < 0.4 and word_count <= 2:
                avg_logprob = -2.0
                
            return TranscriptionResult(text=text, avg_logprob=avg_logprob, no_speech_prob=0.0, detected_lang="en")

        except Exception as e:
            self._log.error("deepgram_stt_error", error=str(e))
            raise STTError(f"Deepgram STT failed: {e}", spoken="") from e
