"""Deepgram Aura TTS — low latency cloud voice synthesis.

Phase 4: Pure cloud TTS replacing local Piper.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from core.config import TTSConfig
from core.errors import TTSError
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.audio import AudioOutput
    from core.event_bus import EventBus

class TTSManager:
    def __init__(
        self,
        cfg: TTSConfig,
        bus: EventBus | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._log = get_logger("tts_manager")
        self._playback_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._playback_task: asyncio.Task[None] | None = None
        self._audio_output = None
        self._flushing = False
        self._flush_lock = asyncio.Lock()
        
        self._deepgram_client = None
        if cfg.deepgram_api_key:
            from core.tts_deepgram import DeepgramTTSClient
            self._deepgram_client = DeepgramTTSClient(
                api_key=cfg.deepgram_api_key,
                cfg=cfg,
            )
            self._log.info("deepgram_tts_configured")
        else:
            self._log.warning("DEEPGRAM_API_KEY is missing! TTS will fail.")

    @property
    def is_speaking(self) -> bool:
        task_alive = self._playback_task is not None and not self._playback_task.done()
        if task_alive:
            return True
        if self._audio_output is not None:
            try:
                if self._audio_output.pending_samples > 0:
                    return True
            except Exception:
                pass
        return False

    def set_volume(self, level: float) -> None:
        if self._audio_output is not None:
            self._audio_output.set_volume(level)

    def get_volume(self) -> float:
        if self._audio_output is not None:
            return self._audio_output.get_volume()
        return 1.0

    def set_active_voice(self, mode: str) -> None:
        if self._deepgram_client is not None:
            self._deepgram_client.set_active_voice(mode)

    async def start(self) -> None:
        self._playback_task = asyncio.create_task(self._playback_loop())
        if self._deepgram_client is not None and hasattr(self._deepgram_client, "warm_up"):
            asyncio.create_task(self._deepgram_client.warm_up())
        self._log.info("tts_manager_started", mode="pure_cloud")

    async def speak(self, text: str) -> None:
        if self._flushing:
            return

        if self._deepgram_client is not None and not self._deepgram_client.is_backed_off:
            try:
                audio = await self._deepgram_client.synthesize(text)
                if audio.size > 0:
                    await self._playback_queue.put(audio)
            except Exception as e:
                self._log.error("deepgram_tts_failed", error=str(e))
        else:
            self._log.error("deepgram_tts_not_configured_or_backed_off")

    async def speak_sync(self, text: str) -> None:
        await self.speak(text)
        await self.wait_until_queue_drained()
        if self._audio_output is not None:
            await self._audio_output.wait_until_drained()

    async def speak_stream(self, chunks: AsyncIterator[str]) -> None:
        async for chunk in chunks:
            if self._flushing:
                break
            await self.speak(chunk)

    async def wait_until_queue_drained(self, timeout: float = 60.0) -> None:
        await asyncio.wait_for(self._playback_queue.join(), timeout=timeout)

    async def flush(self) -> None:
        async with self._flush_lock:
            self._flushing = True

            while not self._playback_queue.empty():
                try:
                    self._playback_queue.get_nowait()
                    self._playback_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            if self._playback_task is not None:
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._playback_task = None

            self._playback_task = asyncio.create_task(self._playback_loop())
            await asyncio.sleep(0)
            self._flushing = False
            self._log.info("tts_flushed")

    async def _playback_loop(self) -> None:
        try:
            while True:
                audio = await self._playback_queue.get()
                if self._audio_output is not None and audio.size > 0:
                    peak = np.max(np.abs(audio))
                    if peak > 1e-6:
                        gain = min(0.9 / peak, 3.0)
                        if gain > 1.1:
                            audio = audio * gain
                    try:
                        await self._audio_output.play(audio)
                    except Exception as e:
                        self._log.error("playback_error", error=str(e))
                self._playback_queue.task_done()
        except asyncio.CancelledError:
            pass

    def set_audio_output(self, audio_out: AudioOutput) -> None:
        self._audio_output = audio_out

    async def stop(self) -> None:
        self._flushing = True
        if self._playback_task is not None:
            self._playback_task.cancel()
            self._playback_task = None
        if self._deepgram_client is not None and hasattr(self._deepgram_client, "close"):
            try:
                await self._deepgram_client.close()
            except Exception:
                pass
        self._log.info("tts_manager_stopped")
