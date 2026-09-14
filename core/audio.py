"""Audio I/O — async sounddevice streams.

Microphone input (16 kHz) and speaker output (22.05 kHz) wrapped in
asyncio-friendly generators so the voice loop never blocks.

Usage:
    audio = AudioIO(cfg)
    async for frame in audio.input_stream():
        ...process frame...
    audio.play_chunk(chunk)   # non-blocking
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import numpy as np
import sounddevice as sd

from core.config import AudioConfig
from core.errors import AudioDeviceError
from core.event_bus import EventBus, EventType
from core.logging_setup import get_logger


@dataclass
class AudioFrame:
    """A single audio frame from the microphone."""

    samples: np.ndarray  # shape: (frame_samples_in,)
    timestamp: float     # perf_counter at capture


class AudioInput:
    """Async generator yielding 80 ms audio frames from the microphone."""

    def __init__(
        self,
        cfg: AudioConfig,
        bus: EventBus | None = None,
        on_error: Callable[[Exception], None] | None = None,
        aec: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._log = get_logger("audio_input")
        self._on_error = on_error
        self._aec = aec
        self._frames_per_buffer = cfg.frame_samples_in  # e.g. 1280 @ 16 kHz
        self._stream: sd.InputStream | None = None
        self._running = False
        self._consecutive_errors = 0
        # 2026-06-20: ghost-stream watchdog. When a
        # Bluetooth headset disconnects abruptly, PortAudio
        # sometimes enters a stuck state where ``read()``
        # blocks forever instead of raising PortAudioError.
        # We track the last time we got a valid frame and
        # force a reconnect if it's been too long (default
        # 5 s). This is the difference between a working
        # voice loop and a JARVIS that just stares at you.
        self._last_valid_frame_at: float = 0.0
        self._watchdog_timeout_s: float = 5.0
        self._reconnect_device_specified: int | None = None
        # M28: bound the reconnect loop. The previous code
        # retried forever on a broken device, which spammed
        # the log. We now give up after ``_MAX_RECONNECTS``
        # consecutive failures and surface the error to the
        # user via ``on_error`` so they know the mic is gone.
        self._reconnect_attempts: int = 0
        self._max_reconnects: int = 3
        # Real-time DC-blocking high-pass filter state (fc ~= 15 Hz @ 16kHz)
        self._dc_b = np.array([1.0, -1.0], dtype=np.float32)
        self._dc_a = np.array([1.0, -0.995], dtype=np.float32)
        self._dc_zi = np.zeros(1, dtype=np.float32)

    def _get_best_input_device(self) -> int | None:
        """Scan devices for connected bluetooth/headset mics, skipping incompatible WDM-KS."""
        if self._cfg.input_device is not None:
            return self._cfg.input_device
        
        try:
            devices = sd.query_devices()
            keywords = ["airpods", "soundcore", "oneplus", "bullets", "necklace", "riro", "hands-free", "headset", "microphone array"]
            for kw in keywords:
                for idx, dev in enumerate(devices):
                    # Skip WDM-KS hostapi (3) — "Blocking API not supported" on Windows
                    if dev.get("hostapi") == 3:
                        continue
                        
                    if dev["max_input_channels"] > 0:
                        name = dev["name"].lower()
                        if kw in name:
                            self._log.info("auto_detected_input_device", name=dev["name"], hostapi=dev.get("hostapi"), index=idx)
                            return idx
        except Exception as e:
            self._log.warning("device_scan_failed", error=str(e))
        return None

    @property
    def sample_rate(self) -> int:
        return self._cfg.sample_rate_in

    @property
    def frame_samples(self) -> int:
        return self._cfg.frame_samples_in

    async def start(self) -> None:
        """Open the input stream (callback-driven, queue-based).

        Uses a PortAudio callback to push frames into an asyncio.Queue
        instead of blocking asyncio.to_thread reads. This makes read_frame()
        cancellation-safe: there is never a thread stuck in stream.read()
        that can corrupt ALSA state when the caller's Task is cancelled
        (e.g. when the wake-word detector exits via return True mid-read).
        """
        device = self._reconnect_device_specified if self._reconnect_device_specified is not None else self._get_best_input_device()
        try:
            loop = asyncio.get_running_loop()
            # 200 frames = ~16 seconds at 80ms/frame — large enough to never fill
            # during normal conversation but bounded to avoid unbounded memory growth.
            self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)

            def _audio_callback(indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
                """PortAudio callback — runs on audio thread, must not block."""
                samples = indata[:, 0].copy() if indata.ndim > 1 else indata.ravel().copy()
                # Schedule put_nowait on the asyncio thread via call_soon_threadsafe.
                # IMPORTANT: QueueFull must be caught INSIDE the scheduled function,
                # not here, because call_soon_threadsafe only schedules the call —
                # the actual put_nowait runs later in the event loop. If we pass
                # put_nowait directly, a QueueFull exception bubbles up through
                # asyncio's internal Handle._run() and pollutes the log.
                def _safe_put() -> None:
                    try:
                        self._frame_queue.put_nowait(samples)
                    except asyncio.QueueFull:
                        pass  # Drop oldest frame — consumer is too slow

                try:
                    loop.call_soon_threadsafe(_safe_put)
                except RuntimeError:
                    pass  # Event loop closed (shutdown)


            self._stream = sd.InputStream(
                samplerate=self._cfg.sample_rate_in,
                blocksize=self._frames_per_buffer,
                device=device,
                channels=self._cfg.channels,
                dtype="float32",
                callback=_audio_callback,
            )
            self._stream.start()
            self._running = True
            self._last_valid_frame_at = time.monotonic()
            self._log.info(
                "input_stream_opened",
                sr=self._cfg.sample_rate_in,
                device=device,
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to open input stream: {e}") from e


    async def read_frame(self) -> AudioFrame | None:
        """Read one frame from the callback queue. Returns None on stream stop.

        This is fully cancellation-safe: unlike asyncio.to_thread(stream.read),
        awaiting a queue.get() can be cancelled at any await point without
        leaving an orphaned ALSA read thread that corrupts stream state.
        """
        if not self._running or self._stream is None:
            return None
        try:
            # Wait for the PortAudio callback to push a frame.
            # Use a short timeout so callers see None promptly on shutdown.
            try:
                samples = await asyncio.wait_for(self._frame_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                return None

            self._consecutive_errors = 0
            self._last_valid_frame_at = time.monotonic()

            if samples.size == 0:
                return None

            timestamp = time.monotonic()
            # Real-time DC blocking high-pass filter (cutoff ~15 Hz at 16kHz)
            # Removes analog hardware DC bias without affecting speech frequencies (80Hz - 8kHz)
            if samples.size > 0:
                import scipy.signal as signal
                samples, self._dc_zi = signal.lfilter(self._dc_b, self._dc_a, samples, zi=self._dc_zi)

            if self._aec is not None:
                samples = self._aec.process(samples)
            # Apply input gain scaling if specified
            gain = getattr(self._cfg, 'gain', 1.0)
            if gain != 1.0:
                samples = np.clip(samples * gain, -1.0, 1.0)

            try:
                from core.metrics import GLOBAL_WATCHDOG
                rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size > 0 else 0.0
                GLOBAL_WATCHDOG.record_audio_frame(rms)
            except Exception:
                pass

            return AudioFrame(samples=samples.copy(), timestamp=timestamp)
        except sd.PortAudioError as e:
            self._consecutive_errors += 1
            self._log.warning("input_read_transient", error=str(e)[:120], consecutive=self._consecutive_errors)
            if self._consecutive_errors >= 5:
                await self._reconnect()
            await asyncio.sleep(0.5)
            return None
        except Exception as e:
            self._log.error("input_read_error", error=str(e))
            if self._on_error:
                self._on_error(e)
            return None
        finally:
            # Ghost-stream watchdog: if it's been too long since we got a frame,
            # the stream may be stuck (Bluetooth dropout). Force a reconnect.
            if (
                self._last_valid_frame_at > 0
                and time.monotonic() - self._last_valid_frame_at
                > self._watchdog_timeout_s
                and self._consecutive_errors == 0
            ):
                self._log.warning(
                    "input_watchdog_no_data",
                    last_data_s=round(
                        time.monotonic() - self._last_valid_frame_at, 1
                    ),
                )
                self._consecutive_errors = 5
                await self._reconnect()


    async def _reconnect(self) -> None:
        """Close the dead stream and re-open with default device (fallback).

        M28: the previous code retried forever on a broken
        device, which spammed the log. We now give up after
        ``self._max_reconnects`` consecutive failures and
        surface the error to the user via ``on_error`` so
        they know the mic is gone. The bound is per-instance
        (default 3), so a temporary device glitch still
        recovers, but a permanently-disconnected headset
        fails cleanly.
        """
        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._max_reconnects:
            self._log.error(
                "input_reconnect_exhausted",
                attempts=self._reconnect_attempts,
                max=self._max_reconnects,
            )
            self._running = False
            if self._on_error:
                try:
                    self._on_error(
                        AudioDeviceError(
                            f"Microphone disconnected after {self._max_reconnects} reconnect attempts."
                        )
                    )
                except Exception:
                    pass
            return
        self._log.info("input_reconnecting", attempt=self._reconnect_attempts)
        await self.stop()
        self._reconnect_device_specified = None  # Use system default
        self._consecutive_errors = 0
        try:
            await self.start()
            # Success — reset the counter so a future disconnect
            # gets the full retry budget again.
            self._reconnect_attempts = 0
            # 2026-06-20: reset the watchdog timestamp
            # so the freshly-opened stream has the full
            # ``_watchdog_timeout_s`` window before the
            # next check.
            self._last_valid_frame_at = time.monotonic()
            self._log.info("input_reconnected")
        except Exception as e:
            self._log.error(
                "input_reconnect_failed",
                attempt=self._reconnect_attempts,
                error=str(e),
            )

    async def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            stream = self._stream
            self._stream = None
            try:
                # Run the blocking stop/close in a thread, but wait for it.
                # Avoids ALSA assertion errors by waiting for callback to finish.
                def _close():
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                t = threading.Thread(target=_close)
                t.start()
                t.join(timeout=1.0)
            except Exception:
                pass
            self._log.info("input_stream_closed")

    def clear_buffer(self) -> None:
        """Discard all pending frames currently sitting in the frame queue.

        Prevents stale echo/noise from being processed by VAD after TTS ends.
        Safe to call from the asyncio thread at any time.
        """
        if not self._running:
            return
        try:
            while not self._frame_queue.empty():
                self._frame_queue.get_nowait()
        except Exception:
            pass


    def set_aec_active(self, active: bool) -> None:
        """Enable/disable AEC on read frames. Active only during TTS playback."""
        if self._aec is not None:
            self._aec.active = active

    async def __aenter__(self) -> AudioInput:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


class AudioOutput:
    """Non-blocking speaker output via callback-driven OutputStream.

    Uses a ring buffer fed by the event loop and drained by PortAudio's
    callback thread. Playback is truly non-blocking — barge-in just
    clears the buffer.
    """

    def __init__(
        self,
        cfg: AudioConfig,
        bus: EventBus | None = None,
        on_play: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._log = get_logger("audio_output")
        self._on_play = on_play
        self._stream: sd.OutputStream | None = None
        # H11: the play() hot path used to do
        # ``self._buffer = np.concatenate([self._buffer, flat])`` per
        # chunk, which is O(n+m) per call and becomes O(n²) total
        # over a long TTS stream (8 chunks = 8 full copies of an
        # ever-growing array). The fix is a ``deque`` of numpy
        # chunks: ``play()`` is O(m) (one append), and the PortAudio
        # callback concatenates lazily from the front of the deque
        # only as many samples as it actually needs.
        self._chunks: deque[np.ndarray] = deque()
        # ``_total_samples`` is the running sum of chunk lengths
        # under the lock. Used by ``pending_samples`` so callers
        # can keep their lock-free read pattern.
        self._total_samples: int = 0
        self._lock = threading.Lock()
        # 2026-06-20: live volume for barge-in ducking. The
        # PortAudio callback multiplies every sample by this
        # scalar before writing to the output buffer. Default
        # 1.0 = full volume. Barge-in ducks to ``duck_volume``
        # (default 0.1) the moment speech is detected, then
        # restores to 1.0 on silence. This is what Siri /
        # Google Assistant do to suppress their own TTS bleed
        # into the mic.
        self._volume: float = 1.0
        # threading.Event — set by the PortAudio callback (which runs on
        # the audio thread, NOT the asyncio loop). Used as the source of
        # truth for "buffer is empty" inside the callback.
        self._ring_buffer_drained = threading.Event()
        self._ring_buffer_drained.set()
        # asyncio.Event — the signal-based completion event awaited by
        # the orchestrator. It is set from the asyncio thread via
        # ``loop.call_soon_threadsafe`` whenever the audio thread signals
        # that the buffer has just become empty. Awaiting this event is
        # what replaces the old 3s polling cap on wait_until_drained.
        # Initially "drained" so the very first caller does not block.
        self._drained_async: asyncio.Event | None = None  # type: ignore[assignment]
        # Captured in start() — the running asyncio loop. The audio
        # thread uses this to schedule asyncio event sets safely.
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def pending_samples(self) -> int:
        """Number of audio samples still queued in the ring buffer."""
        with self._lock:
            return self._total_samples

    def set_volume(self, level: float) -> None:
        """Set live output volume. 1.0 = full, 0.0 = silent.

        2026-06-20: used by the barge-in monitor to duck the
        TTS volume the moment speech is detected during playback.
        This eliminates speaker bleed into the mic, which is
        the #1 cause of false barge-in triggers on laptop
        built-in speakers. The change is picked up by the
        PortAudio callback within one frame (~10 ms) so it
        is effectively instantaneous.

        Args:
            level: Volume scalar, 0.0-1.0. Values outside
                this range are clamped. 1.0 is the natural
                default.
        """
        clamped = max(0.0, min(1.0, float(level)))
        with self._lock:
            self._volume = clamped

    def get_volume(self) -> float:
        """Current output volume. 1.0 = full, 0.0 = silent."""
        with self._lock:
            return self._volume

    @property
    def device_name(self) -> str:
        """Name of the active output device, or empty string if unknown.

        2026-06-20: used by the barge-in monitor to detect
        laptop built-in speakers (severe acoustic feedback)
        vs headphones (no feedback) and tune thresholds.
        """
        try:
            import sounddevice as sd
            if self._stream is not None and self._stream.device is not None:
                idx = self._stream.device
                if isinstance(idx, (list, tuple)):
                    idx = idx[1]  # (input, output) tuple
                info = sd.query_devices(idx)
                return str(info.get("name", ""))
        except Exception:
            pass
        return ""

    @property
    def device_class(self) -> str:
        """Heuristic classification of the active output device.

        2026-06-20: returns one of:
          - ``"headphones"`` — high-impedance output, no acoustic
            feedback into the mic. Easiest barge-in case.
          - ``"bluetooth"`` — wireless headset, variable
            latency. Moderate barge-in difficulty.
          - ``"laptop_speakers"`` — built-in speakers near the
            mic. Worst case for acoustic feedback. Requires
            conservative barge-in thresholds.
          - ``"unknown"`` — can't determine. Conservative default.

        Detection is heuristic (name-based). The orchestrator
        uses this to pick barge-in timing/ducking.
        """
        name = self.device_name.lower()
        if not name:
            return "unknown"
        # Headphones / headsets / earbuds (wired or wireless).
        headphone_markers = (
            "headphone", "headset", "earbud", "earphone", "airpods",
            "razer", "hyperx", " steelseries", "sony wh", "bose",
            "jabra", "sennheiser", "bluetooth",
        )
        for marker in headphone_markers:
            if marker in name:
                return "headphones"
        # Bluetooth explicitly (already covered above but keep separate).
        if "bluetooth" in name:
            return "bluetooth"
        # Built-in / laptop speakers (Realtek, Intel, Conexant, etc).
        speaker_markers = (
            "realtek", "intel(r)", "conexant", "idt", "via hd",
            "built-in", "internal", "laptop",
        )
        for marker in speaker_markers:
            if marker in name:
                return "laptop_speakers"
        # Generic "Speakers" without other context is also
        # likely laptop speakers in a laptop use case.
        if "speakers" in name or "speaker" in name:
            return "laptop_speakers"
        return "unknown"

    def _get_best_output_device(self) -> int | None:
        """Scan devices for connected bluetooth/headset speakers, skipping incompatible WDM-KS."""
        if self._cfg.output_device is not None:
            return self._cfg.output_device
        
        try:
            devices = sd.query_devices()
            keywords = ["airpods", "soundcore", "oneplus", "bullets", "necklace", "riro", "headphones", "headset", "speakers"]
            for kw in keywords:
                for idx, dev in enumerate(devices):
                    # Skip WDM-KS hostapi (3) — "Blocking API not supported" on Windows
                    if dev.get("hostapi") == 3:
                        continue
                        
                    if dev["max_output_channels"] > 0:
                        name = dev["name"].lower()
                        if kw in name:
                            self._log.info("auto_detected_output_device", name=dev["name"], hostapi=dev.get("hostapi"), index=idx)
                            return idx
        except Exception as e:
            self._log.warning("device_scan_failed", error=str(e))
        return None

    @property
    def sample_rate(self) -> int:
        return self._cfg.sample_rate_out

    async def start(self) -> None:
        # Capture the running loop so the PortAudio callback (which runs
        # on a non-asyncio thread) can wake the asyncio event safely.
        self._loop = asyncio.get_running_loop()
        self._drained_async = asyncio.Event()
        self._drained_async.set()

        device = self._get_best_output_device()
        loop_ref = self._loop
        drained_async_ref = self._drained_async

        def _callback(outdata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
            """H11: drain lazily from the front of the chunk deque.

            Instead of maintaining a single concatenated ``_buffer``
            array (O(n²) total over a long stream) we hold individual
            chunks and consume them one at a time, filling ``outdata``
            from the head of the deque. The first chunk is consumed
            directly into ``outdata``; any overflow into the next
            chunk is a single ``np.concatenate`` of two chunks, not
            of "everything we ever queued".

            2026-06-20: apply ``self._volume`` to every output
            sample. The barge-in monitor ducks the volume to ~10%
            the moment speech is detected during TTS, eliminating
            the speaker bleed that's the #1 source of false
            barge-ins. Volume is read under the lock so a set
            from the asyncio thread is picked up by the audio
            thread within one callback (~10 ms).
            """
            with self._lock:
                vol = self._volume
                written = 0
                while written < frames and self._chunks:
                    chunk = self._chunks[0]
                    remaining_in_chunk = len(chunk)
                    need = frames - written
                    if remaining_in_chunk <= need:
                        # Apply volume as we write. ``vol == 1.0``
                        # is the common case and the multiply is
                        # a no-op that numpy folds away; for
                        # ducked playback (vol = 0.1) it scales
                        # the audio down by 20 dB which kills
                        # the bleed.
                        if vol == 1.0:
                            outdata[written:written + remaining_in_chunk, 0] = chunk
                        else:
                            outdata[written:written + remaining_in_chunk, 0] = chunk * vol
                        written += remaining_in_chunk
                        self._chunks.popleft()
                        self._total_samples -= remaining_in_chunk
                    else:
                        if vol == 1.0:
                            outdata[written:frames, 0] = chunk[:need]
                        else:
                            outdata[written:frames, 0] = chunk[:need] * vol
                        self._chunks[0] = chunk[need:]
                        self._total_samples -= need
                        written = frames
                if written < frames:
                    outdata[written:, 0] = 0.0
                if self._total_samples == 0 and not self._ring_buffer_drained.is_set():
                    self._ring_buffer_drained.set()
                    # Wake the orchestrator's awaiter. We are on the
                    # audio thread, so we must hop to the asyncio thread.
                    if loop_ref is not None and drained_async_ref is not None:
                        try:
                            loop_ref.call_soon_threadsafe(drained_async_ref.set)
                        except RuntimeError:
                            # Loop closed (shutdown) — nothing to do.
                            pass
        self._callback_func = _callback

        try:
            self._stream = sd.OutputStream(
                samplerate=self._cfg.sample_rate_out,
                device=device,
                channels=self._cfg.channels,
                dtype="float32",
                callback=_callback,
            )
            self._stream.start()
            self._log.info(
                "output_stream_opened",
                sr=self._cfg.sample_rate_out,
                device=device,
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to open output stream: {e}") from e

    async def play(self, samples: np.ndarray) -> None:
        """Feed audio into the ring buffer (non-blocking, no executor).

        H11: this is O(m) per call where m is the size of the new
        chunk. The old ``np.concatenate`` reallocated the entire
        buffer on every chunk.
        """
        if samples.size == 0:
            return
        flat = samples.ravel()
        with self._lock:
            self._chunks.append(flat)
            self._total_samples += len(flat)
            self._ring_buffer_drained.clear()
            # Also clear the asyncio event under the lock. This means
            # the call_soon_threadsafe in the callback cannot leave a
            # stale "drained" signal in the wake of a fresh play().
            if self._drained_async is not None:
                self._drained_async.clear()
        if self._on_play is not None:
            self._on_play(flat)

    def flush(self) -> None:
        """Immediately stop all output.

        C6 fix: the previous implementation called ``self._stream.stop()``
        and ``self._stream.start()`` on every flush, which pops/glitches
        the audio device on Windows each time the user barges in. The
        PortAudio callback naturally plays silence when the ring buffer
        is empty, so we only need to clear the buffer and signal the
        drained event. If the stream ever does need a hard reset (rare
        device-error case), the caller should use ``reset_stream()``
        explicitly.

        H11: ``flush()`` clears the deque and resets the running sample
        total in one atomic step under the lock.
        """
        with self._lock:
            self._chunks.clear()
            self._total_samples = 0
            self._ring_buffer_drained.set()
            # Mirror the threading event onto the asyncio event so a
            # barged-in wait_until_drained() does not block forever on
            # a stale clear.
            if self._drained_async is not None:
                self._drained_async.set()

    async def wait_until_drained(self, timeout: float = 30.0) -> None:
        """Block until the ring buffer is fully drained by the PortAudio callback.

        Signal-based: the PortAudio callback (audio thread) sets the
        ``_drained_async`` asyncio event the instant the buffer
        becomes empty. This replaces the old 3s polling cap that
        used to cut off long TTS mid-playback (H10 / H28).

        Parameters
        ----------
        timeout : float
            Safety net in seconds. The callback normally signals
            completion well within milliseconds; this is only there
            to keep a runaway edge case from blocking the orchestrator
            forever. Default 30s is plenty for any reasonable TTS.
        """
        if self._stream is None or self._drained_async is None:
            return
        # Yield to the audio thread so it can pick up freshly queued
        # samples and clear the event before we start waiting.
        await asyncio.sleep(0.005)
        try:
            await asyncio.wait_for(
                self._drained_async.wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._log.warning(
                "wait_until_drained_timeout",
                timeout=timeout,
                pending_samples=self.pending_samples,
            )

    async def stop(self) -> None:
        if self._stream is not None:
            stream = self._stream
            self._stream = None
            try:
                def _close():
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                t = threading.Thread(target=_close)
                t.start()
                t.join(timeout=1.0)
            except Exception:
                pass
            self._log.info("output_stream_closed")

        with self._lock:
            # H11: clear the deque and the running sample total
            self._chunks.clear()
            self._total_samples = 0
            if self._drained_async is not None:
                self._drained_async.set()

    async def __aenter__(self) -> AudioOutput:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
