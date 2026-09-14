"""Acoustic Echo Cancellation — Direct Buffer Reference.

Phase 1 rewrite. Replaces the unstable NLMS adaptive filter with a simpler,
more reliable approach:

  - The TTS output buffer is used as the PERFECT reference signal.
    We know EXACTLY what audio was sent to the speaker (it's our own code).
  - Reference is resampled (22050 Hz → 16000 Hz) and stored in a ring buffer.
  - When a mic frame arrives, the delayed reference is subtracted with an
    adaptive gain scalar.
  - No filter divergence. No voice cancellation. Stable by construction.

Why this beats WASAPI Loopback for Jarvis specifically:
  - The reference is digitally perfect (no soundcard ADC noise).
  - We know exact send time — no external loopback device needed.
  - Works on any OS, not just Windows.
  - pyaudiowpatch dependency eliminated.

Acoustic delay compensation:
  Speaker → air → mic delay is typically 20–50ms on a laptop.
  Configured via aec.delay_ms in config.yaml (default: 25ms).

Usage:
    aec = DirectBufferAEC(cfg)
    # Called when TTS audio is fed to speaker (AudioOutput.play callback):
    aec.feed_reference(tts_samples_float32)
    # Called on each mic frame:
    clean_frame = aec.process(mic_frame_float32)
    # Enable only during TTS playback:
    aec.active = True   # on TTS start
    aec.active = False  # on TTS end
"""

from __future__ import annotations

import threading

import numpy as np


class DirectBufferAEC:
    """Echo canceller using the TTS output buffer as reference signal.

    Parameters
    ----------
    filter_length : int
        Ring buffer size in samples (at mic sample rate).
        Default 32000 = 2 seconds at 16 kHz — enough for any room reverb.
    delay_ms : int
        Estimated speaker→mic acoustic travel delay in milliseconds.
        25ms is typical for laptop speakers. Increase for external speakers.
    mic_sample_rate : int
        Microphone sample rate. Must be 16000.
    tts_sample_rate : int
        TTS output sample rate. Must match Piper voice model.
    """

    def __init__(
        self,
        filter_length: int = 32000,
        delay_ms: int = 25,
        mic_sample_rate: int = 16000,
        tts_sample_rate: int = 22050,
    ) -> None:
        self._mic_sr = mic_sample_rate
        self._tts_sr = tts_sample_rate
        self._delay_samples = int(mic_sample_rate * delay_ms / 1000)
        self._ring_size = filter_length + self._delay_samples + 4096

        # Ring buffer for resampled TTS reference (at mic sample rate)
        self._ref_ring: np.ndarray = np.zeros(self._ring_size, dtype=np.float32)
        self._write_pos: int = 0
        self._lock = threading.Lock()

        # Adaptive gain: scalar ratio of mic energy to reference energy.
        # Starts at 0.8 (typical room acoustic gain), adapts slowly.
        self._gain: float = 0.8
        self._gain_alpha: float = 0.02  # EMA smoothing for gain adaptation

        # AEC is active only during TTS playback — never during IDLE.
        self._active: bool = False

    # ------------------------------------------------------------------
    # Reference feed (called from AudioOutput.play callback)
    # ------------------------------------------------------------------

    def feed_reference(self, audio: np.ndarray) -> None:
        """Write TTS output into the reference ring buffer.

        Called automatically by AudioOutput when audio is enqueued.
        Audio is at TTS sample rate (22050 Hz); resampled to mic rate (16000 Hz).
        """
        if audio.size == 0:
            return

        src = audio.ravel().astype(np.float32)

        # Linear interpolation resample: 22050 → 16000 Hz
        n_src = len(src)
        ratio = self._mic_sr / self._tts_sr
        n_dst = max(1, int(n_src * ratio))
        src_idx = np.linspace(0, n_src - 1, n_dst)
        lo = np.floor(src_idx).astype(int)
        hi = np.minimum(lo + 1, n_src - 1)
        frac = (src_idx - lo).astype(np.float32)
        resampled = (src[lo] * (1.0 - frac) + src[hi] * frac).astype(np.float32)

        with self._lock:
            self._write_to_ring(resampled)

    def _write_to_ring(self, data: np.ndarray) -> None:
        """Write data into the ring buffer (called under lock)."""
        n = len(data)
        if n > self._ring_size:
            # If data is larger than the ring buffer, keep only the most recent part
            data = data[-self._ring_size:]
            n = len(data)

        end = self._write_pos + n
        if end <= self._ring_size:
            self._ref_ring[self._write_pos:end] = data
        else:
            first = self._ring_size - self._write_pos
            self._ref_ring[self._write_pos:] = data[:first]
            self._ref_ring[: end - self._ring_size] = data[first:]
        self._write_pos = end % self._ring_size

    # ------------------------------------------------------------------
    # Process (called on every mic frame)
    # ------------------------------------------------------------------

    def process(self, mic: np.ndarray) -> np.ndarray:
        """Subtract reference from mic frame. Pass-through when inactive.

        Parameters
        ----------
        mic : np.ndarray
            Float32 microphone frame at 16000 Hz.

        Returns
        -------
        np.ndarray
            Echo-cancelled frame. Safe range: [-1.0, 1.0].
        """
        if not self._active:
            return mic

        n = len(mic)
        with self._lock:
            ref = self._read_delayed_ref(n)

        # Adaptive gain: estimate from RMS ratio when both signals have energy
        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        mic_rms = float(np.sqrt(np.mean(mic.astype(np.float64) ** 2)))

        if ref_rms > 1e-5 and mic_rms > 1e-5:
            target_gain = min(mic_rms / ref_rms, 2.0)  # cap at 2x to prevent explosion
            self._gain = (1.0 - self._gain_alpha) * self._gain + self._gain_alpha * target_gain

        # Subtract and clip to prevent overflow
        cleaned = mic.astype(np.float64) - ref.astype(np.float64) * self._gain
        return np.clip(cleaned, -1.0, 1.0).astype(np.float32)

    def _read_delayed_ref(self, n: int) -> np.ndarray:
        """Read n samples from ring buffer, offset by acoustic delay."""
        # Read from position (write_pos - delay - n) to (write_pos - delay)
        read_end = (self._write_pos - self._delay_samples) % self._ring_size
        read_start = (read_end - n) % self._ring_size

        if read_start < read_end:
            return self._ref_ring[read_start:read_end].copy()
        else:
            # Wrap-around read
            part1 = self._ref_ring[read_start:]
            part2 = self._ref_ring[:read_end]
            result = np.concatenate([part1, part2])
            # Trim or pad to exactly n samples
            if len(result) >= n:
                return result[:n]
            return np.pad(result, (0, n - len(result)))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True only during TTS playback — never during IDLE or LISTENING."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        if value and not self._active:
            # Entering active mode — reset gain to avoid stale estimates
            self._gain = 0.8
        self._active = value

    def reset(self) -> None:
        """Clear all state (e.g. on device switch or stream restart)."""
        with self._lock:
            self._ref_ring.fill(0.0)
            self._write_pos = 0
        self._gain = 0.8
        self._active = False
