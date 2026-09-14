"""OpenWakeWord listener.

Listens on the microphone input stream and yields events when the
wake word is detected. Runs on CPU with ONNX runtime.

Usage:
    detector = WakeWordDetector(cfg)
    async for detection in detector.listen(audio_input):
        print("Wake word detected!")
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from core.config import WakeWordConfig
from core.errors import WakeWordError
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.audio import AudioInput


class WakeWordDetector:
    """Detects the configured wake word using OpenWakeWord."""

    _active_instance: "WakeWordDetector | None" = None

    def __init__(self, cfg: WakeWordConfig) -> None:
        self._cfg = cfg
        self._log = get_logger("wake_word")
        self._model = None  # lazy-loaded
        self._last_trigger: float = 0.0
        self._cooldown_s = cfg.cooldown_ms / 1000.0
        self._load_failed = False  # prevent infinite retries
        # Score of the most recent wake trigger — read by the orchestrator
        # to distinguish "Hey Jarvis" (high score ≥ 0.70) from just "Jarvis"
        # (lower score near threshold) without consuming any mic audio.
        self._last_trigger_score: float = 0.0
        self._manual_wake_event: asyncio.Event = asyncio.Event()
        WakeWordDetector._active_instance = self

    def _preflight_check(self) -> tuple[bool, str]:
        """Verify OpenWakeWord resources are present before instantiating the model.

        Returns (ok, message). When ok is False, message contains an actionable
        hint for the user (e.g. which file is missing, how to install).
        """
        try:
            import openwakeword
        except ImportError as e:
            return False, (
                "openwakeword package is not installed. "
                "Run: pip install openwakeword"
            )

        oww_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        melspec = oww_dir / "melspectrogram.onnx"
        embedding = oww_dir / "embedding_model.onnx"

        missing: list[str] = []
        if not melspec.exists():
            missing.append(str(melspec))
        if not embedding.exists():
            missing.append(str(embedding))

        if missing:
            try:
                import openwakeword.utils
                self._log.info("downloading_missing_wakeword_models", missing=missing)
                openwakeword.utils.download_models(["hey_jarvis"])
                if melspec.exists() and embedding.exists():
                    return True, ""
            except Exception as dl_err:
                self._log.warning("auto_download_wakeword_failed", error=str(dl_err))

            return False, (
                "OpenWakeWord resource files are missing:\n  - "
                + "\n  - ".join(missing)
                + "\nFix: python -c 'import openwakeword.utils; openwakeword.utils.download_models([\"hey_jarvis\"])'"
            )

        return True, ""

    async def _load_model(self) -> None:
        """Load OpenWakeWord model (lazy, first call).

        On any load failure, sets ``_load_failed = True`` and raises
        ``WakeWordError`` with an actionable message. The orchestrator's
        error handler should treat this as a hard-stop (see
        ``_handle_error``); subsequent calls to ``_load_model`` short-circuit
        so we never enter an infinite error-recovery loop.
        """
        if self._model is not None:
            return
        if self._load_failed:
            raise WakeWordError(
                "Wake word model is unavailable. See earlier log lines for the "
                "actionable fix. Restart JARVIS after addressing the issue."
            )

        ok, preflight_msg = self._preflight_check()
        if not ok:
            self._load_failed = True
            self._log.error("wakeword_preflight_failed", hint=preflight_msg)
            raise WakeWordError(f"Wake word preflight failed: {preflight_msg}")

        try:
            import openwakeword

            models_to_load: list[str] = []
            model_path = Path(self._cfg.model_path)
            if model_path.exists():
                models_to_load.append(str(model_path))
            elif getattr(self._cfg, "fallback_builtin", None):
                models_to_load.append(self._cfg.fallback_builtin)
            else:
                models_to_load = ["hey_jarvis"]

            self._model = openwakeword.Model(
                wakeword_models=models_to_load,
                inference_framework="onnx",
            )
            self._log.info("models_loaded", models=models_to_load)

            # Warm up model buffers with 10 silence frames to clear initial internal FIFOs
            zero_frame = np.zeros(1280, dtype=np.int16)
            for _ in range(10):
                self._model.predict(zero_frame)

        except WakeWordError:
            raise
        except Exception as e:
            self._load_failed = True
            hint = (
                "The openwakeword package may be corrupted or installed against "
                "a different Python interpreter. Try: "
                "pip install --force-reinstall openwakeword"
            )
            self._log.error(
                "wakeword_load_failed",
                error=str(e),
                hint=hint,
            )
            raise WakeWordError(f"Failed to load wake word model: {e}. {hint}") from e

    @property
    def is_available(self) -> bool:
        """Whether the wake word detector is loaded and ready."""
        return self._model is not None

    async def detect(self, audio: AudioInput) -> AsyncIterator[float]:
        """Async generator that yields confidence scores from each audio frame.

        _load_model() raises WakeWordError on failure, so self._model is
        guaranteed to be set after it returns. No None-guard branch needed.
        """
        await self._load_model()
        startup_grace_frames = 12  # Ignore first ~1s of audio frames (hardware clicks/pops)

        while True:
            frame = await audio.read_frame()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            audio_int16 = (frame.samples * 32767).astype(np.int16) if frame.samples.dtype == np.float32 else frame.samples
            predictions = self._model.predict(audio_int16)
            if startup_grace_frames > 0:
                startup_grace_frames -= 1
                yield 0.0
                continue
            score = max(predictions.values()) if predictions else 0.0
            yield score

    def trigger_wake(self) -> None:
        """Trigger wake-up immediately (e.g. from HUD Arc Reactor click or hotkey)."""
        self._log.info("trigger_wake_called")
        self._manual_wake_event.set()

    async def wait_for_wake(self, audio: AudioInput, timeout_s: float | None = None) -> bool:
        """Block until wake word is detected OR Enter is pressed OR manual wake triggered.

        When ``keyboard_trigger`` is True (always-on mode), methods run
        concurrently via asyncio.wait — whichever fires first wins:
          • Say "Hey Jarvis" / "Jarvis" → wake word model triggers
          • Press Enter                 → keyboard fallback triggers
          • Click HUD Arc Reactor       → manual trigger fires
        """
        # Always run manual wake and wake word detection
        wake_event = asyncio.Event()
        stop_event = asyncio.Event()

        async def _wake_word_task() -> bool:
            result = await self._wait_for_wake_word(audio, timeout_s, stop_event=stop_event)
            if result:
                wake_event.set()
            return result

        async def _manual_wake_task() -> bool:
            await self._manual_wake_event.wait()
            self._manual_wake_event.clear()
            self._last_trigger_score = 0.85
            self._log.info("manual_wake_activated")
            wake_event.set()
            return True

        tasks = [
            asyncio.create_task(_wake_word_task()),
            asyncio.create_task(_manual_wake_task()),
        ]

        if self._cfg.keyboard_trigger and sys.stdin and sys.stdin.isatty():
            async def _enter_key_task() -> bool:
                loop = asyncio.get_running_loop()
                self._log.info("keyboard_trigger_ready", instruction="Press Enter OR say 'Hey Void' / 'Void'")
                fut = loop.create_future()

                def _on_stdin():
                    try:
                        sys.stdin.readline()
                        if not fut.done():
                            fut.set_result(True)
                    except (EOFError, KeyboardInterrupt) as err:
                        if not fut.done():
                            fut.set_exception(err)
                    except Exception as err:
                        if not fut.done():
                            fut.set_exception(err)

                use_reader = False
                try:
                    loop.add_reader(sys.stdin.fileno(), _on_stdin)
                    use_reader = True
                except (NotImplementedError, AttributeError, Exception):
                    use_reader = False

                if use_reader:
                    try:
                        await fut
                        if not stop_event.is_set():
                            self._log.info("keyboard_trigger_activated")
                            wake_event.set()
                        return True
                    except (EOFError, KeyboardInterrupt):
                        await stop_event.wait()
                        return False
                    finally:
                        try:
                            loop.remove_reader(sys.stdin.fileno())
                        except Exception:
                            pass
                else:
                    def _thread_target():
                        try:
                            input()
                            if not fut.done():
                                loop.call_soon_threadsafe(fut.set_result, True)
                        except (EOFError, KeyboardInterrupt) as err:
                            if not fut.done():
                                loop.call_soon_threadsafe(fut.set_exception, err)
                        except Exception as err:
                            if not fut.done():
                                loop.call_soon_threadsafe(fut.set_exception, err)

                    import threading
                    t = threading.Thread(target=_thread_target, daemon=True, name="jarvis_keyboard_trigger")
                    t.start()
                    try:
                        await fut
                        if not stop_event.is_set():
                            self._log.info("keyboard_trigger_activated")
                            wake_event.set()
                        return True
                    except (EOFError, KeyboardInterrupt):
                        await stop_event.wait()
                        return False

            tasks.append(asyncio.create_task(_enter_key_task()))

        # Wait for whichever fires first.
        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stop_event.set()
            for t in tasks:
                if not t.done():
                    t.cancel()

        for t in done:
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    raise exc

        return wake_event.is_set()

    async def _wait_for_wake_word(
        self,
        audio: AudioInput,
        timeout_s: float | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> bool:
        """Core wake-word detection loop. Returns True when threshold crossed."""
        threshold = self._cfg.threshold
        start = time.monotonic()
        best_recent = 0.0
        frames = 0

        async for score in self.detect(audio):
            if stop_event is not None and stop_event.is_set():
                return False

            now = time.monotonic()
            if timeout_s is not None and (now - start) > timeout_s:
                return False

            if score > best_recent:
                best_recent = score

            frames += 1
            # Log peak scores every ~2 seconds so you can tune your speaking volume
            if frames % 30 == 0:
                self._log.info("wake_scores", current=round(score, 4), best=round(best_recent, 4), thresh=threshold)

            if score > threshold and (now - self._last_trigger) > self._cooldown_s:
                self._last_trigger = now
                self._last_trigger_score = score  # saved for orchestrator voice switching
                self._log.info("wake_triggered", score=round(score, 4))
                return True

        return False

    def predict(self, audio_int16: np.ndarray) -> float:
        """Predict wake word score on a single int16 audio frame.

        Returns 0.0 if the model isn't loaded. Used by the barge-in monitor
        during TTS playback for single-frame scoring without the async generator.
        """
        if self._model is None:
            return 0.0
        try:
            predictions = self._model.predict(audio_int16)
            return max(predictions.values()) if predictions else 0.0
        except Exception:
            return 0.0

    def reset(self) -> None:
        """Clear internal state (called after detection to prepare for next listen)."""
        if self._model is not None:
            self._model.reset()
        self._last_trigger = 0.0

    async def cleanup(self) -> None:
        """Release model resources."""
        self._model = None
