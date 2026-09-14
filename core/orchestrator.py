"""Voice loop orchestrator — master state machine.

Runs the voice loop: IDLE → WAKE → LISTEN → TRANSCRIBE → THINK → SPEAK → IDLE.
Supports barge-in (user interrupts JARVIS during speech).

Phase 2 addition: tool execution loop in _think_state().
When the LLM returns a ToolCallRequest, the orchestrator:
  1. Speaks a brief acknowledgment
  2. Executes the tool via ToolExecutor
  3. Feeds the result back to the Brain
  4. Repeats until the Brain returns text-only (max 5 calls per turn)

State machine:
  IDLE        - Waiting for wake word. Reads mic frames, feeds to wake detector.
  LISTEN      - Wake detected. VAD collects utterance until end-of-speech.
  TRANSCRIBE  - Utterance sent to faster-whisper for STT.
  THINK       - Text sent to Brain. Tokens stream back. Tool calls intercepted.
  SPEAK       - Tokens split into sentences, synthesized by Piper, played.
                Barge-in monitored during playback.
  ERROR       - Any stage error → spoken apology → IDLE.
"""

from __future__ import annotations

import asyncio
import time
import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from core.audio import AudioInput, AudioOutput
from core.brain_router import BrainRouter, BrainType
from core.config import Config
from core.conversation import ConversationHistory
from core.errors import (
    GPUOutOfMemoryError,
    JarvisError,
    LLMError,
    STTError,
    TTSError,
    WakeWordError,
)
from core.event_bus import EventBus, EventType
from core.latency import LatencyTimer, STAGE_STT, STAGE_TOTAL, STAGE_FIRST_TOKEN
from core.llm import LLMClient
from core.types import ToolCallRequest
# H5: typed ToolSignal enum. Imported at module top so both
# ``_think_state`` (line ~1115) and ``_execute_tool_call`` (line
# ~1213) can reference it. The old code did a local import inside
# ``_execute_tool_call`` only, which left ``_think_state`` with a
# ``NameError: ToolSignal not defined`` whenever a tool with a
# non-NONE signal (reboot, sleep, shutdown) was executed.
try:
    from tools.registry import ToolResult, ToolSignal  # noqa: F401
except ImportError:  # pragma: no cover
    ToolResult = None  # type: ignore
    ToolSignal = None  # type: ignore
from core.latency import LatencyTimer, STAGE_STT, STAGE_TOTAL, STAGE_FIRST_TOKEN
from core.logging_setup import get_logger
from core.stt import STTEngine
from core.streaming import SentenceSplitter
from core.tts import TTSManager
from core.vad import VADProcessor
from core.wake_word import WakeWordDetector


class VoiceOrchestrator:
    """Master state machine coordinating all voice pipeline components."""

    # C8 fix: the re_enroll_voiceprint tool needs a handle on the live
    # orchestrator's audio + VAD + TTS helpers. The constructor sets
    # this to ``self``; tools.py reads it via
    # ``VoiceOrchestrator._active_instance``.
    _active_instance: "VoiceOrchestrator | None" = None

    def __init__(
        self,
        cfg: Config,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        wake_detector: WakeWordDetector,
        vad: VADProcessor,
        stt: STTEngine,
        brain_router: BrainRouter,
        tts: TTSManager,
        timer: LatencyTimer,
        bus: EventBus,
        tool_executor=None,       # ToolExecutor | None (Phase 2)
        tool_registry=None,       # ToolRegistry | None (Phase 2)
        wal=None,                 # MemoryWAL | None (Phase 1)
        vector_engine=None,       # VectorEngine | None (Phase 3)
        knowledge_graph=None,     # KnowledgeGraph | None (Phase 2)
    ) -> None:
        self._cfg = cfg
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._wake = wake_detector
        self._vad = vad
        self._stt = stt
        self._brain_router = brain_router
        self._tts = tts
        self._timer = timer
        self._bus = bus
        self._tool_executor = tool_executor
        self._tool_registry = tool_registry
        self._log = get_logger("orchestrator")
        self._shutdown = False
        self._tts.set_audio_output(audio_output)

        # C8 fix: expose this orchestrator as the active instance so the
        # re_enroll_voiceprint tool can borrow the audio + VAD + TTS
        # helpers. The attribute is class-level so it works regardless
        # of which module imports it first.
        VoiceOrchestrator._active_instance = self
        # Adaptive VAD noise floor (see _drain_after_wake and
        # _listen_state for the threshold formula that uses this)
        self._noise_floor_rms: float = 0.005

        # 2026-06-20: manual-barge-in event. Pressing
        # Ctrl+Space (or any other configured hotkey)
        # sets this event, which the speak_state loop
        # watches in addition to the VAD / wake-word
        # triggers. This is the always-works escape
        # hatch: it doesn't depend on AEC, VAD, or the
        # microphone — just a direct OS-level keyboard
        # signal. The orchestrator-side check is in
        # _barge_in_monitor; the keyboard listener is
        # started by start_push_to_talk() (called from
        # run()).
        self._manual_barge_in: asyncio.Event = asyncio.Event()
        self._push_to_talk_listener = None

        # Consecutive error counter for circuit breaker
        self._consecutive_errors = 0
        
        # Security mode
        self._guest_mode = False

        # Barge-in state
        self._current_sentence: str = ""
        self._last_interrupted_fragment: str | None = None

        # Post-speak silence gate — suppress wake word within this window
        # after TTS finishes, preventing reverb from false-triggering OWW
        self._post_tts_cooldown_until: float = 0.0

        # In-session conversation history (Phase 1 — RAM + WAL disk logging)
        self._wal = wal
        self._vector_engine = vector_engine
        self._knowledge_graph = knowledge_graph
        self._history = ConversationHistory(
            max_turns=cfg.conversation.max_turns if cfg.conversation.enabled else 0,
            wal=self._wal
        )

        # H23: handle to the in-flight weekly-observer task so the
        # finally block in ``run()`` can wait briefly for it on
        # shutdown. Initialised to None; the real task is created
        # at the top of ``run()``.
        self._observer_task: asyncio.Task | None = None

        # M7: when the reboot_jarvis tool returns, the orchestrator
        # defers the handover until the TTS drain completes. These
        # two attributes carry the state across the speak state.
        self._reboot_pending: bool = False
        self._reboot_metadata: dict[str, Any] = {}

        # M20: cache the workstyle file's content + mtime so we
        # only re-read the file when the observer rewrites it
        # (which happens at most once per week). The cache is
        # a plain dict so it can hold heterogeneous types
        # without us writing a dataclass for a 3-field bag.
        self._workstyle_cache: dict[str, Any] | None = None

        # Phase 2: Cache tool schemas (generated once on startup)
        self._gemini_tool_declarations: list[dict[str, Any]] | None = None
        self._ollama_tool_schema: str | None = None
        self._tool_call_format_template: str | None = None

        if tool_registry and tool_registry.count > 0 and cfg.tools.enabled:
            self._gemini_tool_declarations = tool_registry.as_gemini_declarations()
            
            # Prune heavy tools from local model to fix TTFT latency
            ollama_schema = tool_registry.as_ollama_prompt_schema(exclude_categories={"vision", "desktop"})

            # Load the tool call format template
            template_path = cfg.resolve("prompts/tool_call_format.txt")
            try:
                template = template_path.read_text(encoding="utf-8").strip()
                self._ollama_tool_schema = template.replace("{schema}", ollama_schema)
            except FileNotFoundError:
                self._log.warning("tool_call_format_prompt_not_found")
                self._ollama_tool_schema = (
                    f"When you need to use a tool, respond with this JSON on its own line:\n"
                    f'{{"tool_call": {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}}}\n\n'
                    f"Available tools:\n{ollama_schema}"
                )

            self._log.info(
                "tools_enabled",
                tool_count=tool_registry.count,
                gemini_declarations=len(self._gemini_tool_declarations),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Enter the voice loop. Runs until `shutdown()` is called."""
        self._log.info("orchestrator_started")
        self._bus.publish(EventType.IDLE)

        if self._cfg.conversation.enabled and self._wal:
            try:
                await self._history.load_from_wal()
                self._log.info("history_restored_from_wal", turns=self._history.turn_count)
            except Exception as e:
                self._log.warning("failed_to_restore_history", error=str(e))


        # H23: kick off the weekly observer in the BACKGROUND. The voice
        # loop no longer waits for the observer (which can take 5-30 s on
        # the local Qwen) before becoming responsive. The workstyle.md
        # file is read fresh on every turn, so the worst case is that
        # the first few turns use a stale workstyle — a hint, not a
        # critical dependency. The task is held in ``self._observer_task``
        # so the finally block can wait briefly for it to finish.
        from memory.observer import run_observer
        db_path = self._cfg.resolve(self._cfg.paths.audit_log_db)
        workstyle_path = self._cfg.resolve(self._cfg.paths.workstyle)
        last_run_file = self._cfg.resolve(self._cfg.paths.data_dir) / ".observer_last_run"
        model_name = self._cfg.brains.brain1_primary.name

        async def _run_observer_safe() -> None:
            try:
                # Use the cloud brain when available — in cloud_only mode
                # self._brain_router._llm is the Ollama client which is
                # offline, causing observer_failed on every startup.
                # Prefer the cloud client; fall back to Ollama only when
                # the cloud client is not configured.
                gemini = self._brain_router._gemini
                gemini_ok = (
                    gemini is not None
                    and self._cfg.brains.brain2_specialist.enabled
                    and not getattr(gemini, "is_quota_blocked", False)
                )
                observer_llm = gemini if gemini_ok else self._brain_router._llm
                self._log.debug(
                    "observer_using_brain",
                    brain="cloud" if gemini_ok else "local_ollama",
                )
                await run_observer(
                    db_path=db_path,
                    workstyle_path=workstyle_path,
                    llm_client=observer_llm,
                    model_name=model_name,
                    last_run_file=last_run_file,
                )
            except Exception as e:
                self._log.error("weekly_observer_failed", error=str(e))

        self._observer_task = asyncio.create_task(_run_observer_safe())
        self._log.info("observer_scheduled_in_background")

        # Start background telemetry loop
        stats_task = asyncio.create_task(self._stats_loop())

        # 2026-06-20: install the Ctrl+Space push-to-talk
        # keyboard hook. The listener runs on its own
        # thread (Windows low-level hook) and posts to the
        # asyncio loop via call_soon_threadsafe. No-op on
        # non-Windows or if disabled in config.
        if self._cfg.orchestrator.push_to_talk_enabled:
            self.start_push_to_talk()

        # H24: first-run voiceprint enrollment. Previously this blocked
        # the voice loop indefinitely if the user did not speak during
        # the 10 s capture window. The new timeout is
        # ``orchestrator.enrollment_timeout_s`` (default 30 s). On
        # timeout we discard any partial state and continue; the user
        # can re-enroll later via the ``re_enroll_voiceprint`` tool.
        if self._cfg.verification.enabled:
            from security.biometrics import has_voiceprint
            if not has_voiceprint():
                self._log.info("no_voiceprint_found_starting_enrollment")
                await self._enrollment_state_with_timeout()

        try:
            while not self._shutdown:
                try:
                    await self._idle_state()
                    self._consecutive_errors = 0
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await self._handle_error(e)
        finally:
            self._log.info("orchestrator_stopping")
            # H23: wait briefly for the in-flight observer to finish so
            # its workstyle.md write is not lost. Configurable grace via
            # ``orchestrator.observer_shutdown_grace_s`` (default 5 s).
            grace = getattr(self._cfg.orchestrator, "observer_shutdown_grace_s", 5.0)
            if getattr(self, "_observer_task", None) is not None and not self._observer_task.done():
                try:
                    await asyncio.wait_for(self._observer_task, timeout=grace)
                    self._log.info("observer_completed_on_shutdown")
                except asyncio.TimeoutError:
                    self._observer_task.cancel()
                    self._log.warning("observer_shutdown_timeout", grace_s=grace)
                except Exception as e:
                    self._log.error("observer_shutdown_error", error=str(e))
            # (Compaction has been replaced by the Write-Ahead Log engine)
            if getattr(self, "_history", None) is not None:
                try:
                    await self._history.drain_wal()
                except Exception as e:
                    self._log.warning("wal_drain_on_shutdown_failed", error=str(e))

            self._log.info("orchestrator_stopped")

    def _reset_consecutive_errors(self) -> None:
        """H30: clear the error backoff counter at the start of every state.

        The previous implementation only reset ``_consecutive_errors``
        at the end of ``_idle_state`` (line 217). An error in any
        other state (``_wake_event``, ``_listen_state``,
        ``_think_state``, ``_speak_state``) would push the counter
        up, and if the next iteration's ``_idle_state`` also errored
        before returning successfully, the counter accumulated across
        unrelated state transitions. With 5 errors you hit the 8 s
        backoff cap. The fix is to reset at the start of every
        state so a fresh budget is available for the new work.
        """
        if self._consecutive_errors:
            self._log.debug("resetting_consecutive_errors", was=self._consecutive_errors)
            self._consecutive_errors = 0

    async def _enrollment_state_with_timeout(self) -> None:
        """H24: wrapper around ``_enrollment_state`` with a configurable timeout.

        On timeout we discard any partial voiceprint that
        ``enroll_voiceprint_async`` may have written, log a warning,
        speak a brief message to the user, and return so the voice
        loop can start. The user can re-enroll later via the
        ``re_enroll_voiceprint`` voice command.
        """
        timeout_s = float(getattr(self._cfg.orchestrator, "enrollment_timeout_s", 30.0))
        try:
            await asyncio.wait_for(self._enrollment_state(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._log.warning(
                "enrollment_timeout",
                timeout_s=timeout_s,
                hint="user can re-enroll via the 're enroll voiceprint' voice command",
            )
            # Best-effort cleanup of a partial voiceprint
            try:
                from security.biometrics import voiceprint_path
                vp_path = voiceprint_path()
                if vp_path.exists():
                    vp_path.unlink()
                    self._log.info("enrollment_partial_voiceprint_removed", path=str(vp_path))
            except Exception as e:
                self._log.warning("enrollment_partial_cleanup_failed", error=str(e))
            # Briefly apologise and continue. If TTS itself is broken
            # this will fail silently, which is fine.
            try:
                await asyncio.wait_for(
                    self._tts.speak_sync(
                        "I didn't catch your voice in time. You can re-enroll later "
                        "by saying 're enroll voiceprint'."
                    ),
                    timeout=5.0,
                )
            except Exception:
                pass
            # Run the post-speak settle so we don't false-trigger the
            # wake word on the TTS tail.
            try:
                await self._post_speak_settle()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Post-speak settle gate
    # ------------------------------------------------------------------

    async def _post_speak_settle(self) -> None:
        """Gate after TTS playback — signal-based, no heuristics.

        Three stages:
        1. TTS playback queue drained — all items synthesized + sent to AudioOutput
        2. AudioOutput ring buffer drained — all samples consumed by PortAudio callback
           (H10/H28: signal-based; replaces the old 3s polling cap that
           used to cut off long TTS mid-playback)
        3. Physical settle — speaker ring + room reverb guard
           (H31: configurable via ``orchestrator.post_speak_settle_ms``,
           default 300 ms vs the old hardcoded 1000 ms)
        """
        # 1. TTS playback queue — wait for all items to be dequeued and sent to AudioOutput
        try:
            await asyncio.wait_for(
                self._tts.wait_until_queue_drained(),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            self._log.warning("tts_drain_timeout")

        # 2. AudioOutput ring buffer — signal-based wait, returns the
        # instant the PortAudio callback consumes the last sample.
        await self._audio_output.wait_until_drained()

        # 3. Physical settle — hardware buffer tail + speaker ring + room reverb.
        # Configurable so users with Bluetooth speakers or noisy rooms can
        # raise it; users with good AEC can set it to 0.
        settle_ms = getattr(self._cfg.orchestrator, "post_speak_settle_ms", 300)
        if settle_ms > 0:
            await asyncio.sleep(settle_ms / 1000.0)
            
        # 4. Flush the audio input buffer. Since AEC was turned off before this
        # settle delay, the mic ring buffer has now captured the physical room 
        # echo/reverb of the TTS that just finished playing. If we don't clear 
        # it, the next read_frame() call (either from VAD follow-up or wake-word 
        # detector) will instantly read this echo and falsely trigger.
        if hasattr(self._audio_input, 'clear_buffer'):
            self._audio_input.clear_buffer()

    # ------------------------------------------------------------------
    # Barge-in monitor
    # ------------------------------------------------------------------

    async def _barge_in_monitor(
        self,
        play_task: asyncio.Task,
        barge_in: asyncio.Event,
    ) -> bool:
        """Monitor mic for ANY sustained speech during TTS playback.

        2026-06-20: redesigned from wake-word-only to a multi-signal
        barge-in with TTS ducking. The original implementation
        only interrupted when the user said "Hey Jarvis"
        mid-response, which is poor UX — the user has to say
        the wake word even though they're already in an active
        conversation.

        Why the VAD-only attempt was unreliable: on laptop
        built-in speakers, the mic picks up TTS audio that AEC
        doesn't fully cancel. The residual energy crosses the
        VAD threshold, barge-in fires on the BLEED (not the
        user's voice), STT hallucinates ("Clara", "Slap",
        "my name", "not"), the LLM responds, TTS plays, and
        the loop never ends. The user sees JARVIS "talking to
        itself".

        The fix is TTS ducking. The moment mic energy rises
        during TTS playback, we drop the speaker volume to
        ``duck_volume`` (default 10% = -20 dB) which kills the
        bleed at its source. After 150 ms of sustained
        speech-confirms-speech, we full-barge-in. If speech
        doesn't sustain (single spike from a cough or chair
        creak), we restore the volume and keep playing. This
        is the same pattern Siri / Google Assistant / ChatGPT
        voice use.

        Flow:
          1. Read the AEC-processed mic frame.
          2. Run VAD. If prob >= adaptive_threshold:
             a. Duck TTS to ``duck_volume`` IMMEDIATELY
                (single-frame latency).
             b. Continue counting consecutive speech frames.
             c. At ``min_speech_frames`` consecutive frames,
                full barge-in.
             d. If speech stops before that (cough, spike),
                restore TTS to 1.0 and keep playing.
          3. ALSO check the wake-word score as a backup
             (catches "Hey Jarvis" even if VAD is unhappy).

        Returns True on either trigger. Returns False when
        play_task finishes naturally.
        """
        threshold = self._cfg.wake_word.threshold
        consecutive_none = 0
        speech_frames = 0
        # Required sustained speech in ms before barge-in.
        min_speech_ms = self._cfg.orchestrator.barge_in_min_speech_ms
        # 80 ms per frame (faster-whisper frame size from audio.py)
        ms_per_frame = 80
        min_speech_frames = max(2, min_speech_ms // ms_per_frame)
        # Read the configured duck volume BEFORE the
        # device-class branch below, otherwise the
        # device_duck_volume = ... lines raise
        # UnboundLocalError (the function-local name
        # ``duck_volume`` is being assigned later, so
        # Python treats all references as local —
        # including the read on the device-class branch).
        duck_volume = self._cfg.orchestrator.barge_in_duck_volume
        # 2026-06-20: output-device-aware tuning. The
        # same barge-in parameters feel very different
        # on laptop built-in speakers (severe acoustic
        # feedback, hard mode) vs headphones (no
        # feedback, easy mode). We adjust the minimum
        # sustained speech and duck volume per device
        # class to optimise for each case.
        device_class = "unknown"
        try:
            if self._audio_output is not None:
                device_class = self._audio_output.device_class
        except Exception:
            pass
        # Per-class overrides. These are conservative
        # defaults that the global config can also
        # override; the per-class values here are the
        # last-resort defaults if the user hasn't tuned
        # their config.
        if device_class == "laptop_speakers":
            # Worst case: mic sits next to the speaker.
            # Require more sustained speech (avoids
            # false barge-in on consonants) and duck
            # harder (kill the bleed at the source).
            device_min_speech_ms = max(min_speech_ms, 480)
            device_duck_volume = min(duck_volume, 0.05)
        elif device_class in ("headphones", "bluetooth"):
            # Best case: no acoustic feedback, mic
            # hears only the user. Be aggressive.
            device_min_speech_ms = max(160, min_speech_ms - 80)
            device_duck_volume = max(duck_volume, 0.2)
        else:
            # Unknown — use the user's configured values.
            device_min_speech_ms = min_speech_ms
            device_duck_volume = duck_volume
        device_min_speech_frames = max(
            2, device_min_speech_ms // ms_per_frame
        )
        self._log.debug(
            "barge_in_tuning",
            device_class=device_class,
            min_speech_ms=device_min_speech_ms,
            duck_volume=device_duck_volume,
        )
        # Volume to duck to during suspected user speech.
        # 0.05-0.1 (-20 to -26 dB) is enough to kill the
        # bleed on laptop speakers while leaving the user
        # with audible context of what JARVIS was saying.
        duck_volume = device_duck_volume
        # Track whether we've ducked so we know to restore.
        ducked = False

        try:
            while not play_task.done() and not barge_in.is_set():
                # 2026-06-20: push-to-talk check. The user
                # pressing Ctrl+Space sets
                # ``_manual_barge_in``. This is the
                # always-works escape hatch — it doesn't
                # depend on audio quality at all. Reset
                # the event so the next press fires again.
                if self._manual_barge_in.is_set():
                    if ducked:
                        self._tts.set_volume(1.0)
                        ducked = False
                    self._manual_barge_in.clear()
                    self._log.info(
                        "barge_in_triggered",
                        trigger="push_to_talk",
                        source="Ctrl+Space",
                    )
                    barge_in.set()
                    return True
                frame = await self._audio_input.read_frame()
                if frame is None:
                    consecutive_none += 1
                    if consecutive_none >= 20:
                        self._log.warning("barge_in_monitor_aborted_device_failure")
                        return False
                    await asyncio.sleep(0.005)
                    continue
                consecutive_none = 0
                frame_data = frame.samples if hasattr(frame, 'samples') else frame

                # Wake-word check (backup signal).
                # Require high confidence (>= 0.85) to interrupt active speech,
                # preventing false barge-in triggers from acoustic reverb.
                audio_int16 = (frame_data * 32767).astype(np.int16)
                try:
                    score = self._wake.predict(audio_int16)
                    if score >= 0.85:
                        if ducked:
                            self._tts.set_volume(1.0)
                            ducked = False
                        self._log.info(
                            "barge_in_triggered",
                            trigger="wake_word",
                            score=round(score, 4),
                            threshold=0.85,
                        )
                        barge_in.set()
                        return True
                except Exception:
                    # Wake-word model hiccup must not break barge-in.
                    pass

                # VAD check (primary signal).
                # CRITICAL: Only enable acoustic VAD barge-in if using isolated headphones.
                # When playing through desktop/laptop speakers, speaker output feeds directly
                # back into the mic, causing VOID to falsely interrupt its own speech.
                if device_class in ("headphones", "bluetooth"):
                    try:
                        prob = self._vad.speech_probability(frame_data)
                    except Exception:
                        prob = 0.0
                    adaptive_threshold = self._cfg.vad.threshold
                    is_speech = prob >= adaptive_threshold
                    if is_speech:
                        if not ducked:
                            self._tts.set_volume(duck_volume)
                            ducked = True
                            self._log.debug(
                                "barge_in_tts_ducked",
                                volume=duck_volume,
                                prob=round(prob, 3),
                            )
                        speech_frames += 1
                        if speech_frames >= device_min_speech_frames:
                            self._log.info(
                                "barge_in_triggered",
                                trigger="vad",
                                speech_ms=speech_frames * ms_per_frame,
                                prob=round(prob, 3),
                                threshold=round(adaptive_threshold, 3),
                            )
                            barge_in.set()
                            return True
                    else:
                        if ducked:
                            self._tts.set_volume(1.0)
                            ducked = False
                            self._log.debug("barge_in_tts_restored")
                        speech_frames = 0
                else:
                    # Open speakers: avoid self-interruption. Rely on push-to-talk, HUD tap, or wake word.
                    speech_frames = 0

                await asyncio.sleep(0.005)
        finally:
            # Always restore volume when leaving the monitor
            # (normal completion, cancellation, exception).
            # The duck state may have leaked from a barge-in
            # if the play task ended before we set the flag.
            if ducked:
                try:
                    self._tts.set_volume(1.0)
                except Exception:
                    pass
        return False

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal the voice loop to exit at the next opportunity."""
        self._shutdown = True
        # 2026-06-20: stop the push-to-talk keyboard hook
        # on shutdown so the listener thread releases.
        try:
            if self._push_to_talk_listener is not None:
                self._push_to_talk_listener.stop()
        except Exception:
            pass

    def start_push_to_talk(self) -> None:
        """Install the Ctrl+Space keyboard hook for manual barge-in.

        2026-06-20: this is the always-works escape hatch
        for the user. The hook fires the
        ``_manual_barge_in`` event, which the
        ``_barge_in_monitor`` watches in addition to the
        VAD / wake-word triggers. Works regardless of
        audio quality.

        On non-Windows platforms this is a no-op. The
        listener can be disabled by setting
        ``orchestrator.push_to_talk.enabled = false`` in
        config.yaml.
        """
        try:
            from core.push_to_talk import PushToTalkListener
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called outside an asyncio context — skip.
            self._log.debug("push_to_talk_no_loop")
            return
        if self._push_to_talk_listener is not None:
            return  # already running
        listener = PushToTalkListener(
            loop=loop,
            on_activate=self._on_manual_barge_in,
        )
        if listener.start():
            self._push_to_talk_listener = listener
            self._log.info("push_to_talk_enabled", hotkey="Ctrl+Space")
        else:
            self._push_to_talk_listener = None
            # Expected on Linux — PTT uses Windows keyboard hooks.
            # push_to_talk.py already logs debug-level for the no-op.
            self._log.debug("push_to_talk_not_available", platform=__import__("sys").platform)

    def _on_manual_barge_in(self) -> None:
        """Called by the keyboard hook when Ctrl+Space is pressed.

        Sets the ``_manual_barge_in`` event so the
        ``_barge_in_monitor`` loop wakes up and triggers
        a full barge-in. Called via
        ``loop.call_soon_threadsafe`` from the hook
        thread, so this runs on the asyncio thread.
        """
        self._log.info("manual_barge_in_triggered", source="Ctrl+Space")
        self._manual_barge_in.set()

    async def _enrollment_state(self) -> None:
        """First-run flow to collect 10 seconds of audio for master voiceprint.

        C8 fix: this is now a thin wrapper around ``collect_enrollment_audio``
        and ``enroll_voiceprint_async`` so the same code path is used by the
        first-run startup and the re-enrollment tool.
        """
        self._bus.publish(EventType.SPEAKING)
        prompt = "I do not have a master voiceprint on file. Please state your name and speak continuously for 10 seconds to enroll your voice."
        self._log.info("starting_enrollment")

        await self._tts.speak_sync(prompt)
        await self._post_speak_settle()

        self._bus.publish(EventType.LISTENING)
        self._log.info("collecting_enrollment_audio")

        full_audio = await self._collect_enrollment_audio()

        self._bus.publish(EventType.THINKING)
        self._log.info("saving_voiceprint")

        from security.biometrics import enroll_voiceprint_async
        await enroll_voiceprint_async(full_audio)

        self._bus.publish(EventType.SPEAKING)
        await self._tts.speak_sync("Voiceprint locked. Welcome, sir.")
        await self._post_speak_settle()
        self._log.info("enrollment_complete")

    async def _collect_enrollment_audio(self, target_seconds: float = 10.0) -> np.ndarray:
        """Shared audio-collection loop for first-run and re-enrollment.

        Returns a 1-D float32 numpy array of ``target_seconds`` worth of
        mono 16 kHz audio. The VAD threshold is temporarily lowered to
        0.1 to capture everything during enrollment.

        H38: the previous implementation called ``set_threshold(0.1)``
        then ``reset_threshold()`` which only restored the static
        config default — it discarded the *adaptive* state the VAD
        had been computing. If the user had been running for a few
        minutes (so the VAD had learned their noise floor),
        enrollment would clobber that learning and the next listen
        state would re-start from scratch. We now snapshot the
        current effective threshold, set the enrollment value, and
        restore the snapshot on exit. The static-config default is
        used as a fallback if the VAD does not expose a getter.
        """
        import numpy as np

        # H38: snapshot the current effective threshold so the
        # adaptive state is preserved across enrollment.
        previous_threshold = getattr(
            self._vad, "_effective_threshold", self._cfg.vad.threshold
        )
        self._vad.set_threshold(0.1)

        sample_rate = 16000
        target_samples = int(target_seconds * sample_rate)
        collected: list[np.ndarray] = []
        current_samples = 0

        try:
            while current_samples < target_samples:
                utterance = await self._vad.collect_utterance(
                    lambda: self._audio_input.read_frame(),
                    max_duration_s=15.0,
                    max_consecutive_none=20,
                )
                if utterance is not None and len(utterance) > 0:
                    collected.append(utterance)
                    current_samples += len(utterance)
                    self._log.info(
                        "enrollment_progress",
                        seconds=round(current_samples / sample_rate, 2),
                    )
        finally:
            # H38: restore the adaptive threshold (or the config
            # default if we did not have an adaptive state).
            self._vad.set_threshold(previous_threshold)

        full = np.concatenate(collected)
        # H37: if the user spoke more than the requested duration,
        # we silently truncated to ``target_samples``. Log when
        # this happens so the user (or the developer) knows the
        # last few seconds of audio were dropped.
        if len(full) > target_samples:
            self._log.info(
                "enrollment_audio_trimmed",
                target_seconds=target_seconds,
                original_seconds=round(len(full) / sample_rate, 2),
                dropped_seconds=round((len(full) - target_samples) / sample_rate, 2),
            )
        return full[:target_samples]

    async def _idle_state(self) -> None:
        """IDLE: Read mic frames and feed to wake word detector."""
        # H30: each state starts with a fresh error budget.
        self._reset_consecutive_errors()
        # Wait out post-speak cooldown — prevents reverb from false-triggering OWW
        remaining = self._post_tts_cooldown_until - time.monotonic()
        if remaining > 0:
            self._log.debug("post_speak_cooldown", remaining_ms=round(remaining * 1000))
            await asyncio.sleep(remaining)

        self._bus.publish(EventType.IDLE)

        # Trigger same-session memory catch-up if idle for > 30 seconds
        now = time.monotonic()
        last_catch_up = getattr(self, "_last_catch_up_time", 0.0)
        if now - last_catch_up > 30.0 and self._wal and self._vector_engine and self._knowledge_graph:
            from core.memory.catch_up import run_catch_up
            asyncio.create_task(run_catch_up(self._wal, self._vector_engine, self._knowledge_graph))
            self._last_catch_up_time = now

        # Quick ambient noise snapshot before entering wake loop
        sample = await self._audio_input.read_frame()
        if sample is not None:
            rms = float(np.sqrt(np.mean(sample.samples.astype(np.float64) ** 2)))
            self._noise_floor_rms = rms
        self._log.debug("noise_floor", rms=round(self._noise_floor_rms, 6))

        wake_detected = await self._wake.wait_for_wake(self._audio_input)
        if wake_detected and not self._shutdown:
            await self._wake_event()

    async def _wake_event(self) -> None:
        """Wake word detected — begin a new conversational turn."""
        # H30: each state starts with a fresh error budget.
        self._reset_consecutive_errors()

        # --- Wake-phrase voice switching via wake word score ---
        # The previous implementation read 0.8s of mic audio and sent it
        # to Groq STT to detect "hey jarvis" vs "jarvis". This consumed
        # audio that belonged to the user's actual command, causing Groq
        # to receive only the tail of the utterance and hallucinate words
        # from the vocabulary prompt (e.g. "Sperry, Cricbuzz, ESPN").
        #
        # Fix: use the wake word detector's last score instead. The OWW
        # model was trained on "hey jarvis" — a high score (>0.7) means
        # the full phrase was detected; a lower score near threshold means
        # just "jarvis" was said. No audio consumed, zero latency.
        last_score = getattr(self._wake, "_last_trigger_score", None)
        if last_score is None:
            last_score = self._cfg.wake_word.threshold + 0.01
        if last_score >= 0.70:
            self._tts.set_active_voice("sexy")
            self._log.info("wake_voice_switched", mode="sexy", score=round(last_score, 4))
        else:
            self._tts.set_active_voice("normal")
            self._log.info("wake_voice_switched", mode="normal", score=round(last_score, 4))

        self._wake.reset()
        self._timer.begin_turn()
        self._timer.mark("wake")
        self._bus.publish(EventType.WAKE)

        # Play a short acknowledgment tone so the user knows they were heard
        await self._play_wake_chime()
        await self._audio_output.wait_until_drained()

        # Clear queue right after chime so chime echo isn't fed to VAD
        self._audio_input.clear_buffer()

        await self._listen_state()

    async def _drain_after_wake(self) -> None:
        """Clear queue so wake word tail is not passed to VAD."""
        self._audio_input.clear_buffer()

    async def _listen_state(self) -> None:
        """LISTEN: Collect utterance via VAD until end-of-speech or timeout."""
        # H30: each state starts with a fresh error budget.
        self._reset_consecutive_errors()
        self._bus.publish(EventType.LISTENING)
        self._timer.mark("listen_start")

        # Temporarily bypass adaptive threshold for debugging low mic issues
        self._vad.set_threshold(self._cfg.vad.threshold)

        utterance = await self._vad.collect_utterance(
            lambda: self._audio_input.read_frame(),
            max_duration_s=self._cfg.vad.max_utterance_ms / 1000.0,
            max_consecutive_none=20,
        )

        self._vad.reset_threshold()

        if utterance is None or utterance.size == 0:
            self._log.info("empty_utterance_timeout_return_to_idle")
            self._audio_input.clear_buffer()
            self._wake.reset()
            self._bus.publish(EventType.IDLE)
            return

        # Phase 3 Biometric Verification
        from security.biometrics import verify_voice, has_voiceprint
        if has_voiceprint():
            is_user = await asyncio.to_thread(verify_voice, utterance)
            if is_user:
                self._guest_mode = False
            else:
                self._log.warning("unrecognized_voice_guest_mode")
                self._guest_mode = True
        else:
            self._guest_mode = False  # Default to normal if not enrolled

        self._timer.mark("listen_end")
        await self._transcribe_state(utterance)

    async def _follow_up_state(self) -> None:
        """FOLLOW_UP: Continuous multi-turn listen loop after Jarvis speaks.

        2026-06-20: replaced the single-shot 15 s window with a
        continuous loop. After the first turn, JARVIS stays in
        this loop listening for the next command without
        requiring the wake word again. The loop terminates on:

          * ``idle_timeout_s`` of silence between turns
          * ``session_timeout_s`` total since the loop started
          * User says a "go to sleep" phrase (configurable)
          * Shutdown

        The ``_in_follow_up`` flag prevents ``_speak_state``
        from recursively re-entering the loop, which would
        nest follow-ups on every turn and never let the loop
        exit cleanly. The first call from ``_speak_state``
        enters the loop; subsequent calls see the flag and
        skip the call, so the loop continues its own
        iteration naturally.

        The single transcribe is reused: we transcribe the
        utterance, check the result against the sleep-phrase
        list, and either ack-and-return or feed the text into
        ``_think_state`` directly. This avoids a double STT
        round-trip on the wake-up phrase.
        """
        if getattr(self, "_in_follow_up", False):
            return  # already in a follow-up loop (re-entry from speak)
        self._in_follow_up = True

        try:
            fu = self._cfg.orchestrator.follow_up
            idle_timeout_s = getattr(fu, "idle_timeout_s", 30.0)
            session_timeout_s = getattr(fu, "session_timeout_s", 600.0)
            sleep_phrases = [
                p.lower().strip()
                for p in getattr(
                    fu,
                    "sleep_phrases",
                    ("go to sleep", "stop listening"),
                )
            ]

            if fu.headphones_only_warning:
                self._log.info(
                    "follow_up_active",
                    note="headphones_recommended",
                    idle_timeout_s=idle_timeout_s,
                    session_timeout_s=session_timeout_s,
                )
            else:
                self._log.debug(
                    "follow_up_window_open",
                    idle_timeout_s=idle_timeout_s,
                    session_timeout_s=session_timeout_s,
                )

            session_start = time.monotonic()

            try:
                while not self._shutdown and not getattr(self, "_skip_follow_up", False):
                    # Session cap.
                    if time.monotonic() - session_start > session_timeout_s:
                        self._log.info(
                            "follow_up_session_timeout",
                            elapsed_s=round(time.monotonic() - session_start, 1),
                        )
                        return

                    self._bus.publish(EventType.LISTENING)
                    self._timer.mark("listen_start")

                    # Apply adaptive VAD threshold (mirrors _listen_state).
                    adaptive_threshold = self._cfg.vad.threshold
                    self._vad.set_threshold(adaptive_threshold)

                    utterance = await self._vad.collect_utterance(
                        lambda: self._audio_input.read_frame(),
                        max_duration_s=idle_timeout_s,
                        max_consecutive_none=20,
                    )
                    self._vad.reset_threshold()

                    if utterance is None or utterance.size == 0:
                        self._log.info(
                            "follow_up_idle_timeout",
                            idle_s=idle_timeout_s,
                            elapsed_s=round(time.monotonic() - session_start, 1),
                        )
                        return

                    self._log.info(
                        "follow_up_utterance",
                        duration_s=round(len(utterance) / self._cfg.audio.sample_rate_in, 2),
                        elapsed_s=round(time.monotonic() - session_start, 1),
                    )
                    self._timer.mark("listen_end")

                    # Single transcribe. Reject very short noise.
                    utterance_s = len(utterance) / 16000
                    if utterance_s < 0.4:
                        self._log.info("follow_up_too_short", duration_s=f"{utterance_s:.2f}")
                        continue  # stay in the loop, listen again

                    self._bus.publish(EventType.TRANSCRIBING)
                    try:
                        result = await self._stt.transcribe(utterance)
                    except STTError as e:
                        self._log.error("follow_up_stt_failed", error=str(e))
                        self._bus.publish(EventType.ERROR, stage="stt", message=str(e))
                        continue  # stay in the loop, listen again

                    text = result.text
                    if not text:
                        self._log.info("follow_up_empty_transcript")
                        continue  # stay in the loop, listen again

                    # 2026-06-20: reject very low-confidence
                    # barge-in transcripts. TTS bleed can
                    # trigger the follow-up listener and
                    # Whisper hallucinates short garbage
                    # ("not", "Slap") with avg_logprob around
                    # -1.4. Discard those without bothering
                    # the LLM.
                    follow_up_min_conf = self._cfg.orchestrator.follow_up_min_confidence
                    if result.avg_logprob < follow_up_min_conf:
                        self._log.warning(
                            "follow_up_low_confidence_discarded",
                            text=text[:80],
                            avg_logprob=round(result.avg_logprob, 3),
                            threshold=follow_up_min_conf,
                        )
                        continue  # stay in the loop, listen again

                    self._log.info("transcript", text=text[:200])

                    # Sleep-phrase check: short-circuit the LLM.
                    text_lower = text.strip().lower()
                    if any(p in text_lower for p in sleep_phrases):
                        self._log.info("follow_up_sleep_phrase", text=text[:60])
                        try:
                            self._audio_input.set_aec_active(True)
                            await self._tts.speak("Very good, sir.")
                            await self._tts.wait_until_queue_drained()
                            await self._audio_output.wait_until_drained()
                        except TTSError:
                            pass
                        finally:
                            self._audio_input.set_aec_active(False)
                            await self._post_speak_settle()
                        return

                    # Add to history and run the turn.
                    if self._cfg.conversation.enabled:
                        self._history.add("user", text)
                    await self._think_state(text)

                    if self._shutdown:
                        return
            except asyncio.CancelledError:
                raise
        finally:
            self._in_follow_up = False
            self._bus.publish(EventType.IDLE)

    async def _transcribe_state(self, utterance: np.ndarray) -> None:
        """TRANSCRIBE: Send audio to faster-whisper and get text."""
        self._bus.publish(EventType.TRANSCRIBING)

        # Reject utterances too short to be real speech (noise threshold)
        utterance_s = len(utterance) / 16000
        if utterance_s < 0.4:
            self._log.info("utterance_too_short", duration_s=f"{utterance_s:.2f}")
            return

        # Gentle normalization — don't amplify near-silence
        peak = np.max(np.abs(utterance))
        if peak > 0.05:
            scale = 0.9 / peak
            utterance = utterance * min(scale, 1.5)

        stt_start = time.perf_counter()
        try:
            result = await self._stt.transcribe(utterance)
        except STTError as e:
            self._log.error("stt_failed", error=str(e))
            self._bus.publish(EventType.ERROR, stage="stt", message=str(e))
            return  # return to IDLE
        self._timer.record(STAGE_STT, time.perf_counter() - stt_start)

        text = result.text
        if not text:
            self._log.info("empty_transcript")
            return

        # 2026-06-20: hard-coded "go to sleep" / "jarvis stop"
        # safety hatch. If the STT transcript matches a
        # strong control-flow phrase, bypass the LLM and
        # enter standby directly. Checked BEFORE confidence
        # filtering so soft "sleep" commands are never ignored.
        safety_phrases = [
            "sleep",
            "sleep now",
            "sleep void",
            "void sleep",
            "sleep jarvis",
            "jarvis sleep",
            "go to sleep",
            "stop listening",
            "stop void",
            "void stop",
            "jarvis stop",
            "standby",
            "enter standby",
            "shut up",
            "be quiet",
            "goodbye",
            "bye",
            "that's all",
            "that is all",
            "that'll be all",
            "that will be all",
            "that's enough",
            "that is enough",
        ]
        text_lower = text.strip().lower()
        for phrase in safety_phrases:
            if phrase in text_lower:
                self._log.info(
                    "safety_phrase_triggered",
                    phrase=phrase,
                    text=text[:80],
                    result_logprob=round(result.avg_logprob, 3),
                )
                # Add to history so the user can see what was said in this turn
                if self._cfg.conversation.enabled:
                    self._history.add("user", text)
                await self._dispatch_safety_phrase(phrase)
                return

        # Reject low-confidence transcripts on the main wake-word path.
        # Deepgram Nova-3 returns logprob derived from confidence.
        # Discard background noise without bothering the LLM.
        main_path_min_conf = self._cfg.orchestrator.follow_up_min_confidence
        if result.avg_logprob < main_path_min_conf:
            self._log.warning(
                "main_path_low_confidence_discarded",
                text=text[:80],
                avg_logprob=round(result.avg_logprob, 3),
                threshold=main_path_min_conf,
            )
            return  # back to IDLE — don't feed noise to the LLM

        self._log.info("transcript", text=text[:200])

        # Add to conversation history (Phase 1 — in-session context)
        if self._cfg.conversation.enabled:
            self._history.add("user", text)

        await self._think_state(text)

    async def _dispatch_safety_phrase(self, phrase: str) -> None:
        """Handle an instant safety phrase by bypassing the LLM and putting VOID to sleep."""
        try:
            self._audio_input.set_aec_active(True)
            await self._tts.speak("Entering standby mode, sir.")
            await self._tts.wait_until_queue_drained()
            await self._audio_output.wait_until_drained()
        except TTSError:
            pass
        finally:
            self._audio_input.set_aec_active(False)
            await self._post_speak_settle()
        self._skip_follow_up = True
        self._in_follow_up = False
        self._wake.reset()
        self._audio_input.clear_buffer()
        self._bus.publish(EventType.IDLE)
        self._log.info("safety_phrase_dispatched", phrase=phrase)

    async def _think_state(self, text: str) -> None:
        """THINK: Route to brain, stream tokens, handle tool calls, start speaking.

        Phase 2 tool execution loop:
          1. Stream from Brain
          2. If ToolCallRequest received → execute tool → feed result back
          3. Repeat up to max_calls_per_turn times
          4. When Brain returns text-only → stream to TTS
        """
        # H30: each state starts with a fresh error budget.
        self._reset_consecutive_errors()
        self._bus.publish(EventType.THINKING)
        self._timer.mark("think_start")

        # Load system prompt
        system_prompt = await self._load_system_prompt("voice_mode")
        
        current_time = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        system_prompt = f"Current system time: {current_time}\n\n" + (system_prompt or "")

        if self._guest_mode:
            system_prompt = (system_prompt or "") + (
                "\n\nCRITICAL SECURITY INSTRUCTION: You are in GUEST MODE because the voice biometric "
                "did not match the owner. In Guest Mode, you MUST NOT execute any tools or take any actions. "
                "You may only answer general knowledge questions. If asked to do anything else, politely decline "
                "stating that you are in Guest Mode."
            )

        # Inject workstyle profile if it exists.
        # Guard: cap at 2000 chars so we never push the system prompt past
        # the num_ctx=4096 token budget and silently truncate history.
        #
        # M20: the workstyle file used to be re-read on every
        # turn (every ``_think_state`` call). The file changes at
        # most once per week (the observer writes it on Sundays),
        # so 99% of the reads are wasted I/O. We now cache the
        # content + the file's mtime. The cache is invalidated
        # when the mtime changes (observer rewrote the file).
        workstyle_path = self._cfg.resolve(self._cfg.paths.workstyle)
        workstyle_content = ""
        if workstyle_path.exists():
            try:
                mtime = workstyle_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            cached = self._workstyle_cache
            if cached is None or cached.get("mtime") != mtime or cached.get("path") != str(workstyle_path):
                # Cache miss (first call, file changed, or path
                # changed). Read the file.
                try:
                    raw = workstyle_path.read_text(encoding="utf-8").strip()
                except OSError as e:
                    self._log.warning("workstyle_read_failed", path=str(workstyle_path), error=str(e))
                    raw = ""
                self._workstyle_cache = {
                    "path": str(workstyle_path),
                    "mtime": mtime,
                    "content": raw,
                }
                cached = self._workstyle_cache
            workstyle_content = cached.get("content", "")
        if workstyle_content:
            # H7: capture the ORIGINAL length BEFORE any truncation
            # so the log statement reports what we actually read,
            # not the post-truncation length.
            original_chars = len(workstyle_content)
            MAX_WORKSTYLE_CHARS = 2000
            if original_chars > MAX_WORKSTYLE_CHARS:
                workstyle_content = workstyle_content[:MAX_WORKSTYLE_CHARS] + "\n...[truncated]"
                self._log.warning(
                    "workstyle_truncated",
                    original_chars=original_chars,
                    cap=MAX_WORKSTYLE_CHARS,
                )
            system_prompt = (system_prompt or "") + f"\n\n{workstyle_content}"
                
        # Inject recent short-term memories (Second Brain)
        # Inject recent memories (Second Brain Stage 2 & Stage 3)
        force_local_brain = False
        try:
            if self._vector_engine and self._knowledge_graph:
                # 1. Semantic Search (LanceDB) with relevance cutoff
                vector_results = await self._vector_engine.search(text, limit=3, include_local_only=True, max_distance=1.0)
                
                # 2. Graph Spreading Activation (Match words and n-grams against known graph nodes)
                raw_tokens = [w.strip(".,!?:;\"'()[]{}").lower() for w in text.split()]
                stop_words = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "have", "with", "from", "your", "does", "doing", "tell", "about", "please", "could", "would", "the", "and", "for"}
                clean_tokens = [w for w in raw_tokens if w and w not in stop_words]
                
                # Direct check against known graph nodes (allows 'cat', 'bob', 'rust', 'sarah')
                start_nodes = [t for t in clean_tokens if self._knowledge_graph._graph.has_node(t)]
                for i in range(len(clean_tokens) - 1):
                    bigram = f"{clean_tokens[i]} {clean_tokens[i+1]}"
                    if self._knowledge_graph._graph.has_node(bigram):
                        start_nodes.append(bigram)
                if not start_nodes and clean_tokens:
                    start_nodes = clean_tokens[:4]
                    
                graph_results = await self._knowledge_graph.spread_activation(start_nodes, max_hops=2, threshold=0.3)
                
                # Format for prompt with context budgeting
                memory_lines = []
                for res in vector_results[:3]:
                    if res.get("is_local_only"):
                        force_local_brain = True
                    # Truncate text to 200 chars to avoid prompt bloat and TTFT latency
                    snippet = res['text'][:200].strip()
                    memory_lines.append(f"- (Memory) {snippet}")
                    
                for res in graph_results[:4]:
                    node = res["node"]
                    conns = ", ".join(res["connections"][:3])
                    memory_lines.append(f"- (Graph) {node} ({conns})")
                    
                if memory_lines:
                    memory_text = "\n".join(memory_lines)
                    system_prompt = (system_prompt or "") + f"\n\n[Second Brain Memories]:\n{memory_text}"
                    
                    if force_local_brain:
                        self._log.info("privacy_guardrail_triggered", reason="local_only_memory_retrieved")
        except Exception as e:
            self._log.warning("failed_to_inject_second_brain", error=str(e))
            
        # (Periodic compaction replaced by WAL)
                
        # Inject interruption context if previous turn was barge-in interrupted
        if self._last_interrupted_fragment:
            system_prompt = (system_prompt or "") + (
                f"\n\nNote: you were interrupted mid-response. "
                f"You had just said: \"{self._last_interrupted_fragment}\". "
                f"Acknowledge the interruption naturally if relevant."
            )
            self._last_interrupted_fragment = None

        # Build conversation messages for /api/chat (includes history + current turn)
        messages: list[dict[str, str]] = []
        if self._cfg.conversation.enabled and not self._history.is_empty:
            messages = list(self._history.as_messages())
        else:
            messages = [{"role": "user", "content": text}]

        try:
            # Phase 2: Tool execution loop
            final_text_tokens: list[str] = []
            tool_calls_this_turn = 0
            executed_tools_this_turn: list[str] = []
            configured_max = getattr(self._cfg.tools, "max_calls_per_turn", 0)
            max_calls = float("inf") if configured_max <= 0 else configured_max

            current_text = text
            current_system_prompt = system_prompt

            while True:
                from core.brain_router import BrainType
                route_kwargs = {
                    "text": current_text,
                    "messages": messages,
                    "system_prompt": current_system_prompt,
                    "tool_declarations": self._gemini_tool_declarations,
                    "tool_schema_text": self._ollama_tool_schema,
                }
                
                if force_local_brain:
                    route_kwargs["force_brain"] = BrainType.QWEN

                token_stream = self._brain_router.route_and_generate(**route_kwargs)

                tool_call_detected = False
                first_token_marked = False

                async for item in token_stream:
                    if not first_token_marked:
                        self._timer.mark("first_token")
                        self._timer.record(STAGE_FIRST_TOKEN, time.perf_counter() - self._timer._current.marks["think_start"])
                        first_token_marked = True

                    if isinstance(item, ToolCallRequest):
                        tool_call_detected = True
                        tool_calls_this_turn += 1
                        # Clear any speculative pre-announcements or thinking text emitted before the tool call
                        final_text_tokens.clear()

                        if tool_calls_this_turn > max_calls:
                            self._log.warning(
                                "tool_call_limit_reached",
                                max="unlimited" if max_calls == float("inf") else max_calls,
                                tool=item.name,
                            )
                            final_text_tokens.append(
                                f"I've reached my limit of {max_calls} tool calls for this request. "
                                "Let me know if you need more."
                            )
                            break

                        # H3: record the assistant's own tool call into the
                        # conversation history so the next re-route sees
                        # "I called X" before "the result of X is Y". Without
                        # this, the local Qwen re-route was fed only the
                        # result string and lost the context of its own
                        # function call. The reconstruction matches the
                        # exact JSON the model emitted (so the schema is
                        # identical to what the model expects to see in
                        # its own chat history).
                        import json as _json
                        try:
                            assistant_call_text = _json.dumps({
                                "tool_call": {
                                    "name": item.name,
                                    "arguments": item.arguments,
                                }
                            }, separators=(",", ":"))
                        except (TypeError, ValueError):
                            # Best-effort fallback if arguments contain
                            # something not JSON-serialisable (shouldn't
                            # happen, the schema enforces primitives).
                            assistant_call_text = (
                                f'{{"tool_call": {{"name": "{item.name}", '
                                f'"arguments": {{}}}}}}'
                            )
                        messages.append({
                            "role": "assistant",
                            "content": assistant_call_text,
                        })

                        # Execute the tool
                        result = await self._execute_tool_call(item)
                        executed_tools_this_turn.append(
                            f"{item.name}({item.arguments}) -> {'Success' if result.success else 'Failed'}"
                        )

                        # Feed result back to brain for next response.
                        # M24: the tool's ``metadata`` dict (paths,
                        # durations, PIDs, etc.) used to be dropped
                        # here, so the LLM only saw the spoken
                        # ``data`` / ``error`` field. We now append a
                        # compact ``metadata={...}`` line to the
                        # result text when metadata is non-empty.
                        # The control-flow ``signal`` field is
                        # intentionally NOT included here — the
                        # orchestrator reads it from ``result.signal``
                        # above (H5).
                        meta_text = ""
                        if result.metadata:
                            # Cap each value at 100 chars so a
                            # misbehaving tool that stuffs a 10 KB
                            # blob into metadata doesn't blow up
                            # the context.
                            safe_meta = {
                                k: (str(v)[:100] + "..." if len(str(v)) > 100 else v)
                                for k, v in result.metadata.items()
                            }
                            meta_text = f" metadata={safe_meta}"
                        tool_result_text = (
                            f"Tool '{item.name}' result: "
                            f"{'Success' if result.success else 'Failed'}. "
                            f"{result.data if result.data else result.error}"
                            f"{meta_text}\n"
                            f"[System Instruction: State the final confirmation or direct answer in ONE concise sentence. "
                            f"Do NOT explain how the tool works, do NOT narrate the execution steps, and do NOT over-explain.]"
                        )

                        # H3 (continued): the tool result is a user-role
                        # message in the model's view, so it slots in
                        # right after the assistant's tool call. Putting
                        # it in this order preserves the
                        # "assistant called → user returned" structure
                        # that prompt-based tool calling depends on.
                        messages.append({
                            "role": "user",
                            "content": tool_result_text,
                        })

                        current_text = tool_result_text

                        # Load tool chaining prompt for multi-step tasks
                        tool_chain_prompt = await self._load_system_prompt("tool_chaining")
                        if tool_chain_prompt and tool_chain_prompt not in (current_system_prompt or ""):
                            current_system_prompt = f"{current_system_prompt or ''}\n\n{tool_chain_prompt}"

                        # H4: tool failure path. The previous implementation
                        # made an unconditional second LLM call to
                        # "evaluate the error" and broke out of the loop
                        # without producing any text. If that call also
                        # failed, the user heard nothing. We now wrap
                        # the call in try/except and always produce a
                        # ``final_text_tokens`` entry on tool failure.
                        if not result.success:
                            current_system_prompt = system_prompt
                            tool_chain_prompt = await self._load_system_prompt("tool_chaining")
                            if tool_chain_prompt and tool_chain_prompt not in (current_system_prompt or ""):
                                current_system_prompt = f"{current_system_prompt or ''}\n\n{tool_chain_prompt}"

                            produced_response = False
                            try:
                                eval_text = (
                                    f"User intervened with a new instruction. Carry out the user's latest instruction immediately: {result.error}"
                                    if "User intervened with new instruction" in (result.error or "")
                                    else f"Tool failed. Evaluate what went wrong and ask for clarification based on this result: {tool_result_text}"
                                )
                                async for token in self._brain_router.route_and_generate(
                                    text=eval_text,
                                    messages=messages,
                                    system_prompt=current_system_prompt,
                                    tool_declarations=self._gemini_tool_declarations,
                                    tool_schema_text=self._ollama_tool_schema,
                                ):
                                    if isinstance(token, str):
                                        final_text_tokens.append(token)
                                        produced_response = True
                            except Exception as eval_err:
                                self._log.error(
                                    "tool_failure_evaluation_failed",
                                    tool=item.name,
                                    error=str(eval_err),
                                )

                            if not produced_response:
                                # Canned fallback (H4). The user always
                                # hears something on a tool failure —
                                # silence is the bug we are fixing.
                                fallback = (
                                    f"That tool didn't work: {result.error or 'unknown error'}. "
                                    "Would you like me to try a different approach?"
                                )
                                final_text_tokens.append(fallback)
                                self._log.info(
                                    "tool_failure_evaluation_fallback_used",
                                    tool=item.name,
                                    error=result.error or 'unknown error',
                                )
                            break

                        # H5: control flow is read from the typed
                        # ``result.signal`` enum, NOT from
                        # ``result.metadata.get("action")``. The metadata
                        # dict is now pure data (e.g. ``{"pid": ...}`` for
                        # the reboot handover) and can never accidentally
                        # trigger shutdown/sleep.
                        if result.signal == ToolSignal.SHUTDOWN:
                            self._log.info("shutdown_signal_received", tool=item.name)
                            self.shutdown()
                            final_text_tokens.append(" Shutting down.")
                            tool_call_detected = False
                            break

                        if result.signal == ToolSignal.SLEEP:
                            self._log.info("sleep_signal_received", tool=item.name)
                            # Suppress follow-up listening; return directly to IDLE
                            self._skip_follow_up = True
                            self._in_follow_up = False
                            self._audio_input.clear_buffer()
                            self._bus.publish(EventType.IDLE)
                            tool_call_detected = False
                            break

                        if result.signal == ToolSignal.REBOOT:
                            # H5: reboot handover flow. The handover
                            # payload is already in ``result.metadata``.
                            # Set flags so the finally block in ``run()``
                            # spawns the child process after cleanup.
                            self._log.info("reboot_signal_received", tool=item.name)
                            self._shutdown = True
                            self._reboot_pending = True
                            self._reboot_metadata = dict(result.metadata or {})
                            tool_call_detected = False
                            break

                        # Break the async for loop so the while loop can re-route with new stream!
                        break
                    else:
                        # Regular text token
                        final_text_tokens.append(item)

                if not tool_call_detected or tool_calls_this_turn > max_calls:
                    break  # Finished all tool calls or reached limit

            # Stream the accumulated text tokens to TTS
            if final_text_tokens:
                splitter = SentenceSplitter(self._cfg.streaming)

                async def _token_iter() -> AsyncIterator[str]:
                    for t in final_text_tokens:
                        yield t

                chunk_stream = splitter.process(_token_iter())
                full_raw_text = " ".join(final_text_tokens).strip()
                await self._speak_state(chunk_stream, executed_tools=executed_tools_this_turn, fallback_text=full_raw_text)
            else:
                # Brain returned no speakable text. This happens when:
                # 1. A tool call returned a signal (sleep/reboot) with no spoken words.
                # 2. The model responded with only JSON / internal thoughts.
                # 3. A casual phrase was misrouted (e.g. "just wait" matched a tool).
                # Always give the user audio feedback so they know JARVIS heard them.
                self._log.info("empty_response_from_brain")
                fallback_stream = self._make_canned_stream("I heard you, sir. How can I help?")
                await self._speak_state(fallback_stream, executed_tools=executed_tools_this_turn, fallback_text="I heard you, sir. How can I help?")


        except GPUOutOfMemoryError:
            # GPU OOM — hand off to the SmolLM failsafe via BrainRouter
            # so the user keeps the same conversation history, system
            # prompt, and event-bus notifications. SmolLM is too small
            # to call tools reliably, so tool support is bypassed for
            # this turn.
            self._log.warning("gpu_oom_fallback_to_smollm")
            self._bus.publish(EventType.BRAIN_CHANGE, brain="smollm_failsafe")
            try:
                token_stream = self._brain_router.route_and_generate(
                    text=text,
                    messages=messages,
                    system_prompt=system_prompt,
                    tool_declarations=None,
                    tool_schema_text=None,
                    force_brain=BrainType.SMOLLM,
                    bypass_tools=True,
                )
                splitter = SentenceSplitter(self._cfg.streaming)
                chunk_stream = splitter.process(token_stream)
                await self._speak_state(chunk_stream, executed_tools=executed_tools_this_turn)
            except Exception as e2:
                await self._handle_error(e2)
        except LLMError as e:
            await self._handle_error(e)

    async def _execute_tool_call(self, tool_call: ToolCallRequest) -> Any:
        """Execute a tool call and return the result.

        Speaks a brief acknowledgment before execution.
        """
        # 2026-06-20: ToolResult / ToolSignal are now imported
        # at module top (H5 follow-up). The previous local
        # import only made the names visible inside this
        # method, which meant ``_think_state`` crashed with
        # ``NameError: ToolSignal not defined`` whenever a
        # tool with a control-flow signal (reboot / sleep /
        # shutdown) ran. Now both call sites can read
        # ``result.signal`` without scoping surprises.
        self._log.info(
            "tool_call_received",
            tool=tool_call.name,
            args=str(tool_call.arguments)[:100],
        )

        if self._tool_executor is None:
            self._log.warning("tool_executor_not_available")
            return ToolResult(
                success=False,
                error="Tool execution is not available.",
            )

        # Determine if confirmation is required
        if self._tool_registry:
            tool_spec = self._tool_registry.get(tool_call.name)
            is_verified = tool_spec and tool_spec.verify
        else:
            is_verified = False

        # M23: if the tool is verify=True and the system is
        # currently LOCKED, refuse the call BEFORE speaking the
        # ack and BEFORE opening the mic. The previous flow
        # would ask the user to "Please confirm execution of
        # write_file", listen for a "yes", and THEN
        # ToolExecutor would block the call with "verification
        # required" — the user heard the prompt, said yes, and
        # got told the action is still locked. The new flow
        # checks the auth state once, up front, and returns
        # immediately if locked. Saves 2-3 s of TTS + VAD
        # round-trip and gives the user a clear "unlock first"
        # message instead of a confusing confirm-then-block.
        if is_verified and self._cfg.verification.enabled:
            from security.verification import verification_manager, AuthState
            state = await verification_manager.get_state()
            if state == AuthState.LOCKED:
                self._log.warning(
                    "tool_blocked_pre_voice_confirm",
                    tool=tool_call.name,
                )
                self._bus.publish(
                    EventType.TOOL_DONE,
                    tool_name=tool_call.name,
                    success=False,
                    duration_ms=0.0,
                )
                return ToolResult(
                    success=False,
                    error=(
                        f"'{tool_call.name}' requires security verification. "
                        "Say 'unlock jarvis' first to enable destructive tools."
                    ),
                )

        if is_verified:
            ack = f"Sir, please confirm execution of {tool_call.name}."
        else:
            # Speak brief acknowledgment
            acknowledgments = {
                "read_file": "Let me read that file.",
                "list_dir": "Let me check that directory.",
                "web_search": "Let me search for that.",
                "get_system_info": "Checking system info.",
                "get_cpu_usage": "Checking CPU usage.",
                "get_ram_usage": "Checking memory usage.",
                "get_disk_usage": "Checking disk space.",
                "save_note": "Saving that note.",
                "get_note": "Retrieving that note.",
                "list_notes": "Let me check your notes.",
                "run_python": "Running that code.",
                "run_shell": "Running that command.",
                "read_clipboard": "Reading clipboard.",
                "write_clipboard": "Copying to clipboard.",
                "launch_app": "Opening that app.",
                "open_url": "Opening that link.",
            }
            ack = acknowledgments.get(tool_call.name, f"Running {tool_call.name}.")

        try:
            self._audio_input.set_aec_active(True)
            await self._tts.speak(ack)
            # Wait for acknowledgment to finish playing
            await self._tts.wait_until_queue_drained()
            await self._audio_output.wait_until_drained()
        except TTSError:
            pass  # Don't let ack failure block tool execution
        finally:
            self._audio_input.set_aec_active(False)
            await self._post_speak_settle()

        # Voice Confirmation Protocol Execution
        if is_verified:
            self._log.info("awaiting_voice_confirmation", tool=tool_call.name)
            self._bus.publish(EventType.LISTENING)
            
            adaptive_threshold = self._cfg.vad.threshold
            self._vad.set_threshold(adaptive_threshold)
            
            utterance = await self._vad.collect_utterance(
                lambda: self._audio_input.read_frame(),
                max_duration_s=self._cfg.vad.max_utterance_ms / 1000.0,
                max_consecutive_none=20,
            )
            
            self._vad.reset_threshold()
            
            if utterance is None:
                return ToolResult(success=False, error="Confirmation timed out or aborted.")
                
            self._bus.publish(EventType.TRANSCRIBING)
            try:
                peak = np.max(np.abs(utterance))
                if peak > 0.05:
                    scale = 0.9 / peak
                    utterance = utterance * min(scale, 1.5)
                result = await self._stt.transcribe(utterance)
                text = result.text
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to transcribe confirmation: {e}")
                
            if not text:
                return ToolResult(success=False, error="No voice detected. Tool execution cancelled.")
                
            text_lower = text.lower().strip()
            self._log.info("voice_confirmation_received", text=text_lower)
            
            affirmative = [
                "yes", "yeah", "yep", "do it", "proceed", "go ahead", "affirmative",
                "ok", "okay", "sure", "fine", "confirm", "confirmed", "i confirm",
                "please do", "execute", "run it", "go for it", "absolutely", "definitely",
            ]
            denial = [
                "no", "nope", "cancel", "stop", "don't", "dont", "abort", "negative",
                "never mind", "nevermind", "dismiss",
            ]
            
            is_no = any(word in text_lower for word in denial)
            if "no problem" in text_lower or "no worries" in text_lower:
                is_no = False
            is_yes = any(word in text_lower for word in affirmative)

            # Always record the user's speech during confirmation to conversation history
            if self._cfg.conversation.enabled:
                self._history.add("user", text)

            if is_no or not is_yes:
                words = text_lower.split()
                # If the user gave a full command or new instruction instead of a pure denial
                is_pure_denial = is_no and (len(words) <= 2 or any(text_lower == d for d in denial))
                
                try:
                    self._audio_input.set_aec_active(True)
                    if is_pure_denial:
                        await self._tts.speak("Cancelled, sir.")
                    else:
                        await self._tts.speak("Understood, sir.")
                    await self._tts.wait_until_queue_drained()
                    await self._audio_output.wait_until_drained()
                except TTSError:
                    pass
                finally:
                    self._audio_input.set_aec_active(False)
                    await self._post_speak_settle()

                if is_pure_denial:
                    return ToolResult(success=False, error="Tool execution cancelled by user.")
                else:
                    return ToolResult(
                        success=False,
                        error=f"User intervened with new instruction: '{text}'. Abort previous action and fulfill this new instruction immediately."
                    )

        # Execute
        result = await self._tool_executor.execute(
            tool_call.name, tool_call.arguments
        )

        try:
            from core.metrics import GLOBAL_METRICS
            status = "success" if result.success else "error"
            GLOBAL_METRICS.increment(
                "jarvis_tool_executions_total",
                1.0,
                labels={"tool": tool_call.name, "status": status},
                doc="Total tool calls executed",
            )
        except Exception:
            pass

        self._log.info(
            "tool_call_completed",
            tool=tool_call.name,
            success=result.success,
        )

        return result

    async def _speak_state(
        self,
        chunk_stream: AsyncIterator[str],
        executed_tools: list[str] | None = None,
        fallback_text: str = "",
    ) -> None:
        """SPEAK: Play sentence chunks with OWW barge-in monitor.

        Phase 1: wake-word barge-in. AEC is active only during playback
        to cancel speaker echo without cancelling user speech in IDLE.
        Response chunks are accumulated and saved to conversation history.
        """
        # H30: each state starts with a fresh error budget.
        self._reset_consecutive_errors()
        self._bus.publish(EventType.SPEAKING)
        self._timer.mark("speak_start")

        # Calculate and log the core latency metric: end of speech to first audio
        if self._timer._current and "listen_end" in self._timer._current.marks:
            latency_s = time.perf_counter() - self._timer._current.marks["listen_end"]
            self._timer.record(STAGE_TOTAL, latency_s)
            if latency_s > 2.0:
                self._log.warning("latency_budget_exceeded", latency_s=round(latency_s, 2), target="<= 2.0s")
            else:
                self._log.info("latency_budget_met", latency_s=round(latency_s, 2))

        # Activate AEC for the duration of TTS playback
        self._audio_input.set_aec_active(True)

        # Accumulate response chunks for conversation history
        response_accumulator: list[str] = []

        barge_in = asyncio.Event()

        async def _play_all() -> None:
            nonlocal response_accumulator
            try:
                async for chunk in chunk_stream:
                    if barge_in.is_set() or self._shutdown:
                        break
                    self._current_sentence = chunk
                    response_accumulator.append(chunk)
                    self._bus.publish(EventType.SPEAKING, text=chunk)
                    await self._tts.speak(chunk)
                
                # Wait for playback to actually finish before marking task done
                if not barge_in.is_set() and not self._shutdown:
                    await self._tts.wait_until_queue_drained()
                    await self._audio_output.wait_until_drained()
            except TTSError as e:
                self._log.error("tts_failed", error=str(e))
                self._bus.publish(EventType.ERROR, stage="tts", message=str(e))
            except asyncio.CancelledError:
                pass

        play_task = asyncio.create_task(_play_all())

        barge_in_detected = await self._barge_in_monitor(play_task, barge_in)

        if barge_in_detected:
            self._log.info("barge_in_detected")
            # 2026-06-20: the barge-in monitor may have
            # ducked TTS volume before this point. Restore
            # it so the next response plays at full volume.
            # The ducking happens during the 320 ms
            # sustained-speech confirmation window — by the
            # time we get here the volume is already low.
            try:
                self._tts.set_volume(1.0)
            except Exception:
                pass
            self._audio_input.set_aec_active(False)
            play_task.cancel()
            await self._tts.flush()
            self._audio_output.flush()
            self._last_interrupted_fragment = self._current_sentence
            # Partial response — still record to history so context is preserved
            if self._cfg.conversation.enabled:
                partial = " ".join(response_accumulator).strip()
                if not partial and fallback_text:
                    partial = fallback_text.strip()
                if partial or executed_tools:
                    tool_meta = {"tools": executed_tools} if executed_tools else None
                    self._history.add("assistant", (partial + " [interrupted]") if partial else "[interrupted]", metadata=tool_meta)
            self._current_sentence = ""
            self._bus.publish(EventType.LISTENING)
            await self._listen_state()
        else:
            await play_task
            # 2026-06-20: restore volume defensively in case
            # the barge-in monitor ducked but exited without
            # a full barge-in (e.g., play_task completed
            # before sustained speech was confirmed).
            try:
                self._tts.set_volume(1.0)
            except Exception:
                pass
            self._audio_input.set_aec_active(False)
            # Record full response to conversation history
            if self._cfg.conversation.enabled:
                full_response = " ".join(response_accumulator).strip()
                if not full_response and fallback_text:
                    full_response = fallback_text.strip()
                if full_response or executed_tools:
                    tool_meta = {"tools": executed_tools} if executed_tools else None
                    self._history.add("assistant", full_response or "[Action completed]", metadata=tool_meta)
                    self._log.debug(
                        "history_updated",
                        turns=self._history.turn_count,
                        response_chars=len(full_response),
                    )

            await self._post_speak_settle()
            # Cooldown gate: suppress wake word for settle_ms + 500ms buffer.
            # This prevents the wake word detector from firing on speaker reverb
            # immediately after TTS finishes. The settle already waited for the
            # buffer to drain; this extra window covers room reverb decay.
            settle_s = getattr(self._cfg.orchestrator, "post_speak_settle_ms", 1200) / 1000.0
            self._post_tts_cooldown_until = time.monotonic() + settle_s + 0.5

            self._timer.mark("speak_end")
            timings = self._timer.finish_turn()
            if timings:
                self._log.info("turn_complete", timings=_fmt_turn_ms(timings))
                self._bus.publish(EventType.LATENCY_REPORT, timings=timings)
                # M13: also call ``log_report`` so the rolling
                # percentile history (``p50/p95/p99``) surfaces
                # in the log. Without this, the ``latency_timer``
                # only ever shows a single turn's stages; the
                # percentile view (which the HUD claims to
                # display) was never populated.
                self._timer.log_report()

            # M7: if the reboot_jarvis tool fired, the farewell
            # line has now been spoken. Drain the post-speak
            # queue, spawn the handover script, and then mark
            # shutdown. The handover script waits 2 s, kills the
            # current process by PID, and launches a fresh
            # instance — so the user hears the full
            # "Rebooting now." line before the SIGTERM arrives.
            if self._reboot_pending:
                await self._perform_reboot_handover()

            # Follow-up window: listen again without wake word (headphones only).
            # Skip if we're already inside a follow-up loop (avoids recursion —
            # the loop iteration is the natural way to continue).
            if (
                self._cfg.orchestrator.follow_up.enabled
                and not self._shutdown
                and not getattr(self, '_skip_follow_up', False)
                and not getattr(self, '_in_follow_up', False)
            ):
                await self._follow_up_state()
            self._skip_follow_up = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _perform_reboot_handover(self) -> None:
        """M7: drain TTS, spawn the reboot handover script, then shutdown.

        Called by ``_speak_state`` after the farewell line has been
        spoken. The handover script is fully detached from this
        process: it waits 2 s, kills the current PID, and launches
        a fresh JARVIS instance. We do NOT shutdown the orchestrator
        here directly — the handover script's SIGTERM is the
        shutdown mechanism, so the user's "Rebooting now." line
        has a chance to drain fully.

        The metadata from the tool result is in
        ``self._reboot_metadata`` and is the single source of
        truth for the PID, restart script, and project root.
        """
        meta = dict(self._reboot_metadata or {})
        self._reboot_pending = False
        self._reboot_metadata = {}

        pid = meta.get("pid")
        restart_script = meta.get("restart_script")
        project_root = meta.get("project_root")

        if not pid or not restart_script or not project_root:
            self._log.error(
                "reboot_handover_missing_metadata",
                have_pid=bool(pid),
                have_script=bool(restart_script),
                have_root=bool(project_root),
            )
            # Best-effort: just shut down. The user has already
            # heard "Rebooting now."; we just won't actually reboot.
            self.shutdown()
            return

        from pathlib import Path
        restart_script_path = Path(restart_script)
        if not restart_script_path.exists():
            self._log.error(
                "reboot_handover_script_missing",
                path=str(restart_script_path),
            )
            self.shutdown()
            return

        # Make sure the TTS queue is fully drained before we
        # hand the process over. The script's 2 s sleep plus the
        # SIGTERM should arrive after the speaker finishes.
        try:
            await self._tts.wait_until_queue_drained(timeout=10.0)
        except Exception as e:
            self._log.warning("reboot_tts_drain_failed", error=str(e))
        try:
            await self._audio_output.wait_until_drained(timeout=10.0)
        except Exception as e:
            self._log.warning("reboot_audio_drain_failed", error=str(e))

        self._log.info("reboot_handover_spawning", pid=pid, script=str(restart_script_path))
        try:
            import subprocess
            import sys
            if sys.platform == "win32":
                subprocess.Popen(
                    [
                        "powershell.exe",
                        "-ExecutionPolicy", "Bypass",
                        "-WindowStyle", "Hidden",
                        "-File", str(restart_script_path),
                        "-JarvisPID", str(pid),
                    ],
                    cwd=str(project_root),
                    creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["bash", str(restart_script_path), str(pid)],
                    cwd=str(project_root),
                    start_new_session=True,
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            self._log.error("reboot_handover_spawn_failed", error=str(e))

        # Mark shutdown so the voice loop exits. The handover
        # script's SIGTERM will arrive ~2 s from now.
        self.shutdown()

    @staticmethod
    async def _make_canned_stream(text: str) -> AsyncIterator[str]:
        """Yield a single hardcoded string as an AsyncIterator[str] for _speak_state."""
        yield text

    async def _play_wake_chime(self) -> None:

        """Play a short two-tone chime indicating JARVIS heard the wake word."""
        sr = self._cfg.audio.sample_rate_out
        t1 = np.linspace(0, 0.08, int(sr * 0.08), endpoint=False, dtype=np.float32)
        t2 = np.linspace(0, 0.08, int(sr * 0.08), endpoint=False, dtype=np.float32)
        tone1 = np.sin(2 * np.pi * 660 * t1, dtype=np.float32) * 0.25
        tone2 = np.sin(2 * np.pi * 880 * t2, dtype=np.float32) * 0.25
        gap = np.zeros(int(sr * 0.04), dtype=np.float32)
        chime = np.concatenate([tone1, gap, tone2])
        await self._audio_output.play(chime)

    async def _load_system_prompt(self, prompt_name: str) -> str | None:
        """Load a system prompt from the prompts directory."""
        prompt_path_str = getattr(self._cfg.prompts, prompt_name, None)
        if prompt_path_str is None:
            return None
        prompt_path = self._cfg.resolve(prompt_path_str)
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self._log.warning("prompt_not_found", path=str(prompt_path))
            return None

    async def _handle_error(self, error: Exception) -> None:
        """Log and publish any error, speak an apology, return to IDLE.

        ``WakeWordError`` is treated as a hard-stop: the wake-word model is
        a startup prerequisite, and the only useful thing we can do is
        surface a clear error and exit. Retrying in a hot loop is worse
        than crashing because it floods the log and burns CPU.
        """
        self._consecutive_errors += 1
        self._log.error("orchestrator_error", error_msg=str(error))
        spoken = "I ran into a problem."

        if isinstance(error, JarvisError):
            spoken = error.spoken_message

        if isinstance(error, WakeWordError):
            # Hard stop — no point looping when the wake-word model is gone.
            self._log.critical(
                "wakeword_unrecoverable_shutting_down",
                error_msg=str(error),
            )
            self._bus.publish(EventType.ERROR, error_msg=str(error), spoken=spoken)
            try:
                self._audio_input.set_aec_active(True)
                await self._tts.speak(spoken)
                await self._tts.wait_until_queue_drained()
                await self._audio_output.wait_until_drained()
            except Exception:
                pass
            finally:
                self._audio_input.set_aec_active(False)
                await self._post_speak_settle()
            self._shutdown = True
            return

        # Publish error event (avoid keyword conflict with structlog)
        self._bus.publish(EventType.ERROR, error_msg=str(error), spoken=spoken)

        # Speak the apology (non-blocking — speak() enqueues internally)
        try:
            self._audio_input.set_aec_active(True)
            await self._tts.speak(spoken)
            await self._tts.wait_until_queue_drained()
            await self._audio_output.wait_until_drained()
        except Exception:
            pass
        finally:
            self._audio_input.set_aec_active(False)
            await self._post_speak_settle()

        # Backoff to prevent error spam loop
        backoff = min(0.5 * (1 << self._consecutive_errors), 10.0)  # 1s, 2s, 4s, 8s, 10s max
        if self._consecutive_errors > 1:
            self._log.info("error_backoff", seconds=round(backoff, 1))
            await asyncio.sleep(backoff)

        self._bus.publish(EventType.IDLE)

    async def _stats_loop(self) -> None:
        """Background task to publish system stats periodically."""
        try:
            import psutil
        except ImportError:
            self._log.warning("psutil_not_installed_stats_disabled")
            return

        while not self._shutdown:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                ram_gb = round(ram.used / (1024 ** 3), 1)

                try:
                    from core.metrics import GLOBAL_METRICS
                    GLOBAL_METRICS.set_gauge("jarvis_system_cpu_percent", cpu, doc="System CPU utilization percent")
                    GLOBAL_METRICS.set_gauge("jarvis_system_ram_used_gb", ram_gb, doc="System RAM used in GB")
                except Exception:
                    pass

                self._bus.publish(EventType.SYSTEM_STATS, cpu=cpu, ram=ram_gb)
            except Exception as e:
                self._log.error("stats_loop_error", error=str(e))
                
            await asyncio.sleep(2.0)


def _fmt_turn_ms(timings: dict[str, float]) -> str:
    """Format turn timings for logging (e.g. 'stt=287ms first_token=394ms')."""
    parts = []
    for key, val in timings.items():
        parts.append(f"{key}={val * 1000:.0f}ms")
    return " ".join(parts)
