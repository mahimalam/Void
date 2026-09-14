"""J.A.R.V.I.S. Phase 2 — Voice loop + Tool system entry point.

Usage:
    python -m core.main

Or from the project root:
    .\\run.ps1
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

# Ensure pip-installed CUDA libraries are on PATH (e.g. nvidia-cublas-cu12)
_venv = Path(__file__).resolve().parent.parent
_site = _venv / ".venv" / "Lib" / "site-packages"
for _pkg in ("nvidia.cublas.bin", "nvidia.cuda_runtime.bin", "nvidia.cudnn.bin"):
    _dll_dir = _site / _pkg.replace(".", os.sep)
    if _dll_dir.is_dir():
        os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")

from core.config import load_config
from core.event_bus import EventBus, console_subscriber
from core.latency import LatencyTimer
from core.logging_setup import configure_logging, get_logger


async def main(config_path: str = "config.yaml") -> None:
    """Initialize all components and run the voice loop."""

    # --- 1. Load config ---
    # M31: the previous code used ``Path.cwd()`` as the project
    # root. That breaks when the user invokes JARVIS from a
    # different directory (e.g. ``cd /tmp && python -m
    # core.main``) — every config-relative path resolves under
    # the wrong root, the database lands in /tmp, and the
    # setup.ps1 launcher (which already runs from the project
    # root) is the only thing that ever "works". We now derive
    # the project root from ``__file__`` so it always points
    # to the actual JARVIS installation, regardless of CWD.
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(config_path, project_root=project_root)

    # --- 2. Setup logging ---
    configure_logging(cfg.logging, project_root)
    log = get_logger("main")
    log.info("jarvis_starting", version="0.2.0-phase2", cwd=str(project_root))

    # --- 3. Create event bus and UI server ---
    from core.event_bus import start_websocket_server
    bus = EventBus()
    # Expose bus globally so security/verification.py can emit AUTH_CHANGE
    import core.event_bus as _eb_module
    _eb_module._global_bus = bus
    console_task = asyncio.create_task(console_subscriber(bus))
    # M30: hold the returned handle so the cleanup block in this
    # function can call ``ws_handle.stop()`` to gracefully close
    # the server on shutdown (instead of leaving the socket and
    # the broadcast task alive).
    try:
        ws_handle = await start_websocket_server(bus, port=9001)
    except OSError as e:
        if e.errno == 10048 or "10048" in str(e):  # WSAEADDRINUSE
            log.error(
                "port_9001_already_in_use",
                hint="A previous JARVIS instance is still running. "
                     "Run .\\run.ps1 (it now kills stale processes automatically) "
                     "or manually kill the old python process and retry.",
            )
            print(
                "\n[ERROR] Port 9001 is already in use.\n"
                "A previous JARVIS instance is still running.\n"
                "Fix: close the old terminal window, or run .\\run.ps1 "
                "(it now kills stale processes automatically).\n"
            )
        raise

    # Start Metrics / Observability HTTP server if enabled
    metrics_server = None
    if getattr(cfg, "observability", None) and cfg.observability.enabled:
        from core.metrics import GLOBAL_METRICS, GLOBAL_WATCHDOG, MetricsServer
        metrics_server = MetricsServer(
            GLOBAL_METRICS,
            GLOBAL_WATCHDOG,
            host=cfg.observability.host,
            port=cfg.observability.port,
        )
        try:
            await metrics_server.start()
        except Exception as e:
            log.warning("metrics_server_startup_failed", error=str(e))

    # Launch Electron HUD
    import subprocess
    import sys
    hud_dir = project_root / "ui" / "hud"
    hud_proc = None
    if hud_dir.exists():
        # 2026-06-20: only launch the HUD if one isn't already
        # running. Before this check, every JARVIS restart
        # spawned a fresh Electron window (no single-instance
        # lock in main.js), stacking 4+ HUDs on screen. The
        # Electron side now also enforces a single instance via
        # ``app.requestSingleInstanceLock`` in main.js, but we
        # skip the npm spawn here too as a belt-and-suspenders
        # so we don't briefly flash a second window during the
        # ~250 ms it takes for the lock check to fire.
        try:
            import psutil
            found_already = False
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    if not cmdline:
                        continue
                    cmd_str = " ".join(cmdline).lower()
                    # Only match if it's running the JARVIS HUD directory
                    if "ui/hud" in cmd_str:
                        if any(term in cmdline[0].lower() for term in ("electron", "node", "npm")):
                            found_already = True
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            if found_already:
                log.info("electron_hud_already_running")
                hud_proc = None
            else:
                hud_log_path = cfg.resolve(cfg.paths.data_dir) / "hud.log"
                hud_log_file = open(hud_log_path, "a", encoding="utf-8")
                hud_env = {**os.environ, "ELECTRON_DISABLE_SANDBOX": "1"}
                hud_proc = subprocess.Popen(
                    ["npm", "start"],
                    cwd=str(hud_dir),
                    stdout=hud_log_file,
                    stderr=subprocess.STDOUT,
                    env=hud_env
                )
                log.info("electron_hud_launched", pid=hud_proc.pid)
        except Exception as e:
            log.warning("electron_hud_launch_failed", error=str(e),
                        hint="Check data/hud.log or run 'npm start' in ui/hud")

    # --- 3.5 Configure & Initialize Security ---
    data_dir = cfg.resolve(cfg.paths.data_dir)

    # Configure absolute paths for security modules (avoids CWD-relative path bugs)
    from security import biometrics as _biometrics_mod
    _biometrics_mod.configure(data_dir)

    # C8 fix: write a lock file so the standalone enroll.py CLI refuses
    # to run while JARVIS is using the same voiceprint file.
    _biometrics_mod.write_lock_file()

    from security.verification import verification_manager
    verification_manager.configure(data_dir)
    await verification_manager.start()

    # --- 4. Create latency timer ---
    timer = LatencyTimer(bus=bus)

    # --- 5. Initialize audio components ---
    from core.audio import AudioInput, AudioOutput

    # Create AEC echo canceller if enabled (Phase 1: DirectBufferAEC)
    echo_canceller = None
    if cfg.aec.enabled:
        from core.aec import DirectBufferAEC
        echo_canceller = DirectBufferAEC(
            filter_length=cfg.aec.filter_length,
            delay_ms=cfg.aec.delay_ms,
            mic_sample_rate=cfg.aec.mic_sample_rate,
            tts_sample_rate=cfg.aec.tts_sample_rate,
        )
        log.info(
            "aec_enabled",
            method="direct_buffer",
            delay_ms=cfg.aec.delay_ms,
            filter_length=cfg.aec.filter_length,
        )

    audio_input = AudioInput(cfg.audio, bus=bus, aec=echo_canceller)
    audio_output = AudioOutput(
        cfg.audio,
        bus=bus,
        on_play=echo_canceller.feed_reference if echo_canceller else None,
    )

    from core.wake_word import WakeWordDetector
    wake = WakeWordDetector(cfg.wake_word)

    from core.vad import VADProcessor
    vad = VADProcessor(cfg.vad)
    await vad.load_model()

    from core.stt import STTEngine
    stt = STTEngine(cfg.stt)
    await stt.load_model()

    from core.llm import LLMClient
    llm = LLMClient(cfg.ollama)

    # Cloud Brain
    gemini_client = None
    if cfg.brains.brain2_specialist.enabled:
        from core.openai_client import OpenAIClient
        gemini_client = OpenAIClient(cfg.brains.brain2_specialist)
        gemini_client.preferred_model_type = cfg.ollama.compaction_brain
        log.info(
            "openai_client_created",
            model=cfg.brains.brain2_specialist.model_flash,
            compaction_brain=cfg.ollama.compaction_brain,
        )
        try:
            gemini_client.initialize()
        except Exception as e:
            log.warning("openai_client_eager_init_failed", error=str(e))

    # BrainRouter — wraps LLM + Gemini, handles classification + routing
    from core.brain_router import BrainRouter
    brain_router = BrainRouter(
        cfg=cfg,
        llm=llm,
        bus=bus,
        gemini_client=gemini_client,
    )

    from core.tts import TTSManager
    tts = TTSManager(cfg.tts, bus=bus)
    if cfg.tts.deepgram_api_key:
        log.info(
            "deepgram_tts_active",
            note="Aura-2: andromeda-en (Jarvis) / amalthea-en (Hey Jarvis)",
        )
    else:
        log.info("tts_using_piper_local", note="No Deepgram key — using Piper.")

    # --- 6. Initialize Phase 2: Tool System ---
    tool_registry = None
    tool_executor = None
    audit_log = None

    if cfg.tools.enabled:
        from tools.registry import ToolRegistry
        from core.tool_executor import ToolExecutor
        from core.audit import AuditLog

        # Audit log
        audit_db_path = cfg.resolve(cfg.paths.audit_log_db)
        audit_log = AuditLog(audit_db_path)
        await audit_log.initialize()
        log.info("audit_log_initialized", db_path=str(audit_db_path))

        # Tool registry — auto-discover tools from tools/ directory
        tool_registry = ToolRegistry()
        tools_dir = project_root / "tools"
        tool_count = tool_registry.discover(tools_dir)
        log.info("tools_discovered", count=tool_count, dir=str(tools_dir))

        # Tool executor
        tool_executor = ToolExecutor(
            registry=tool_registry,
            audit=audit_log,
            bus=bus,
            cfg=cfg.tools,
        )

        # Ensure sandbox and notes directories exist
        sandbox_dir = cfg.resolve(cfg.tools.sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        notes_dir = cfg.resolve(cfg.tools.notes_dir)
        notes_dir.mkdir(parents=True, exist_ok=True)

    # --- 7. Start audio and TTS ---
    await audio_input.start()
    await audio_output.start()
    await tts.start()

    # --- 7.5 Initialize Phase 4/5 Memory WAL & Catch-up Engine ---
    from core.memory.wal import MemoryWAL
    from core.memory.catch_up import run_catch_up
    from core.memory.vector_engine import VectorEngine
    from core.memory.graph import KnowledgeGraph

    data_dir = cfg.resolve(cfg.paths.data_dir)
    
    wal_db_path = data_dir / "memory_wal.db"
    wal = MemoryWAL(wal_db_path)
    await wal.initialize()
    
    vector_db_path = data_dir / "lancedb"
    vector_engine = VectorEngine(vector_db_path)
    await vector_engine.initialize()
    
    graph_db_path = data_dir / "jarvis_graph.db"
    graph = KnowledgeGraph(graph_db_path)
    await graph.initialize()

    # Start the Catch-up Engine asynchronously so it doesn't block boot
    asyncio.create_task(run_catch_up(wal, vector_engine, graph))

    # --- 8. Create orchestrator ---
    from core.orchestrator import VoiceOrchestrator

    orchestrator = VoiceOrchestrator(
        cfg=cfg,
        audio_input=audio_input,
        audio_output=audio_output,
        wake_detector=wake,
        vad=vad,
        stt=stt,
        brain_router=brain_router,
        tts=tts,
        timer=timer,
        bus=bus,
        tool_executor=tool_executor,
        tool_registry=tool_registry,
        wal=wal,
        vector_engine=vector_engine,
        knowledge_graph=graph,
    )

    # --- 9. Run the voice loop ---
    log.info(
        "jarvis_ready",
        brain_primary=cfg.brains.brain1_primary.name,
        brain_specialist_enabled=cfg.brains.brain2_specialist.enabled,
        conversation_history_enabled=cfg.conversation.enabled,
        aec_enabled=cfg.aec.enabled,
        tools_enabled=cfg.tools.enabled,
        tool_count=tool_registry.count if tool_registry else 0,
    )
    print("J.A.R.V.I.S. Phase 2 ready. Say the wake word to start.")

    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        pass
    finally:
        # --- 10. Cleanup ---
        log.info("jarvis_shutting_down")
        orchestrator.shutdown()
        console_task.cancel()

        # M30: shut the WebSocket server down gracefully so
        # connected HUD clients receive a clean close.
        if ws_handle is not None:
            try:
                await ws_handle.stop()
            except Exception as e:
                log.warning("websocket_stop_failed", error=str(e))

        if metrics_server is not None:
            try:
                await metrics_server.stop()
            except Exception as e:
                log.warning("metrics_server_stop_failed", error=str(e))

        await tts.stop()
        await audio_input.stop()
        await audio_output.stop()

        await llm.cleanup()
        await stt.cleanup()
        await vad.cleanup()
        await wake.cleanup()

        # Close the persistent Oracle OCI HTTP connection pool.
        if gemini_client is not None:
            try:
                await gemini_client.close()
            except Exception as e:
                log.debug("gemini_client_close_failed", error=str(e))

        if audit_log:
            await audit_log.close()

        # C8 fix: remove the lock file so a future enroll.py can run.
        from security import biometrics as _biometrics_mod
        _biometrics_mod.remove_lock_file()

        if hud_proc:
            try:
                hud_proc.terminate()
            except Exception as e:
                # M1: log the HUD terminate failure. The HUD may
                # already be exiting on its own, so a failure here
                # is usually benign — log at debug to keep INFO
                # output clean.
                log.debug("hud_terminate_failed", error=str(e))

        log.info("jarvis_stopped")


def _acquire_single_instance_mutex() -> bool:
    """Acquire a Linux fcntl file lock to enforce single-instance execution.

    Returns True if this is the ONLY running JARVIS instance and we
    successfully claimed the lock. Returns False if another instance
    already holds it.

    The lock is automatically released by the OS when this process exits,
    even if it crashes.
    """
    import fcntl
    
    lock_file = "/tmp/jarvis_single_instance.lock"
    # O_RDWR | O_CREAT
    try:
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError:
        return False
        
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # We intentionally leak the file descriptor 'fd' so the lock 
        # is held for the lifetime of the process.
        return True
    except (IOError, OSError):
        return False


def entry() -> None:
    """Synchronous entry point for `python -m core.main`."""
    # ------------------------------------------------------------------ #
    # Single-instance guard (MUST be first, before asyncio.run or any     #
    # imports that touch audio/GPU).                                       #
    #                                                                      #
    # Why here and not in main()? We want to exit BEFORE any async setup  #
    # or resource allocation. If we're the duplicate, we do nothing and   #
    # leave immediately. Exit code 0 is critical: the VBS boot-retry loop #
    # only retries on code 1 (crash). Code 0 = "clean stop", VBS halts.  #
    # ------------------------------------------------------------------ #
    if not _acquire_single_instance_mutex():
        print(
            "[JARVIS] Another instance is already running (lock held). "
            "Exiting cleanly."
        )
        import sys
        sys.exit(0)

    async def _run() -> None:
        loop = asyncio.get_running_loop()

        # Handle Ctrl+C gracefully
        def _signal_handler() -> None:
            for task in asyncio.all_tasks(loop):
                task.cancel()

        # L25: ``signal`` is imported at module level (line 13)
        # so it can be used in the POSIX branch below. On
        # Windows the ``add_signal_handler`` call raises
        # ``NotImplementedError`` and we fall through, leaving
        # the import unused at runtime on this platform. The
        # unused-import check would flag this — the
        # ``# noqa: F401`` keeps the import alive for the POSIX
        # branch and silences the warning on Windows.
        for sig in (signal.SIGINT, signal.SIGTERM):  # noqa: F401
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        await main()

    # H35: let the event loop exit naturally. ``os._exit(0)``
    # bypasses all pending ``finally`` blocks (daily compaction
    # write, lock file cleanup, database close, etc.) and is
    # only appropriate for hard error recovery. The natural
    # asyncio.run() return already lets the loop drain
    # gracefully; KeyboardInterrupt and CancelledError are
    # handled the same way (the ``await main()`` line is the
    # one that should have already run the cleanup logic).
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    entry()
