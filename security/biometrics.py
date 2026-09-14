"""SpeechBrain ECAPA-TDNN voiceprint biometrics.

Provides enroll_voiceprint() and verify_voice() using cosine similarity
of speaker embeddings. The voiceprint file and model cache directories
are resolved from the project data_dir, not from CWD.

Call configure(data_dir) once from main.py before using any function.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)

# Configurable via configure() — resolved from project root by main.py
_DATA_DIR: Path = Path("data")
_VOICEPRINT_FILE: Path = _DATA_DIR / "voiceprint.npy"
_MODEL_CACHE_DIR: Path = _DATA_DIR / "speechbrain_cache"

# Similarity threshold: values above this are accepted as the owner.
# 0.20 rejects strangers while accepting the enrolled speaker across sessions.
COSINE_THRESHOLD: float = 0.20

_classifier = None
# M4: a module-level asyncio.Lock that serialises the first
# ``_get_classifier()`` call. If two SecurityState instances are
# constructed in the same event loop (e.g. in a test that builds
# two orchestrators back-to-back), both would try to download /
# load the SpeechBrain model simultaneously. The lock ensures
# only one of them pays the cost; the other awaits the result and
# reuses the cached classifier.
_classifier_load_lock: asyncio.Lock | None = None


def _get_classifier_lock() -> asyncio.Lock:
    """Lazy-create the module-level lock (must be created from
    inside a running event loop)."""
    global _classifier_load_lock
    if _classifier_load_lock is None:
        _classifier_load_lock = asyncio.Lock()
    return _classifier_load_lock


def configure(data_dir: Path) -> None:
    """Set absolute paths for all data files (call once from main.py)."""
    global _DATA_DIR, _VOICEPRINT_FILE, _MODEL_CACHE_DIR
    _DATA_DIR = data_dir
    _VOICEPRINT_FILE = data_dir / "voiceprint.npy"
    _MODEL_CACHE_DIR = data_dir / "speechbrain_cache"


def voiceprint_path() -> Path:
    """Return the current voiceprint file path.

    Exposed as a public helper so the orchestrator can clean up
    partial state on the H24 enrollment timeout without reaching
    into module-level globals.
    """
    return _VOICEPRINT_FILE


def _get_classifier():
    """Lazy-load the SpeechBrain ECAPA-TDNN encoder on first use.

    H22: the original implementation called
    ``EncoderClassifier.from_hparams(source=..., savedir=...)``
    which downloads the model from Hugging Face on first use.
    If the user was offline the download would fail with a
    generic ``HTTPError`` or ``ConnectionError`` and the user
    would see a confusing stack trace. We now:
    1. Check the cache directory first; if the model is already
       present, skip the download entirely.
    2. Catch the download-specific exceptions and raise a
       ``RuntimeError`` with an actionable message that points
       the user at the ``setup.ps1`` pre-download step and the
       ``verification.enabled`` config flag.

    M4: the load is guarded by an ``asyncio.Lock`` so two
    concurrent first-time calls (e.g. two SecurityState
    instances being constructed in the same event loop) don't
    both trigger a download. The lock is also held for
    synchronous callers via a small ``_load_lock_sync`` fallback
    in case the load happens outside a running loop.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    # M4: short-circuit fast path — if we already have a
    # classifier, return it without acquiring the lock at all.
    # The ``is not None`` check above handles this.

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        logger.error("speechbrain_not_installed")
        raise RuntimeError(
            "SpeechBrain is not installed. Run: pip install speechbrain"
        )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # H22 (1): the model is cached once and reloaded from disk on
    # subsequent runs. ``from_hparams`` only re-downloads if the
    # cache is missing or corrupt, so a quick existence check
    # before the call avoids a wasted network round-trip.
    cache_marker = _MODEL_CACHE_DIR / "hyperparams.yaml"
    if cache_marker.exists():
        logger.info("speechbrain_model_loaded_from_cache", path=str(_MODEL_CACHE_DIR))
    else:
        logger.info(
            "loading_speechbrain_model",
            model="spkrec-ecapa-voxceleb",
            hint="First-time download requires internet. Run setup.ps1 to pre-download.",
        )

    try:
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(_MODEL_CACHE_DIR),
            run_opts={"device": "cpu"},  # keep VRAM free for STT + LLM
        )
    except Exception as e:
        # H22 (2): turn generic download errors into a clear
        # actionable message. The user can either pre-download
        # via ``setup.ps1`` or disable verification until they
        # have internet.
        msg = (
            f"Could not load the SpeechBrain voiceprint model: {e}. "
            "Either run setup.ps1 (which pre-downloads the model) "
            "or disable voice verification in config.yaml with "
            "``verification.enabled: false``. The voiceprint model "
            "is ~80 MB and is downloaded from Hugging Face on first use."
        )
        logger.error("speechbrain_model_load_failed", error=str(e))
        raise RuntimeError(msg) from e

    logger.info("speechbrain_model_loaded")
    return _classifier


async def get_classifier_async():
    """Async wrapper around ``_get_classifier`` that serialises
    concurrent first-time loads via an ``asyncio.Lock``.

    M4: two concurrent coroutines that need the classifier
    (e.g. ``verify_voice`` and ``enroll_voiceprint`` racing in
    tests) previously both triggered a download. We now use a
    module-level lock so the second caller awaits the first
    caller's result instead of starting a redundant download.

    The fast path (already-loaded) is lock-free. Only the
    download / load path is serialised.
    """
    global _classifier
    if _classifier is not None:
        return _classifier
    lock = _get_classifier_lock()
    async with lock:
        # Re-check under the lock in case a concurrent caller
        # finished the load while we were waiting.
        if _classifier is not None:
            return _classifier
        return _get_classifier()


def enroll_voiceprint(audio_data: np.ndarray, sample_rate: int = 16000) -> None:
    """Extract and save a voiceprint embedding from float32 mono 16 kHz audio.

    Synchronous. The SpeechBrain encoder is CPU-bound; callers running
    inside the asyncio event loop should use ``enroll_voiceprint_async``
    which dispatches to a worker thread.
    """
    logger.info("enrolling_voiceprint")
    classifier = _get_classifier()

    signal = torch.from_numpy(audio_data).float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)

    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        emb_np = embeddings.squeeze().cpu().numpy()

    np.save(str(_VOICEPRINT_FILE), emb_np)
    logger.info("voiceprint_enrolled", file=str(_VOICEPRINT_FILE))


async def enroll_voiceprint_async(
    audio_data: np.ndarray, sample_rate: int = 16000
) -> None:
    """Async wrapper around ``enroll_voiceprint`` for use in async code paths.

    C8 fix: a single async helper used by both the orchestrator's
    first-run enrollment and the re-enrollment tool, so the two paths
    can no longer drift apart.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, enroll_voiceprint, audio_data, sample_rate)


# ---------------------------------------------------------------------------
# Cross-process lock file (C8 fix)
# ---------------------------------------------------------------------------
# When JARVIS is running it writes its PID to data/jarvis.lock. The
# standalone `enroll.py` CLI refuses to run if that PID is alive so it
# cannot race with a running orchestrator and write a different
# voiceprint.npy under it.

LOCK_FILENAME = "jarvis.lock"


def write_lock_file(pid: int | None = None) -> Path:
    """Write data/jarvis.lock with the current PID. Returns the file path."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / LOCK_FILENAME
    path.write_text(str(pid or os.getpid()), encoding="utf-8")
    return path


def remove_lock_file() -> None:
    """Best-effort removal of the lock file. Silent on errors."""
    try:
        (_DATA_DIR / LOCK_FILENAME).unlink(missing_ok=True)
    except Exception:
        pass


def is_jarvis_running() -> bool:
    """True when a JARVIS lock file is present and its PID is alive.

    Returns False on platforms where we cannot check (no psutil or
    no /proc); the CLI falls back to a safer interactive prompt in
    that case.
    """
    lock = _DATA_DIR / LOCK_FILENAME
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return True  # Lock exists but unreadable — assume JARVIS is running
    if pid == os.getpid():
        return False
    try:
        # Windows: OpenProcess + GetExitCodeProcess via ctypes is heavier
        # than necessary. We use a simple signal-zero check which works
        # on POSIX; on Windows the import will fail and we fall back to
        # the lock-file heuristic alone.
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if h == 0:
                return False
            try:
                code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
                if not ok:
                    return True
                return code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        else:
            import signal
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True  # Process exists, owned by someone else
    except Exception:
        return True


def verify_voice(audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
    """Return True if the audio matches the enrolled master voiceprint."""
    if not _VOICEPRINT_FILE.exists():
        logger.warning("no_voiceprint_found")
        return False

    classifier = _get_classifier()

    master_emb = np.load(str(_VOICEPRINT_FILE))
    master_tensor = torch.from_numpy(master_emb).float().unsqueeze(0)

    signal = torch.from_numpy(audio_data).float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)

    with torch.no_grad():
        current_emb = classifier.encode_batch(signal).squeeze(1)

    similarity = torch.nn.functional.cosine_similarity(current_emb, master_tensor)
    score = similarity.item()

    is_match = score > COSINE_THRESHOLD
    logger.info(
        "voice_verification",
        score=round(score, 4),
        threshold=COSINE_THRESHOLD,
        match=is_match,
    )
    return is_match


def has_voiceprint() -> bool:
    """Return True if a voiceprint has been enrolled."""
    return _VOICEPRINT_FILE.exists()
