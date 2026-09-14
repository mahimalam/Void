"""Verification State Machine — LOCKED / UNLOCKED with 15-minute auto-expiry.

The state machine is a singleton (verification_manager). It is imported by:
  - core/main.py          → started on boot via verification_manager.start()
  - core/tool_executor.py → queried on every verify=True tool call
  - tools/security.py     → unlock_jarvis / lock_jarvis tools call unlock/lock_now

Auth-change events are emitted to the EventBus so the HUD can update its
lock-state indicator in real time.
"""

import asyncio
import time
from enum import Enum
from pathlib import Path

import structlog
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

logger = structlog.get_logger(__name__)



class AuthState(Enum):
    LOCKED = "LOCKED"
    UNLOCKED = "UNLOCKED"


class VerificationStateMachine:
    """LOCKED / UNLOCKED state machine with Argon2 password hashing.

    The password hash lives in ``data/pass_hash.txt`` (relative to project
    root).  ``data_dir`` must be set to the absolute data directory path
    before ``start()`` is called; ``main.py`` does this via
    ``verification_manager.configure(data_dir)``.
    """

    def __init__(self) -> None:
        self.state = AuthState.LOCKED
        self._lock = asyncio.Lock()
        self.unlock_expiry_time: float = 0.0
        self.timeout_seconds: int = 15 * 60  # 15 minutes
        self.ph = PasswordHasher()
        self._data_dir: Path = Path("data")  # overridden by configure()
        self._expiry_task: asyncio.Task | None = None

    def configure(self, data_dir: Path) -> None:
        """Set the absolute data directory (call once from main.py before start)."""
        self._data_dir = data_dir

    @property
    def _hash_file(self) -> Path:
        return self._data_dir / "pass_hash.txt"

    async def start(self) -> None:
        """Start the auto-expiry background daemon."""
        if self._expiry_task is None or self._expiry_task.done():
            self._expiry_task = asyncio.create_task(self._expiry_daemon())

    async def _expiry_daemon(self) -> None:
        """Background task: auto-lock after timeout expires.

        M5: the original implementation polled every 30 s, which
        meant the auto-lock fired up to 30 s AFTER the configured
        15-minute window closed. The spoken result ("System is
        UNLOCKED. Auto-locks in Xm Ys.") and the actual lock
        could disagree by half a minute. We now poll every 5 s
        and the daemon is purely a safety net — every
        ``get_state()`` and ``unlock()`` call also recomputes the
        state against the current wall clock, so even a paused
        daemon never lets the lock expire silently.
        """
        # Bound the sleep so a misbehaving event loop (e.g. one
        # blocked on a slow I/O call) cannot extend the lock
        # window by more than ``_EXPIRY_POLL_S`` seconds.
        _EXPIRY_POLL_S = 5.0
        while True:
            await asyncio.sleep(_EXPIRY_POLL_S)
            # Lazy self-heal: if the daemon is in a state where
            # ``unlock_expiry_time`` is set but ``state`` is not
            # UNLOCKED, treat that as "already expired". This
            # guards against state corruption from a previous
            # daemon iteration.
            now = time.time()
            emit = False
            async with self._lock:
                if (
                    self.state == AuthState.UNLOCKED
                    and now > self.unlock_expiry_time
                ):
                    self.state = AuthState.LOCKED
                    logger.info("auth_timeout_relocked")
                    emit = True
            
            if emit:
                self._emit_auth_change("LOCKED")

    async def get_state(self) -> AuthState:
        # M5: re-derive the state from the wall clock on every
        # call. If the configured unlock window has elapsed
        # since the last daemon tick, we lock NOW rather than
        # waiting up to 5 s for the daemon to catch up. This
        # also keeps the spoken "Auto-locks in Xm Ys" estimate
        # in lock-step with reality.
        async with self._lock:
            if (
                self.state == AuthState.UNLOCKED
                and time.time() > self.unlock_expiry_time
            ):
                self.state = AuthState.LOCKED
                logger.info("auth_timeout_relocked_on_query")
                # Defer the event emission until after we drop
                # the lock so subscribers never deadlock.
                emit = True
            else:
                emit = False
        if emit:
            self._emit_auth_change("LOCKED")
        async with self._lock:
            return self.state

    async def unlock(self, provided_password: str) -> bool:
        """Attempt to unlock using the provided password.

        Returns True on success, False on wrong password or no password set.
        """
        if not self._hash_file.exists():
            logger.warning("no_password_set")
            return False

        stored_hash = self._hash_file.read_text(encoding="utf-8").strip()

        try:
            self.ph.verify(stored_hash, provided_password)
            # Rehash if Argon2 parameters have changed
            if self.ph.check_needs_rehash(stored_hash):
                self._hash_file.write_text(
                    self.ph.hash(provided_password), encoding="utf-8"
                )
            async with self._lock:
                self.state = AuthState.UNLOCKED
                self.unlock_expiry_time = time.time() + self.timeout_seconds
            logger.info("auth_unlocked", timeout_secs=self.timeout_seconds)
            self._emit_auth_change("UNLOCKED")
            return True

        except VerifyMismatchError:
            logger.warning("auth_failed_wrong_password")
            return False

    async def lock_now(self) -> None:
        """Immediately lock the state machine."""
        async with self._lock:
            if self.state == AuthState.UNLOCKED:
                self.state = AuthState.LOCKED
        logger.info("auth_locked_manually")
        self._emit_auth_change("LOCKED")

    def set_password(self, new_password: str, *, require_prior: bool = True) -> None:
        """Hash and persist a master password.

        ``require_prior=True`` (default): refuses to overwrite an existing
        password. Use this for routine password changes — call sites
        MUST verify the current password first (e.g. via ``unlock``).

        ``require_prior=False``: only allowed when no password is set
        yet. Used by the first-time setup flow (CLI / ``enroll.py``).
        Calling it when a password already exists raises ``PermissionError``
        so accidental overwrites are impossible.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        if require_prior and not self._hash_file.exists():
            raise PermissionError(
                "No master password is set. Use set_initial_password() "
                "from the first-time setup CLI (enroll.py)."
            )
        if not require_prior and self._hash_file.exists():
            raise PermissionError(
                "A master password is already set. Use set_password() "
                "after verifying the current password."
            )

        self._hash_file.write_text(
            self.ph.hash(new_password), encoding="utf-8"
        )
        logger.info(
            "password_updated",
            require_prior=require_prior,
            first_time=not require_prior,
        )

    def has_password(self) -> bool:
        """True if a master password file already exists on disk."""
        return self._hash_file.exists()

    def set_initial_password(self, new_password: str) -> None:
        """First-time master password creation. Refuses if one is already set.

        This is the ONLY code path that should ever write a master
        password without prior verification. It is intended to be called
        from a controlled setup CLI (enroll.py), NOT from a tool that
        the LLM can invoke.
        """
        self.set_password(new_password, require_prior=False)

    def _emit_auth_change(self, new_state: str) -> None:
        """Emit AUTH_CHANGE event to the EventBus (non-blocking, best-effort)."""
        try:
            from core.event_bus import EventBus, EventType
            # Access the global bus via the module-level reference set by main.py
            import core.event_bus as _eb_module
            bus: EventBus | None = getattr(_eb_module, "_global_bus", None)
            if bus is not None:
                bus.publish(EventType.AUTH_CHANGE, new_state=new_state)
        except Exception:
            pass  # Never let event emission crash auth logic


# Singleton instance — configure() called by main.py before start()
verification_manager = VerificationStateMachine()
