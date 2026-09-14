"""Security and verification tools for J.A.R.V.I.S.

Provides voice-callable tools to lock and unlock the system,
change the master password, and query the current auth state.

L16: this module exposes 4 tools — ``unlock_jarvis``,
``lock_jarvis``, ``get_auth_status``, and
``set_master_password`` — all ``verify=False`` because
the unlock process IS the verification; you cannot
require a lock to unlock itself.

These tools are gated on ``verification.enabled: true`` in config.yaml.
When the gate is off, every tool here returns an error that points the
user at the setup CLI. The first-time master password can ONLY be
created via ``enroll.py`` — never by a tool the LLM can invoke.
"""

from pathlib import Path

from security.verification import verification_manager
from tools.registry import ToolResult, tool


def _verification_disabled_error() -> ToolResult:
    return ToolResult(
        success=False,
        error=(
            "Security verification is disabled in config.yaml. "
            "Set verification.enabled: true and create a master password "
            "by running: python scripts/enroll.py"
        ),
    )


def _no_password_set_error() -> ToolResult:
    return ToolResult(
        success=False,
        error=(
            "No master password is configured. Run the first-time "
            "setup CLI: python scripts/enroll.py"
        ),
    )


def _is_enabled() -> bool:
    """Read verification.enabled from config without forcing a load."""
    try:
        from core.config import load_config
        return bool(load_config().verification.enabled)
    except Exception:
        return False


@tool(name="unlock_jarvis", verify=False, category="security")
async def unlock_jarvis(password: str) -> ToolResult:
    """Unlock J.A.R.V.I.S. security for 15 minutes using the master password.

    Refuses to operate if no master password has been set yet — that
    setup is the job of the ``enroll.py`` CLI, not a voice-callable
    tool. This prevents the first-run bypass where saying "unlock
    with password X" would both create X and unlock the system.
    """
    if not _is_enabled():
        return _verification_disabled_error()
    if not verification_manager.has_password():
        return _no_password_set_error()

    success = await verification_manager.unlock(password)
    if success:
        return ToolResult(
            success=True,
            data="System is now UNLOCKED. All tools available for 15 minutes.",
        )
    return ToolResult(success=False, error="Incorrect password. Access denied.")


@tool(name="lock_jarvis", verify=False, category="security")
async def lock_jarvis() -> ToolResult:
    """Immediately lock J.A.R.V.I.S. security, blocking all destructive tools."""
    if not _is_enabled():
        return _verification_disabled_error()
    await verification_manager.lock_now()
    return ToolResult(success=True, data="System locked. Destructive tools are now blocked.")


@tool(name="get_auth_status", verify=False, category="security")
async def get_auth_status() -> ToolResult:
    """Check whether J.A.R.V.I.S. is currently locked or unlocked."""
    if not _is_enabled():
        return ToolResult(
            success=True,
            data="Security verification is disabled in config.yaml.",
        )
    from security.verification import AuthState
    state = await verification_manager.get_state()
    if state == AuthState.UNLOCKED:
        import time
        remaining_s = max(0, int(verification_manager.unlock_expiry_time - time.time()))
        remaining_min = remaining_s // 60
        remaining_sec = remaining_s % 60
        return ToolResult(
            success=True,
            data=f"System is UNLOCKED. Auto-locks in {remaining_min}m {remaining_sec}s.",
        )
    return ToolResult(success=True, data="System is LOCKED. Use unlock_jarvis to enable destructive tools.")


@tool(name="set_master_password", verify=False, category="security")
async def set_master_password(current_password: str, new_password: str) -> ToolResult:
    """Change the J.A.R.V.I.S. master password.

    Requires the current password for verification before allowing the change.
    Refuses to operate when no master password has been set up — that
    initial setup is the job of ``enroll.py``.
    """
    if not _is_enabled():
        return _verification_disabled_error()
    if not verification_manager.has_password():
        return _no_password_set_error()

    # Verify the current password first
    success = await verification_manager.unlock(current_password)
    if not success:
        return ToolResult(success=False, error="Current password is incorrect.")

    try:
        verification_manager.set_password(new_password, require_prior=True)
    except PermissionError as e:
        return ToolResult(success=False, error=str(e))
    return ToolResult(success=True, data="Master password updated successfully.")
