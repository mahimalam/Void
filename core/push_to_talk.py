"""Push-to-talk keyboard listener.

2026-06-20: registers a Windows low-level keyboard hook that
fires a callback when Ctrl+Space is pressed. The callback
sets an asyncio.Event on the orchestrator, which the
barge-in monitor watches in addition to VAD / wake-word
triggers. This is the always-works escape hatch: it
doesn't depend on AEC, VAD, or microphone quality — just
a direct OS-level keyboard signal.

Why a low-level hook (WH_KEYBOARD_LL) instead of msvcrt:
- msvcrt.getch() only reads from the console's stdin. If
  JARVIS is launched via a .vbs script (no console), msvcrt
  blocks forever. WH_KEYBOARD_LL works for any process and
  is the same mechanism AutoHotkey uses.
- The hook runs on a dedicated thread (set by the Windows
  API), so the asyncio loop is never blocked. The callback
  posts a thread-safe event to the orchestrator via
  ``loop.call_soon_threadsafe``.

Why Ctrl+Space and not Space alone:
- Space alone would conflict with every text input on the
  user's machine (forms, games, etc.). Ctrl+Space is rare
  enough that it's safe to claim.

Disabling:
- The hook can be disabled by setting
  ``orchestrator.push_to_talk.enabled = false`` in
  config.yaml. Default is true.
"""

from __future__ import annotations

import asyncio
import ctypes
import sys
import threading

from core.logging_setup import get_logger

if sys.platform == "win32":
    from ctypes import wintypes
    _USER32 = ctypes.windll.user32
    _KERNEL32 = ctypes.windll.kernel32

    # Virtual key codes we care about.
    _VK_CONTROL = 0x11
    _VK_SPACE = 0x20

    # Low-level keyboard hook constant.
    _WH_KEYBOARD_LL = 13

    # Low-level keyboard event codes.
    _WM_KEYDOWN = 0x0100
    _WM_SYSKEYDOWN = 0x0104

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        """Low-level keyboard hook event struct (Windows API)."""
        _fields_ = [
            ("vkCode", wintypes.DWORD),
            ("scanCode", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    # Function pointer type for the hook callback.
    _LowLevelKeyboardProc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.INT, wintypes.WPARAM, ctypes.POINTER(KBDLLHOOKSTRUCT)
    )
else:
    _USER32 = None
    _KERNEL32 = None
    _VK_CONTROL = None
    _VK_SPACE = None
    _WH_KEYBOARD_LL = None
    _WM_KEYDOWN = None
    _WM_SYSKEYDOWN = None
    KBDLLHOOKSTRUCT = None
    _LowLevelKeyboardProc = None


class PushToTalkListener:
    """Background keyboard listener that fires on Ctrl+Space.

    Usage:
        listener = PushToTalkListener(loop, on_activate)
        listener.start()       # non-blocking
        ...
        listener.stop()        # unhooks
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_activate: callable,
    ) -> None:
        self._loop = loop
        self._on_activate = on_activate
        self._log = get_logger("push_to_talk")
        self._hook_id: int | None = None
        self._thread_id: int | None = None
        self._running = False
        self._lock = threading.Lock()
        # Track the current Ctrl+Space state so we don't
        # fire on key repeat. Windows sends WM_KEYDOWN
        # multiple times while a key is held; we only want
        # one fire per physical press.
        self._was_pressed = False

    def start(self) -> bool:
        """Install the keyboard hook. Returns False on non-Windows or failure."""
        if sys.platform != "win32":
            # Push-to-talk uses Windows low-level keyboard hooks — expected no-op on Linux.
            self._log.debug("push_to_talk_linux_no_op", platform=sys.platform)
            return False
        if self._running:
            return True

        with self._lock:
            if self._running:
                return True
            # The hook callback MUST be a ctypes function
            # pointer, not a Python method. We use
            # WINFUNCTYPE which keeps a reference and
            # prevents GC.
            self._thread_id = _KERNEL32.GetCurrentThreadId()

            @_LowLevelKeyboardProc
            def _hook_callback(nCode, wParam, lParam):
                try:
                    if nCode >= 0 and lParam:
                        vk = lParam.contents.vkCode
                        if wParam in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
                            if vk == _VK_SPACE:
                                ctrl_down = (
                                    _USER32.GetAsyncKeyState(_VK_CONTROL) & 0x8000
                                ) != 0
                                if ctrl_down and not self._was_pressed:
                                    self._was_pressed = True
                                    # Post to asyncio thread.
                                    try:
                                        self._loop.call_soon_threadsafe(
                                            self._on_activate
                                        )
                                    except RuntimeError:
                                        # Loop closed during shutdown.
                                        pass
                            elif vk == _VK_CONTROL:
                                # Control released first; reset latch.
                                if not (_USER32.GetAsyncKeyState(_VK_SPACE) & 0x8000):
                                    self._was_pressed = False
                except Exception:
                    # The hook callback must NEVER raise, or
                    # Windows terminates the process. Swallow
                    # all exceptions.
                    pass
                return _USER32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

            # The hook procedure must reference the ctypes
            # callback object directly (not a closure),
            # otherwise ctypes can't read the function
            # pointer. We store it on self to keep it alive.
            self._hook_callback = _hook_callback

            self._hook_id = _USER32.SetWindowsHookExW(
                _WH_KEYBOARD_LL,
                self._hook_callback,
                None,  # hMod = NULL for low-level hooks
                0,     # dwThreadId = 0 = all threads
            )
            if not self._hook_id:
                err = ctypes.get_last_error() or "unknown"
                self._log.warning("push_to_talk_hook_failed", error_code=err)
                return False
            self._running = True
            self._log.info("push_to_talk_started", hotkey="Ctrl+Space")
            return True

    def stop(self) -> None:
        """Unhook the keyboard listener."""
        with self._lock:
            if not self._running:
                return
            if self._hook_id:
                _USER32.UnhookWindowsHookEx(self._hook_id)
            self._hook_id = None
            self._running = False
            self._log.info("push_to_talk_stopped")
