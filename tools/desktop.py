"""Desktop automation and UI control tools.

Uses PyAutoGUI for cross-platform desktop control (Linux, Windows, macOS).
Guarded against mouseinfo's sys.exit(1) on Linux systems without Tkinter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

from tools.registry import ToolResult, tool


def _get_pyautogui() -> tuple[Any | None, str | None]:
    """Safely import pyautogui, guarding against mouseinfo's sys.exit(1) on Linux without Tkinter."""
    if sys.platform.startswith("linux"):
        try:
            import tkinter  # noqa: F401
        except (ImportError, Exception, SystemExit, BaseException):
            return (
                None,
                "Desktop automation requires Tkinter on Linux. Please install it using: sudo apt install python3-tk",
            )
    try:
        import pyautogui
        return pyautogui, None
    except (ImportError, Exception, SystemExit, BaseException) as e:
        return None, f"PyAutoGUI is unavailable: {e}"


@tool(name="click_screen", verify=False, category="desktop")
async def click_screen(x: int, y: int) -> ToolResult:
    """Clicks a specific coordinate on the screen.

    Args:
        x: The X coordinate.
        y: The Y coordinate.
    """
    pyautogui, err = _get_pyautogui()
    if not pyautogui:
        return ToolResult(success=False, error=err or "pyautogui unavailable")

    try:
        pyautogui.click(x=x, y=y)
        return ToolResult(success=True, data=f"Clicked at ({x}, {y})")
    except (Exception, BaseException) as e:
        return ToolResult(success=False, error=f"Failed to click: {e}")


@tool(name="type_text", verify=False, category="desktop")
async def type_text(text: str) -> ToolResult:
    """Types text into the currently focused window.

    Args:
        text: The text to type.
    """
    pyautogui, err = _get_pyautogui()
    if not pyautogui:
        return ToolResult(success=False, error=err or "pyautogui unavailable")

    try:
        pyautogui.typewrite(text)
        return ToolResult(success=True, data=f"Typed text: {text}")
    except (Exception, BaseException) as e:
        return ToolResult(success=False, error=f"Failed to type text: {e}")


@tool(name="press_key", verify=False, category="desktop")
async def press_key(key: str) -> ToolResult:
    """Presses a special keyboard key or combination.

    Args:
        key: The key to press. Use PyAutoGUI format. Examples: "enter", "esc", "ctrl,c", "alt,tab".
    """
    pyautogui, err = _get_pyautogui()
    if not pyautogui:
        return ToolResult(success=False, error=err or "pyautogui unavailable")

    try:
        if "," in key:
            # Hotkey like "ctrl,c"
            keys = [k.strip() for k in key.split(",")]
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(key)
        return ToolResult(success=True, data=f"Pressed key: {key}")
    except (Exception, BaseException) as e:
        return ToolResult(success=False, error=f"Failed to press key: {e}")


@tool(name="focus_window", verify=False, category="desktop")
async def focus_window(title: str) -> ToolResult:
    """Brings a specific application window to the foreground by matching its title.

    Args:
        title: Part or all of the window title. E.g., "Chrome", "Terminal".
    """
    try:
        if sys.platform == "linux":
            result = subprocess.run(["wmctrl", "-a", title], capture_output=True, text=True)
            if result.returncode == 0:
                return ToolResult(success=True, data=f"Brought window matching '{title}' to foreground.")
            return ToolResult(success=False, error=f"Could not find or focus window '{title}'.")
        else:
            return ToolResult(success=False, error="focus_window is only implemented for Linux via wmctrl.")
    except FileNotFoundError:
        return ToolResult(success=False, error="wmctrl is not installed. Run: sudo apt install wmctrl")
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to focus window: {e}")
