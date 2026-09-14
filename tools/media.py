"""Media control tools for Windows and Linux (Play/Pause/Skip tracks, Volume control)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from typing import Tuple

from tools.registry import ToolResult, tool


def _linux_media_action(action: str) -> Tuple[bool, str]:
    """Execute media action (play_pause, next, previous) on Linux."""
    # 1. First try playerctl if installed
    if shutil.which("playerctl"):
        cmd_map = {"play_pause": "play-pause", "next": "next", "previous": "previous"}
        sub_cmd = cmd_map.get(action, action)
        try:
            res = subprocess.run(["playerctl", sub_cmd], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return True, f"Media {action.replace('_', ' ')} executed via playerctl."
        except Exception:
            pass

    # 2. Try standard MPRIS via dbus-send
    if shutil.which("dbus-send"):
        try:
            list_cmd = [
                "dbus-send",
                "--session",
                "--dest=org.freedesktop.DBus",
                "--type=method_call",
                "--print-reply",
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus.ListNames",
            ]
            res = subprocess.run(list_cmd, capture_output=True, text=True, timeout=3)
            mpris_names = re.findall(r'"(org\.mpris\.MediaPlayer2\.[^"]+)"', res.stdout)
            if mpris_names:
                target = mpris_names[0]
                mpris_method = {
                    "play_pause": "PlayPause",
                    "next": "Next",
                    "previous": "Previous",
                }.get(action, "PlayPause")
                call_cmd = [
                    "dbus-send",
                    "--session",
                    f"--dest={target}",
                    "--type=method_call",
                    "/org/mpris/MediaPlayer2",
                    f"org.mpris.MediaPlayer2.Player.{mpris_method}",
                ]
                call_res = subprocess.run(call_cmd, capture_output=True, text=True, timeout=3)
                if call_res.returncode == 0:
                    player_label = target.split(".")[-1]
                    return True, f"Sent {action.replace('_', ' ')} to {player_label}."
        except Exception as e:
            return False, f"DBus media control error: {e}"

    return False, "No active media player detected (playerctl or MPRIS DBus required)."


def _linux_set_volume(direction: str, steps: int) -> Tuple[bool, str]:
    """Control system volume on Linux using pactl or wpctl."""
    delta_pct = steps * 2
    if shutil.which("pactl"):
        try:
            if direction == "mute":
                res = subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    return True, "Toggled system mute."
            else:
                val = f"+{delta_pct}%" if direction == "up" else f"-{delta_pct}%"
                res = subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", val], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    return True, f"Turned volume {direction} by {delta_pct}%."
        except Exception as e:
            return False, f"pactl error: {e}"

    if shutil.which("wpctl"):
        try:
            if direction == "mute":
                res = subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    return True, "Toggled system mute."
            else:
                val = f"{delta_pct}%+" if direction == "up" else f"{delta_pct}%-"
                res = subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", val], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    return True, f"Turned volume {direction} by {delta_pct}%."
        except Exception as e:
            return False, f"wpctl error: {e}"

    return False, "Neither pactl nor wpctl is available to adjust volume on Linux."


@tool(name="play_pause_media", verify=False, category="media")
async def play_pause_media() -> ToolResult:
    """Toggle play/pause for the currently active media player (e.g. Spotify, YouTube, VLC)."""
    if sys.platform == "win32":
        try:
            import win32api
            import win32con

            # 0xB3 = VK_MEDIA_PLAY_PAUSE
            win32api.keybd_event(0xB3, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
            win32api.keybd_event(0xB3, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
            return ToolResult(success=True, data="Toggled media play/pause state.")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to control media on Windows: {e}")

    # Linux implementation
    ok, msg = _linux_media_action("play_pause")
    return ToolResult(success=ok, data=msg if ok else "", error="" if ok else msg)


@tool(name="next_track", verify=False, category="media")
async def next_track() -> ToolResult:
    """Skip to the next media track."""
    if sys.platform == "win32":
        try:
            import win32api
            import win32con

            # 0xB0 = VK_MEDIA_NEXT_TRACK
            win32api.keybd_event(0xB0, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
            win32api.keybd_event(0xB0, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
            return ToolResult(success=True, data="Skipped to next track.")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to control media on Windows: {e}")

    # Linux implementation
    ok, msg = _linux_media_action("next")
    return ToolResult(success=ok, data=msg if ok else "", error="" if ok else msg)


@tool(name="previous_track", verify=False, category="media")
async def previous_track() -> ToolResult:
    """Go back to the previous media track."""
    if sys.platform == "win32":
        try:
            import win32api
            import win32con

            # 0xB1 = VK_MEDIA_PREV_TRACK
            win32api.keybd_event(0xB1, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
            win32api.keybd_event(0xB1, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
            return ToolResult(success=True, data="Went back to previous track.")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to control media on Windows: {e}")

    # Linux implementation
    ok, msg = _linux_media_action("previous")
    return ToolResult(success=ok, data=msg if ok else "", error="" if ok else msg)


@tool(name="set_volume", verify=False, category="media")
async def set_volume(direction: str, steps: int = 5) -> ToolResult:
    """Change the system volume. 'direction' must be 'up', 'down', or 'mute'. 'steps' controls how much."""
    direction = direction.lower()
    if direction not in ["up", "down", "mute"]:
        return ToolResult(success=False, error="Direction must be 'up', 'down', or 'mute'.")

    steps = max(1, steps)

    if sys.platform == "win32":
        try:
            import win32api
            import win32con

            VK_VOLUME_MUTE = 0xAD
            VK_VOLUME_DOWN = 0xAE
            VK_VOLUME_UP = 0xAF

            if direction == "mute":
                win32api.keybd_event(VK_VOLUME_MUTE, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
                win32api.keybd_event(VK_VOLUME_MUTE, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
                return ToolResult(success=True, data="Toggled system mute.")

            vk = VK_VOLUME_UP if direction == "up" else VK_VOLUME_DOWN
            for _ in range(steps):
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.01)

            return ToolResult(success=True, data=f"Turned volume {direction} by {steps*2}%.")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to control volume on Windows: {e}")

    # Linux implementation
    ok, msg = _linux_set_volume(direction, steps)
    return ToolResult(success=ok, data=msg if ok else "", error="" if ok else msg)
