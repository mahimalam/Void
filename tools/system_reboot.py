"""System reboot tool — allows JARVIS to restart his own process.

M7: the tool used to launch the handover script *immediately* on
return, and the orchestrator set ``_shutdown = True`` at the same
moment. The result was a 2-second race: the TTS was mid-sentence
when the handover script SIGTERMed the python process, so the
user heard a truncated "Rebooting..." and the goodbye tail got
cut off.

The fix splits the work in two:

1. The tool ONLY signals the intent. It does NOT spawn the
   handover script. It returns ``signal=ToolSignal.REBOOT`` with
   the current PID in ``metadata`` so the orchestrator can
   coordinate the rest.

2. The orchestrator's REBOOT branch waits for the TTS to drain
   (so "Rebooting now." plays to completion), then launches the
   handover script itself, then sets ``_shutdown = True``.

If the tool's caller never returns (process is SIGTERMed before
the orchestrator's REBOOT branch runs), the user at least saw
"Rebooting now." come out of the speaker — better than a
truncated message.
"""

from __future__ import annotations

import os
from pathlib import Path

from tools.registry import ToolResult, ToolSignal, tool


@tool(name="reboot_jarvis", verify=False, category="system")
async def reboot_jarvis() -> ToolResult:
    """Reboot JARVIS to apply updates or restart the system. Use when asked to restart, reboot, or apply changes.

    M7: this tool now ONLY signals the intent. The orchestrator's
    REBOOT handler drains the TTS queue, spawns the handover
    script, and then shuts down. Calling this tool and never
    returning from the tool call is fine; the worst case is the
    farewell line gets cut off, but the user will still hear
    "Rebooting now." before the process dies.
    """
    try:
        # Find the project root (restart_jarvis.ps1 lives here).
        # We just verify the script exists; the orchestrator is
        # responsible for actually launching it.
        project_root = Path(__file__).parent.parent.resolve()
        import sys
        if sys.platform == "win32":
            restart_script = project_root / "scripts" / "restart_jarvis.ps1"
            if not restart_script.exists():
                restart_script = project_root / "restart_jarvis.ps1"
        else:
            restart_script = project_root / "restart_jarvis.sh"

        if not restart_script.exists():
            return ToolResult(
                success=False,
                error=f"Restart script not found at: {restart_script}",
            )

        current_pid = os.getpid()

        return ToolResult(
            success=True,
            data="Rebooting now.",
            signal=ToolSignal.REBOOT,
            metadata={
                "pid": current_pid,
                "restart_script": str(restart_script),
                "project_root": str(project_root),
                "handover_deferred": True,
            },
        )

    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Failed to prepare reboot: {e}",
        )
