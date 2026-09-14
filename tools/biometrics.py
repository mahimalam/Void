"""Biometric tools (voiceprint re-enrollment).

The first-run enrollment is handled by the orchestrator's
``_enrollment_state`` at startup. This module provides the
re-enrollment tool that an unlocked user can invoke later
(e.g. after a cold, after a long break) without restarting JARVIS.
"""

from __future__ import annotations

import asyncio

from tools.registry import ToolResult, tool


@tool(name="re_enroll_voiceprint", verify=True, category="biometrics")
async def re_enroll_voiceprint() -> ToolResult:
    """Re-record and replace the master voiceprint.

    Requires the system to be UNLOCKED (verification.enabled: true and
    the master password supplied within the timeout) because
    replacing the voiceprint is a security-sensitive operation. After
    this completes, the next voice command will be checked against
    the new embedding.

    C8 fix: previously there was no way to re-enroll without deleting
    ``data/voiceprint.npy`` and restarting. Now this tool is always
    available to an unlocked user.
    """
    # Lazy import to keep the tool module import-cheap; the orchestrator
    # wires up its audio_input and vad only at startup.
    try:
        from core.orchestrator import VoiceOrchestrator
    except Exception as e:
        return ToolResult(success=False, error=f"Orchestrator unavailable: {e}")

    singleton = getattr(VoiceOrchestrator, "_active_instance", None)
    if singleton is None:
        return ToolResult(
            success=False,
            error="Re-enrollment requires the running orchestrator. "
            "Restart JARVIS or run scripts/enroll.py to set the "
            "voiceprint for the first time.",
        )

    # Confirm we have the helpers we need
    if not hasattr(singleton, "_collect_enrollment_audio"):
        return ToolResult(
            success=False,
            error="Orchestrator does not expose an enrollment-audio "
            "helper; update the orchestrator first.",
        )

    from security.biometrics import enroll_voiceprint_async

    # Speak the prompt and wait for the user to start speaking
    try:
        await singleton._tts.speak_sync(
            "I will record a new master voiceprint. Please state your "
            "name and speak continuously for ten seconds."
        )
        from core.event_bus import EventType

        await singleton._post_speak_settle()
        singleton._bus.publish(EventType.LISTENING)

        full_audio = await singleton._collect_enrollment_audio()

        singleton._bus.publish(EventType.THINKING)
        await enroll_voiceprint_async(full_audio)

        singleton._bus.publish(EventType.SPEAKING)
        await singleton._tts.speak_sync("Voiceprint updated. The next voice command will use it.")

        from security.biometrics import has_voiceprint
        return ToolResult(
            success=True,
            data=(
                "Master voiceprint re-enrolled."
                if has_voiceprint()
                else "Voiceprint re-enrollment finished but the file is missing — check logs."
            ),
        )
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            error="Re-enrollment timed out collecting audio.",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Re-enrollment failed: {e}",
        )
