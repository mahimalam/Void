"""Event bus.

The orchestrator emits structured events. Subscribers (console in Phase 0,
Electron HUD in Phase 3) listen via an asyncio.Queue.

This is the contract between backend and UI. Backend never imports UI; UI
never imports backend.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.logging_setup import get_logger

# Module-level reference set by main.py after creating the bus.
# security/verification.py reads this to emit AUTH_CHANGE events
# without causing a circular import at module-load time.
_global_bus: "EventBus | None" = None


class EventType(str, Enum):
    WAKE = "wake"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_DONE = "tool_done"
    SPEAKING = "speaking"
    IDLE = "idle"
    ERROR = "error"
    AUTH_CHANGE = "auth_change"
    BRAIN_CHANGE = "brain_change"
    LATENCY_REPORT = "latency_report"
    SYSTEM_STATS = "system_stats"


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """In-memory broadcast queue. One producer (orchestrator), many consumers."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._log = get_logger("event_bus")

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def publish(self, event_type: EventType, **payload: Any) -> None:
        ev = Event(type=event_type, payload=payload)
        for q in self._subscribers:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                self._log.warning("event_dropped", event=event_type.value)


async def console_subscriber(bus: EventBus) -> None:
    """Phase 0 stand-in for the HUD: prints beautiful, clean state changes to the console."""
    import sys
    q = bus.subscribe()


    # Simple ANSI colors for Windows console
    CLR_CYAN = "\033[96m"
    CLR_GREEN = "\033[92m"
    CLR_YELLOW = "\033[93m"
    CLR_RED = "\033[91m"
    CLR_MAGENTA = "\033[95m"
    CLR_RESET = "\033[0m"

    while True:
        try:
            ev = await q.get()
        except asyncio.CancelledError:
            return

        ev_type = ev.type.value

        if ev_type == "idle":
            sys.stdout.write(f"\r{CLR_CYAN}● VOID is Sleeping... Say \"Hey Void\" or \"Void\" to wake up.{CLR_RESET}")
            sys.stdout.flush()
        elif ev_type == "wake":
            sys.stdout.write(f"\n{CLR_YELLOW}⚡ VOID: [Woke Up!]{CLR_RESET}\n")
            sys.stdout.flush()
        elif ev_type == "listening":
            sys.stdout.write(f"\r{CLR_GREEN}🎤 VOID: [Listening...]{CLR_RESET}")
            sys.stdout.flush()
        elif ev_type == "transcribing":
            sys.stdout.write(f"\r{CLR_MAGENTA}✍️ VOID: [Transcribing...]{CLR_RESET}")
            sys.stdout.flush()
        elif ev_type == "thinking":
            sys.stdout.write(f"\r{CLR_MAGENTA}🧠 VOID: [Thinking...]{CLR_RESET}")
            sys.stdout.flush()
        elif ev_type == "speaking":
            text = ev.payload.get("text", "")
            if text:
                sys.stdout.write(f"\n{CLR_GREEN}🎙️ VOID: {text}{CLR_RESET}\n")
            else:
                sys.stdout.write(f"\r{CLR_GREEN}🎙️ VOID: [Speaking...]{CLR_RESET}\n")
            sys.stdout.flush()
        elif ev_type == "error":
            msg = ev.payload.get("spoken", ev.payload.get("error_msg", "An error occurred"))
            sys.stdout.write(f"\n{CLR_RED}❌ VOID: [Error] {msg}{CLR_RESET}\n")
            sys.stdout.flush()
        # M14: the previous implementation dropped TOOL_START,
        # TOOL_DONE, AUTH_CHANGE, BRAIN_CHANGE, and SYSTEM_STATS
        # events on the floor. They're now printed so a tail-mode
        # operator can see tool calls, auth state changes, brain
        # switches, and live CPU/RAM stats without opening the HUD.
        elif ev_type == "tool_start":
            name = ev.payload.get("tool_name", "?")
            args = ev.payload.get("args_preview", "")
            sys.stdout.write(f"\n{CLR_CYAN}→ Tool: {name} ({args}){CLR_RESET}\n")
            sys.stdout.flush()
        elif ev_type == "tool_done":
            name = ev.payload.get("tool_name", "?")
            ok = ev.payload.get("success", False)
            dur = ev.payload.get("duration_ms", 0.0)
            tag = CLR_GREEN if ok else CLR_RED
            mark = "OK" if ok else "FAIL"
            sys.stdout.write(f"{tag}  {name} ({dur:.0f}ms){CLR_RESET}\n")
            sys.stdout.flush()
        elif ev_type == "auth_change":
            state = ev.payload.get("new_state", "?")
            tag = CLR_GREEN if state == "UNLOCKED" else CLR_YELLOW
            sys.stdout.write(f"{tag}🔐 Auth: {state}{CLR_RESET}\n")
            sys.stdout.flush()
        elif ev_type == "brain_change":
            new_brain = ev.payload.get("brain", "?")
            prev = ev.payload.get("prev_brain", "?")
            sys.stdout.write(f"{CLR_MAGENTA}🧠 Brain: {prev} → {new_brain}{CLR_RESET}\n")
            sys.stdout.flush()
        elif ev_type == "system_stats":
            cpu = ev.payload.get("cpu", "?")
            ram = ev.payload.get("ram", "?")
            sys.stdout.write(f"\r{CLR_CYAN}📊 CPU: {cpu}%  RAM: {ram}GB{CLR_RESET}")
            sys.stdout.flush()
        elif ev_type == "latency_report":
            # Silent in HUD but nice to print for debugging
            pass


async def start_websocket_server(bus: EventBus, port: int = 9001) -> None:
    """Starts a WebSocket server that broadcasts all bus events to connected clients (like Electron).

    Returns a ``WebSocketServerHandle`` that exposes a ``stop()`` coroutine
    so the orchestrator can cleanly shut the server down on exit. The
    connected-clients set is protected by an ``asyncio.Lock`` so a
    concurrent add (in ``handler``) and remove (in the finally block)
    cannot raise ``RuntimeError: Set changed size during iteration``
    while ``websockets.broadcast`` is iterating it.
    """
    import websockets

    # M30: track the server + handler task so the caller can stop them.
    # H29: protect the client set with a lock so concurrent add/remove
    # never races with ``websockets.broadcast``.
    connected_clients: set = set()
    clients_lock = asyncio.Lock()

    # State captured for ``stop()``.
    server_obj = None
    broadcast_task: asyncio.Task | None = None

    async def handler(websocket):
        async with clients_lock:
            connected_clients.add(websocket)
        bus._log.info("websocket_client_connected", clients=len(connected_clients))
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    cmd = data.get("command")
                    if cmd == "shutdown":
                        bus._log.info("shutdown_command_received_via_websocket")
                        # Cancel all running tasks to trigger a graceful shutdown
                        # This triggers the finally blocks in core/main.py
                        for task in asyncio.all_tasks():
                            task.cancel()
                    elif cmd == "wake":
                        bus._log.info("wake_command_received_via_websocket")
                        from core.wake_word import WakeWordDetector
                        if WakeWordDetector._active_instance is not None:
                            WakeWordDetector._active_instance.trigger_wake()
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            async with clients_lock:
                connected_clients.discard(websocket)
            bus._log.info("websocket_client_disconnected", clients=len(connected_clients))

    async def broadcast_loop():
        q = bus.subscribe()
        while True:
            try:
                ev = await q.get()
                # H29: snapshot the set under the lock so ``broadcast``
                # iterates a stable view even if a client connects /
                # disconnects mid-iteration.
                async with clients_lock:
                    if not connected_clients:
                        continue
                    clients_snapshot = set(connected_clients)

                message = json.dumps({
                    "type": ev.type.value,
                    "payload": ev.payload
                })

                # Broadcast to the snapshot, NOT the live set.
                websockets.broadcast(clients_snapshot, message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                bus._log.error("websocket_broadcast_error", error=str(e))

    # Start the broadcast loop in the background
    broadcast_task = asyncio.create_task(broadcast_loop())

    # Start the server (non-blocking — returns immediately)
    server_obj = await websockets.serve(handler, "localhost", port)
    bus._log.info("websocket_server_started", port=port)

    class _Handle:
        """Return value from ``start_websocket_server``. Exposes a
        single ``stop()`` coroutine that cancels the broadcast task
        and closes the server socket. Awaiting it from the
        orchestrator's cleanup path gives the connected Electron
        HUD a chance to receive the final events before
        disconnecting."""

        def __init__(self) -> None:
            self._server = server_obj
            self._broadcast_task = broadcast_task

        async def stop(self) -> None:
            bus._log.info("websocket_server_stopping")
            if self._broadcast_task is not None and not self._broadcast_task.done():
                self._broadcast_task.cancel()
                try:
                    await self._broadcast_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self._server is not None:
                self._server.close()
                try:
                    await self._server.wait_closed()
                except Exception as e:
                    # M1: log the close failure so a port-already-in-use
                    # collision or a hung socket close surfaces in the
                    # log instead of vanishing.
                    bus._log.warning("websocket_server_close_failed", error=str(e))
            # Drop any stragglers so the ``bus.subscribe`` queue
            # doesn't keep the event loop alive.
            async with clients_lock:
                connected_clients.clear()
            bus._log.info("websocket_server_stopped")

    return _Handle()
