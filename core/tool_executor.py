"""Tool execution engine.

Bridges the gap between the LLM's tool-call intent and actual tool
execution.  Handles:
  - Tool lookup in the registry
  - Verification gating (Phase 2: verify=True tools are blocked)
  - Async timeout enforcement
  - Event bus notifications (TOOL_START / TOOL_DONE)
  - Audit log writing

Usage:
    executor = ToolExecutor(registry, audit, bus, cfg)
    result = await executor.execute("read_file", {"path": "/tmp/x"})
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from core.audit import AuditLog
from core.config import ToolsConfig
from core.errors import ToolExecutionError
from core.event_bus import EventBus, EventType
from core.logging_setup import get_logger
from tools.registry import ToolRegistry, ToolResult


class ToolExecutor:
    """Executes tools by name with full lifecycle management.

    Parameters
    ----------
    registry : ToolRegistry
        The populated tool registry.
    audit : AuditLog
        SQLite audit log writer.
    bus : EventBus
        For TOOL_START / TOOL_DONE events.
    cfg : ToolsConfig
        Execution timeout and other settings.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditLog,
        bus: EventBus,
        cfg: ToolsConfig,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._bus = bus
        self._cfg = cfg
        self._log = get_logger("tool_executor")

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name.

        Parameters
        ----------
        tool_name : str
            The registered tool name (e.g. "read_file").
        arguments : dict
            Keyword arguments to pass to the tool function.

        Returns
        -------
        ToolResult
            The result of the tool execution, whether success or failure.
            Never raises — all errors are captured as ToolResult(success=False).
        """
        # 1. Lookup
        spec = self._registry.get(tool_name)
        if spec is None:
            self._log.warning("tool_not_found", name=tool_name)
            await self._audit.log(
                tool_name=tool_name,
                arguments=arguments,
                verification_state="N/A",
                success=False,
                error="Tool not found",
                duration_ms=0.0,
            )
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        # 2. Verification gate (Phase 3)
        if spec.verify:
            from security.verification import verification_manager, AuthState
            state = await verification_manager.get_state()
            state_str = state.value
            
            if state == AuthState.LOCKED:
                self._log.warning(
                    "tool_blocked_verification_required",
                    name=tool_name,
                )
                await self._audit.log(
                    tool_name=tool_name,
                    arguments=arguments,
                    verification_state=state_str,
                    success=False,
                    error="Verification required (Locked)",
                    duration_ms=0.0,
                )
                self._bus.publish(
                    EventType.TOOL_DONE,
                    tool_name=tool_name,
                    success=False,
                    duration_ms=0.0,
                )
                return ToolResult(
                    success=False,
                    error="That action requires security verification. Please unlock J.A.R.V.I.S. with your password.",
                )
        else:
            state_str = "N/A"

        # 3. Execute with timeout
        self._bus.publish(
            EventType.TOOL_START,
            tool_name=tool_name,
            args_preview=_preview_args(arguments),
        )
        self._log.info(
            "tool_executing",
            name=tool_name,
            args=_preview_args(arguments),
        )

        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                spec.func(**arguments),
                timeout=float(self._cfg.execution_timeout_s),
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log.error("tool_timeout", name=tool_name, timeout_s=self._cfg.execution_timeout_s)
            await self._audit.log(
                tool_name=tool_name,
                arguments=arguments,
                verification_state=state_str,
                success=False,
                error=f"Timed out after {self._cfg.execution_timeout_s}s",
                duration_ms=duration_ms,
            )
            self._bus.publish(
                EventType.TOOL_DONE,
                tool_name=tool_name,
                success=False,
                duration_ms=round(duration_ms, 1),
            )
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' timed out after {self._cfg.execution_timeout_s} seconds.",
            )
        except TypeError as e:
            # Wrong arguments passed to tool function
            duration_ms = (time.perf_counter() - start) * 1000
            error_msg = f"Invalid arguments for '{tool_name}': {e}"
            self._log.error("tool_argument_error", name=tool_name, error=str(e))
            await self._audit.log(
                tool_name=tool_name,
                arguments=arguments,
                verification_state=state_str,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
            self._bus.publish(
                EventType.TOOL_DONE,
                tool_name=tool_name,
                success=False,
                duration_ms=round(duration_ms, 1),
            )
            return ToolResult(success=False, error=error_msg)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            error_msg = f"Tool '{tool_name}' crashed: {e}"
            self._log.error("tool_crash", name=tool_name, error=str(e))
            await self._audit.log(
                tool_name=tool_name,
                arguments=arguments,
                verification_state=state_str,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            self._bus.publish(
                EventType.TOOL_DONE,
                tool_name=tool_name,
                success=False,
                duration_ms=round(duration_ms, 1),
            )
            return ToolResult(success=False, error=error_msg)

        # 4. Success path
        duration_ms = (time.perf_counter() - start) * 1000

        self._log.info(
            "tool_completed",
            name=tool_name,
            success=result.success,
            duration_ms=round(duration_ms, 1),
        )

        await self._audit.log(
            tool_name=tool_name,
            arguments=arguments,
            verification_state=state_str,
            success=result.success,
            error=result.error,
            duration_ms=duration_ms,
        )

        self._bus.publish(
            EventType.TOOL_DONE,
            tool_name=tool_name,
            success=result.success,
            duration_ms=round(duration_ms, 1),
        )

        return result


def _preview_args(args: dict[str, Any], max_len: int = 100) -> str:
    """Create a short preview of tool arguments for logging."""
    if not args:
        return "{}"
    parts: list[str] = []
    for k, v in args.items():
        val_str = str(v)
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        parts.append(f"{k}={val_str}")
    preview = ", ".join(parts)
    if len(preview) > max_len:
        preview = preview[:max_len - 3] + "..."
    return preview
