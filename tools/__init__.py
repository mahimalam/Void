"""J.A.R.V.I.S. tool system (Phase 2).

Each tool is a single file under tools/ decorated with @tool.
On startup the registry auto-discovers every decorated function.

Public API:
    from tools import ToolRegistry, ToolResult, ToolSpec, ToolSignal, tool
"""

from __future__ import annotations

from tools.registry import ToolRegistry, ToolResult, ToolSignal, ToolSpec, tool

__all__ = ["ToolRegistry", "ToolResult", "ToolSignal", "ToolSpec", "tool"]
