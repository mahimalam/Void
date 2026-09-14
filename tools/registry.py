"""Tool registry — the heart of Phase 2.

Provides:
  @tool       — decorator that registers async functions as Jarvis tools.
  ToolResult  — frozen dataclass returned by every tool.
  ToolSpec    — metadata captured by @tool (name, schema, category, verify).
  ToolRegistry — discovers, stores, and exports all registered tools.

Usage:
    @tool(name="read_file", verify=False, category="filesystem")
    async def read_file(path: str) -> ToolResult:
        '''Read the contents of a file at the given path.'''
        ...

On startup, ToolRegistry.discover("tools/") walks the directory, imports
every .py file, and collects all @tool-decorated functions automatically.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, get_type_hints

from core.logging_setup import get_logger

# ---------------------------------------------------------------------------
# H5: Tool control-flow signals
# ---------------------------------------------------------------------------
#
# Before H5, tools that needed to influence the orchestrator (e.g. shutdown,
# sleep) smuggled control flow through ``ToolResult.metadata["action"]``.
# That was a hidden control channel: any tool that ever returned a
# ``metadata={"action": ...}`` dict by accident would trigger shutdown.
#
# The fix is a typed ``ToolSignal`` enum returned on ``ToolResult.signal``.
# The orchestrator now switches on ``result.signal``, never on metadata.
# Tools that have no control-flow implication keep ``signal=ToolSignal.NONE``.
# ---------------------------------------------------------------------------

class ToolSignal(str, Enum):
    """Control-flow signal a tool can return to the orchestrator.

    The enum is mixed with ``str`` so it serialises cleanly to JSON
    for the audit log and the WebSocket HUD payload. The orchestrator
    is the only consumer; tools should set the signal explicitly when
    they need to influence the voice loop (e.g. ``jarvis_sleep`` sets
    ``SLEEP`` so the orchestrator skips the follow-up window).
    """

    NONE = "none"
    SHUTDOWN = "shutdown"
    SLEEP = "sleep"
    REBOOT = "reboot"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolResult:
    """Immutable result returned by every tool function.

    Attributes
    ----------
    success : bool
        Whether the tool ran to completion successfully.
    data : str
        Human-readable payload for the LLM to consume.
    error : str
        Error message when ``success`` is False. Empty otherwise.
    metadata : dict
        Pure-data bag for the audit log and HUD. The orchestrator
        does **not** read control-flow values out of metadata — that
        is what ``signal`` is for (H5).
    signal : ToolSignal
        Optional control-flow hint. ``NONE`` means "carry on as
        normal". Tools like ``jarvis_sleep`` set ``SLEEP`` so the
        orchestrator skips the follow-up window after the TTS
        finishes.
    """
    success: bool
    data: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    signal: ToolSignal = ToolSignal.NONE

    def to_dict(self) -> dict[str, Any]:
        """Serialize for LLM consumption and audit logging.

        Empty fields (``data``, ``error``, ``metadata``) are omitted
        to keep the dict compact. ``signal`` follows the same rule:
        ``NONE`` is the default and is omitted; non-default signals
        (SLEEP, SHUTDOWN, REBOOT) are included so the audit log
        records them.
        """
        d: dict[str, Any] = {"success": self.success}
        if self.data:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        if self.signal != ToolSignal.NONE:
            d["signal"] = self.signal.value
        return d


@dataclass(frozen=True)
class ToolSpec:
    """Metadata for a single registered tool."""
    name: str
    func: Callable[..., Any]
    description: str
    parameters: dict[str, Any]   # JSON Schema object
    category: str
    verify: bool

    def to_gemini_declaration(self) -> dict[str, Any]:
        """Convert to Vertex AI FunctionDeclaration dict format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# Global collector — @tool decorated functions land here before registry
# ---------------------------------------------------------------------------

_PENDING_TOOLS: list[ToolSpec] = []


# ---------------------------------------------------------------------------
# Type hint → JSON Schema mapping
# ---------------------------------------------------------------------------

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_json_schema(annotation: Any) -> dict[str, str]:
    """Convert a Python type hint to a minimal JSON Schema type descriptor.

    Handles: str, int, float, bool.  Falls back to "string" for anything
    complex (list, dict, Optional, Union) — keeps the schema simple and
    compatible with both Gemini and Ollama JSON-mode.
    """
    # Unwrap Optional[X] → X (typing.Optional is Union[X, None])
    import typing
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = typing.get_args(annotation)
        if type(None) in args:
            annotation = next((a for a in args if a is not type(None)), annotation)

    # Direct primitive
    json_type = _PYTHON_TYPE_TO_JSON.get(annotation)
    if json_type:
        return {"type": json_type}

    # Fallback — treat everything else as string
    return {"type": "string"}


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Auto-generate a JSON Schema 'object' from a function's type hints.

    Inspects the function signature and type annotations to produce a
    JSON Schema compatible with Gemini FunctionDeclaration and Ollama
    prompt-based tool calling.

    Parameters with defaults are marked optional; those without are required.
    """
    sig = inspect.signature(func)

    # Try get_type_hints first (resolves string annotations from __future__)
    # Fall back to sig.parameters[x].annotation if that fails
    try:
        hints = get_type_hints(func, include_extras=False)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self', 'cls', and return annotation
        if param_name in ("self", "cls"):
            continue

        # Prefer resolved type hints, fall back to annotation from signature
        annotation = hints.get(param_name)
        if annotation is None or annotation is inspect.Parameter.empty:
            annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            annotation = str
        # If annotation is still a string (from __future__ annotations), resolve it
        if isinstance(annotation, str):
            annotation = _resolve_string_annotation(annotation)

        prop_schema = _type_to_json_schema(annotation)

        properties[param_name] = prop_schema

        # Parameters without a default value are required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _resolve_string_annotation(annotation_str: str) -> type:
    """Best-effort resolution of a string annotation to a type object."""
    mapping = {"str": str, "int": int, "float": float, "bool": bool}
    return mapping.get(annotation_str, str)



# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

def tool(
    *,
    name: str,
    verify: bool = False,
    category: str = "general",
) -> Callable[..., Any]:
    """Decorator that registers an async function as a Jarvis tool.

    Usage:
        @tool(name="read_file", verify=False, category="filesystem")
        async def read_file(path: str) -> ToolResult:
            '''Read the contents of a file at the given path.'''
            ...

    The decorator:
      1. Extracts the first line of the docstring as the tool description.
      2. Builds a JSON Schema from the function's type hints.
      3. Appends a ToolSpec to the global _PENDING_TOOLS list.
      4. Returns the original function unchanged.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@tool decorated function '{name}' must be async (use 'async def')"
            )

        # Extract description from docstring (first non-empty line)
        doc = inspect.getdoc(func) or ""
        description = doc.split("\n")[0].strip() if doc else f"Tool: {name}"

        # Build parameter schema from type hints
        parameters = _build_parameters_schema(func)

        spec = ToolSpec(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
            category=category,
            verify=verify,
        )
        _PENDING_TOOLS.append(spec)
        return func

    return decorator


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry of all discovered tools.

    Lifecycle:
      1. Instantiate: ``registry = ToolRegistry()``
      2. Discover:    ``registry.discover(tools_dir)``  — imports tool files
      3. Use:         ``spec = registry.get("read_file")``
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._log = get_logger("tool_registry")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, tools_dir: Path) -> int:
        """Walk *tools_dir*, import every .py file, collect @tool functions.

        Returns the number of tools registered.  Skips __init__.py and
        registry.py to avoid circular imports.
        """
        if not tools_dir.is_dir():
            self._log.warning("tools_dir_not_found", path=str(tools_dir))
            return 0

        skip = {"__init__.py", "registry.py"}
        discovered = 0

        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name in skip:
                continue

            module_name = f"tools.{py_file.stem}"

            # H15: do NOT clear _PENDING_TOOLS here. The old
            # ``_PENDING_TOOLS = []`` assignment was a race magnet:
            # two concurrent discover() calls (or a hot-reload that
            # runs while another discover is mid-flight) could
            # stomp on each other. We now capture a length
            # snapshot *before* importing and consume the new
            # tools from ``_PENDING_TOOLS[before:]`` after the
            # import returns. Any concurrent discover picks up
            # only its own slice.
            before = len(_PENDING_TOOLS)

            try:
                if module_name in sys.modules:
                    # Re-import (useful during tests / hot-reload)
                    importlib.reload(sys.modules[module_name])
                else:
                    spec = importlib.util.spec_from_file_location(
                        module_name, str(py_file)
                    )
                    if spec is None or spec.loader is None:
                        self._log.warning(
                            "tool_file_skip_no_spec", file=py_file.name
                        )
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)

                # H15: collect only the tools this import appended.
                # ``before`` is the length snapshot from before the
                # import. New tools live in ``[before:]``; tools
                # appended by a concurrent discover are in
                # ``[concurrent_before:]`` and are left for that
                # other discover to pick up.
                for tool_spec in _PENDING_TOOLS[before:]:
                    if tool_spec.name in self._tools:
                        self._log.warning(
                            "tool_name_collision",
                            name=tool_spec.name,
                            file=py_file.name,
                        )
                    self._tools[tool_spec.name] = tool_spec
                    discovered += 1
                    self._log.info(
                        "tool_registered",
                        name=tool_spec.name,
                        category=tool_spec.category,
                        verify=tool_spec.verify,
                    )

            except Exception as e:
                self._log.error(
                    "tool_file_import_error",
                    file=py_file.name,
                    error=str(e),
                )

        # H15: do NOT clear _PENDING_TOOLS after discovery. Tools
        # we did not claim (because of a concurrent discover)
        # belong to that other caller. Leaving the list as-is
        # keeps the global state append-only across discoveries,
        # which is race-free.

        self._log.info("tool_discovery_complete", count=discovered)
        return discovered

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        """Look up a tool by its registered name."""
        return self._tools.get(name)

    def all_specs(self) -> list[ToolSpec]:
        """Return all registered tool specs (sorted by name)."""
        return sorted(self._tools.values(), key=lambda s: s.name)

    @property
    def count(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Schema export: Gemini (native function calling)
    # ------------------------------------------------------------------

    def as_gemini_declarations(self, exclude_categories: set[str] | None = None) -> list[dict[str, Any]]:
        """Export all tools as Vertex AI FunctionDeclaration dicts.

        Used by GeminiClient.stream_with_tools() for native function calling.
        """
        exclude = exclude_categories or set()
        return [spec.to_gemini_declaration() for spec in self.all_specs() if spec.category not in exclude]

    # ------------------------------------------------------------------
    # Schema export: Ollama (text-based prompt injection)
    # ------------------------------------------------------------------

    def as_ollama_prompt_schema(self, exclude_categories: set[str] | None = None) -> str:
        """Generate a human-readable tool schema for Ollama system prompt injection.

        The output is a text block listing every tool with its name,
        description, and parameter schema.  The LLM parses this and
        emits JSON tool-call blocks.
        """
        if not self._tools:
            return ""

        exclude = exclude_categories or set()
        lines: list[str] = []
        for spec in self.all_specs():
            if spec.category in exclude:
                continue
            params_desc = _format_params_for_prompt(spec.parameters)
            lines.append(
                f"- {spec.name}: {spec.description}\n"
                f"  Parameters: {params_desc}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_params_for_prompt(schema: dict[str, Any]) -> str:
    """Format a JSON Schema 'object' as a concise parameter description.

    Example output: ``path (string, required), max_bytes (integer, optional)``
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not props:
        return "none"

    parts: list[str] = []
    for pname, pschema in props.items():
        ptype = pschema.get("type", "string")
        req = "required" if pname in required else "optional"
        parts.append(f"{pname} ({ptype}, {req})")

    return ", ".join(parts)
