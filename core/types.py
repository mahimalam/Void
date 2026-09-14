from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class ToolCallRequest:
    """Represents a tool call emitted by an LLM brain."""
    name: str
    arguments: dict[str, Any]
