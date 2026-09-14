"""Clipboard tools — read and write system clipboard.

Uses pyperclip for cross-platform clipboard access.
"""

from __future__ import annotations

from tools.registry import ToolResult, tool


def _get_pyperclip():
    """Lazy-import pyperclip to avoid import errors."""
    try:
        import pyperclip
        return pyperclip
    except ImportError:
        return None


@tool(name="read_clipboard", verify=False, category="clipboard")
async def read_clipboard() -> ToolResult:
    """Read the current text from the system clipboard."""
    pyperclip = _get_pyperclip()
    if not pyperclip:
        return ToolResult(success=False, error="pyperclip is not installed. Run: pip install pyperclip")

    try:
        content = pyperclip.paste()
        if not content:
            return ToolResult(success=True, data="Clipboard is empty.")

        # Truncate very long clipboard content
        if len(content) > 5000:
            truncated = content[:5000] + f"\n\n... [truncated, {len(content)} chars total]"
            return ToolResult(
                success=True,
                data=truncated,
                metadata={"chars": len(content), "truncated": True},
            )

        return ToolResult(
            success=True,
            data=content,
            metadata={"chars": len(content)},
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to read clipboard: {e}")


@tool(name="write_clipboard", verify=False, category="clipboard")
async def write_clipboard(text: str) -> ToolResult:
    """Write text to the system clipboard."""
    pyperclip = _get_pyperclip()
    if not pyperclip:
        return ToolResult(success=False, error="pyperclip is not installed. Run: pip install pyperclip")

    if not text:
        return ToolResult(success=False, error="Cannot write empty text to clipboard.")

    try:
        pyperclip.copy(text)
        return ToolResult(
            success=True,
            data=f"Copied {len(text)} characters to clipboard.",
            metadata={"chars": len(text)},
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to write clipboard: {e}")
