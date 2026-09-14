"""Code execution tools — run_python, run_shell.

Both tools execute in a sandboxed subprocess with:
  - Hard timeout (configurable, default 30s)
  - Restricted working directory (data/sandbox/)
  - CREATE_NO_WINDOW flag on Windows
  - stdout + stderr captured, truncated to 4KB
  - Both require verification (verify=True)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from security.sandbox import run_sandboxed

from tools.registry import ToolResult, tool


_MAX_OUTPUT_BYTES = 4096  # 4 KB output cap
_DEFAULT_TIMEOUT = 30     # seconds


def _get_sandbox_dir() -> Path:
    """Get or create the sandbox working directory."""
    # Walk up from this file to find project root, then data/sandbox
    project_root = Path(__file__).resolve().parent.parent
    sandbox = project_root / "data" / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _truncate_output(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate output to max_bytes, appending a truncation notice."""
    if len(text.encode("utf-8", errors="replace")) <= max_bytes:
        return text
    # Truncate by characters (approximate)
    truncated = text[:max_bytes]
    return truncated + f"\n\n... [output truncated at {max_bytes} bytes]"


def _safe_env() -> dict[str, str]:
    """Build a minimal environment for sandboxed execution.

    Inherits PATH and essential vars but strips sensitive ones.
    """
    keep_keys = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "HOME", "USER", "USERPROFILE"}
    env = {k: v for k, v in os.environ.items() if k.upper() in keep_keys}
    # Block network access hint (not enforced at OS level without firewall rules)
    env["no_proxy"] = "*"
    return env


@tool(name="run_python", verify=True, category="code")
async def run_python(code: str) -> ToolResult:
    """Execute a Python code snippet in a sandboxed subprocess."""
    sandbox = _get_sandbox_dir()

    # Write code to a temporary file in the sandbox
    tmp_file = sandbox / "_jarvis_exec.py"
    try:
        tmp_file.write_text(code, encoding="utf-8")
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to write temp file: {e}")

    try:
        command = [sys.executable, str(tmp_file)]
        
        success, stdout, stderr = await asyncio.to_thread(
            run_sandboxed,
            command=command,
            cwd=str(sandbox),
            timeout_secs=_DEFAULT_TIMEOUT,
            memory_limit_mb=512
        )

        stdout = _truncate_output(stdout)
        stderr = _truncate_output(stderr)

        if success:
            output = stdout if stdout.strip() else "(no output)"
            return ToolResult(
                success=True,
                data=output,
                metadata={"return_code": 0},
            )
        else:
            error_msg = stderr if stderr.strip() else "Process failed or exited with non-zero code."
            return ToolResult(
                success=False,
                data=stdout if stdout.strip() else "",
                error=error_msg,
                metadata={"return_code": 1},
            )

    except Exception as e:
        return ToolResult(success=False, error=f"Execution failed: {e}")
    finally:
        # Clean up temp file
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass


@tool(name="run_shell", verify=True, category="code")
async def run_shell(command: str) -> ToolResult:
    """Execute a shell command in a sandboxed subprocess."""
    sandbox = _get_sandbox_dir()

    try:
        if sys.platform == "win32":
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["/bin/sh", "-c", command]

        success, stdout, stderr = await asyncio.to_thread(
            run_sandboxed,
            command=shell_cmd,
            cwd=str(sandbox),
            timeout_secs=_DEFAULT_TIMEOUT,
            memory_limit_mb=512
        )

        stdout = _truncate_output(stdout)
        stderr = _truncate_output(stderr)

        if success:
            output = stdout if stdout.strip() else "(no output)"
            return ToolResult(
                success=True,
                data=output,
                metadata={"return_code": 0},
            )
        else:
            error_msg = stderr if stderr.strip() else "Command failed or exited with non-zero code."
            return ToolResult(
                success=False,
                data=stdout if stdout.strip() else "",
                error=error_msg,
                metadata={"return_code": 1},
            )

    except Exception as e:
        return ToolResult(success=False, error=f"Command execution failed: {e}")
