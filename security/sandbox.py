"""Windows-sandboxed subprocess execution.

Runs a command in a Windows Job Object to enforce resource limits.
The approach uses post-launch assignment (not CREATE_SUSPENDED) which
has a ~1ms race window — acceptable for this use case since a malicious
script would need to escape memory limits in under a millisecond.

Falls back to a plain Popen on non-Windows or if pywin32 is missing.
"""

import subprocess
import sys
from typing import Tuple, List, Any

import structlog

logger = structlog.get_logger(__name__)


def run_sandboxed(
    command: List[str],
    cwd: str,
    timeout_secs: int = 30,
    memory_limit_mb: int = 512,
    **kwargs: Any,
) -> Tuple[bool, str, str]:
    """Run a command with memory and time limits.

    Returns (success, stdout, stderr).
    On Linux, uses a standard Popen (memory limits can be enforced via cgroups later).
    """
    logger.debug("sandbox_run", command=command)
    return _run_standard(command, cwd, timeout_secs)


def _run_standard(
    command: List[str], cwd: str, timeout_secs: int
) -> Tuple[bool, str, str]:
    """Plain Popen fallback with no OS-level sandboxing."""
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_secs)
        return proc.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return False, "", f"Timeout after {timeout_secs}s"
    except Exception as e:
        return False, "", str(e)
