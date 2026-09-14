"""System information tools — CPU, RAM, disk, and general system info.

Uses psutil for cross-platform system metrics and platform for OS info.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from tools.registry import ToolResult, ToolSignal, tool


def _get_psutil():
    """Lazy-import psutil to avoid import error if not installed."""
    try:
        import psutil
        return psutil
    except ImportError:
        return None


@tool(name="get_system_info", verify=False, category="system")
async def get_system_info() -> ToolResult:
    """Get general system information: OS, CPU, RAM, Python version."""
    psutil = _get_psutil()

    info_lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Hostname: {platform.node()}",
        f"Python: {platform.python_version()}",
        f"CPU: {platform.processor() or 'Unknown'}",
    ]

    if psutil:
        info_lines.append(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        mem = psutil.virtual_memory()
        info_lines.append(f"RAM: {mem.total / (1024**3):.1f} GB total")

    # Try to get GPU info
    try:
        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 5}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW on Windows
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            **kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_lines = result.stdout.strip().split("\n")
            for i, line in enumerate(gpu_lines):
                parts = line.strip().split(", ")
                if len(parts) == 2:
                    info_lines.append(f"GPU {i}: {parts[0]} ({parts[1]} MB)")
    except Exception:
        pass  # nvidia-smi not available

    return ToolResult(
        success=True,
        data="\n".join(info_lines),
    )


@tool(name="get_cpu_usage", verify=False, category="system")
async def get_cpu_usage() -> ToolResult:
    """Get current CPU usage percentage per core and overall."""
    psutil = _get_psutil()
    if not psutil:
        return ToolResult(success=False, error="psutil is not installed.")

    overall = psutil.cpu_percent(interval=0.5)
    per_core = psutil.cpu_percent(interval=0, percpu=True)

    lines = [f"Overall CPU usage: {overall}%"]
    for i, pct in enumerate(per_core):
        lines.append(f"  Core {i}: {pct}%")

    return ToolResult(
        success=True,
        data="\n".join(lines),
        metadata={"overall_percent": overall, "per_core": per_core},
    )


@tool(name="get_ram_usage", verify=False, category="system")
async def get_ram_usage() -> ToolResult:
    """Get current RAM usage in GB and percentage."""
    psutil = _get_psutil()
    if not psutil:
        return ToolResult(success=False, error="psutil is not installed.")

    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)

    data = (
        f"RAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total ({mem.percent}%)\n"
        f"Available: {available_gb:.1f} GB"
    )

    return ToolResult(
        success=True,
        data=data,
        metadata={
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "percent": mem.percent,
        },
    )


@tool(name="get_disk_usage", verify=False, category="system")
async def get_disk_usage() -> ToolResult:
    """Get disk usage percentage for all mounted partitions."""
    psutil = _get_psutil()
    if not psutil:
        return ToolResult(success=False, error="psutil is not installed.")

    lines: list[str] = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            lines.append(
                f"{part.device} ({part.mountpoint}): "
                f"{used_gb:.1f} GB used / {total_gb:.1f} GB total "
                f"({usage.percent}%, {free_gb:.1f} GB free)"
            )
        except (PermissionError, OSError):
            lines.append(f"{part.device} ({part.mountpoint}): access denied")

    return ToolResult(
        success=True,
        data="\n".join(lines) if lines else "No partitions found.",
    )


@tool(name="jarvis_sleep", verify=False, category="system")
async def jarvis_sleep() -> ToolResult:
    """Put Jarvis into standby mode until the wake word is spoken again."""
    return ToolResult(
        success=True,
        data="Entering standby mode.",
        signal=ToolSignal.SLEEP,
    )
