import asyncio
import time
import subprocess
from pathlib import Path
from core.config import load_config
from tools.registry import tool, ToolResult

_MAX_SCREENSHOTS_KEPT = 10

def _prune_old_files(directory: Path, prefix: str, keep: int) -> int:
    """Keep only the newest screenshots to prevent unbounded storage growth."""
    if not directory.is_dir():
        return 0
    try:
        candidates = sorted(
            (p for p in directory.iterdir() if p.is_file() and p.name.startswith(prefix)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return 0
    deleted = 0
    for stale in candidates[keep:]:
        try:
            stale.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted

@tool(name="capture_screen", verify=False, category="system")
async def capture_screen() -> ToolResult:
    """Takes a screenshot of the user's monitors. ALWAYS use this tool when the user asks you to look at their screen or read their code.
    
    Privacy note: screenshots may contain sensitive content.
    """
    cfg = load_config()
    sandbox = cfg.resolve(cfg.tools.sandbox_dir)
    sandbox.mkdir(parents=True, exist_ok=True)

    pruned = _prune_old_files(sandbox, "screenshot_", _MAX_SCREENSHOTS_KEPT)

    filename = f"screenshot_{time.time():.3f}.jpg"
    filepath = sandbox / filename

    try:
        # Use native KDE Spectacle for ultra-fast, Wayland-compatible screen capture
        # -b: background (no GUI)
        # -n: nonotify (no popup)
        # -o: output file
        result = await asyncio.to_thread(
            subprocess.run,
            ["spectacle", "-b", "-n", "-o", str(filepath)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return ToolResult(
                success=False, 
                error=f"Failed to capture screen with Spectacle: {result.stderr}"
            )

        if not filepath.exists():
            return ToolResult(success=False, error="Spectacle completed but file was not created.")

        return ToolResult(
            success=True,
            data=str(filepath),
            metadata={
                "path": str(filepath),
                "pruned_old_files": pruned,
                "retained_max": _MAX_SCREENSHOTS_KEPT,
                "privacy_note": "May contain sensitive content; user is responsible.",
            },
        )
    except FileNotFoundError:
        return ToolResult(
            success=False, 
            error="Spectacle is not installed. Native KDE Wayland capture failed."
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Error capturing screen: {str(e)}")
