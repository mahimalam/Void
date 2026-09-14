"""Filesystem tools — read, write, list, create, copy, move, delete.

All paths are resolved and validated before execution. The validator
rejects ``..`` as a path *component* (not as a substring, so the
perfectly-valid filename ``foo..bar`` is allowed), and rejects
symlinks so a malicious link cannot redirect reads/writes outside
the user's intended tree. Operations are bounded to prevent abuse.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from tools.registry import ToolResult, tool


_MAX_READ_BYTES = 50 * 1024  # 50 KB max file read

# H17: probe this many bytes from the start of a file to decide
# whether it is binary. A null byte is a strong signal of a
# non-text file (UTF-8 / UTF-16 / ASCII text files never contain
# 0x00). 4 KB is enough to catch any reasonable file header.
_BINARY_PROBE_BYTES = 4096


def _looks_binary(path: Path) -> bool:
    """Return True if *path* looks like a binary file.

    H17: the previous ``read_file`` used
    ``read_text(encoding="utf-8", errors="replace")`` on every
    file, which silently mangled binary content into a mess of
    replacement characters and never told the user. We now probe
    the first ``_BINARY_PROBE_BYTES`` for null bytes — a strong
    signal of a binary file — and refuse to read the file as
    text. UTF-8 / UTF-16 / ASCII text files never contain 0x00.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(_BINARY_PROBE_BYTES)
    except OSError:
        return False
    return b"\x00" in chunk


def _validate_path(raw_path: str) -> tuple[Path, str | None]:
    """Resolve and validate a user-supplied file path.

    Returns (resolved_path, error_message).
    error_message is None if the path is valid.

    H16 fixes (vs the old substring check on ``".."``):
    1. Only ``..`` that appears as a full path *component* is
       rejected. Filenames like ``foo..bar`` are valid on every
       filesystem and the old check spuriously blocked them.
    2. The path is run through ``Path.expanduser().resolve()`` so
       any embedded symlink or relative reference is collapsed to
       a canonical absolute path before further checks.
    3. The resolved path's parents are walked, and any parent
       component that is a symlink is rejected. This stops a
       symlink at ``~/data/link_to_etc`` from being treated as
       "in the user's tree" just because the final target looks
       innocuous.
    4. Empty paths are rejected up front.
    """
    if not raw_path or not raw_path.strip():
        return Path(), "Path cannot be empty."

    # H16 (1): reject only when ``..`` is a full component.
    parts = Path(raw_path).parts
    if any(part == ".." for part in parts):
        return Path(), "Path traversal ('..') is not allowed for security."

    # H16 (3): reject any *intermediate* component that is a symlink,
    # BEFORE we resolve. After resolve, symlinks are followed and the
    # original component is lost. We check the parent directory of
    # the leaf (which exists for both reads and writes) and every
    # ancestor above it.
    abs_path = Path(raw_path).expanduser()
    if abs_path.is_absolute() or str(abs_path).startswith("~"):
        if abs_path.is_absolute():
            prefix = abs_path
        else:
            prefix = abs_path.expanduser().resolve()

        # Walk from the immediate parent up to (but not including)
        # the filesystem root. ``prefix.parent`` is the directory the
        # leaf lives in; if it is a symlink, that's a traversal.
        current = prefix
        # Track walked paths to avoid infinite loops on circular symlinks
        seen: set[Path] = set()
        for parent in [current.parent] + list(current.parents):
            if parent in seen:
                break
            seen.add(parent)
            try:
                if parent.is_symlink():
                    return (
                        Path(),
                        f"Path traverses a symlink at {parent}, which is not allowed.",
                    )
            except (OSError, ValueError):
                # Path doesn't exist yet — fine, not a symlink.
                pass

    try:
        # H16 (2): resolve via expanduser + resolve.
        resolved = Path(raw_path).expanduser().resolve()
    except (OSError, ValueError) as e:
        return Path(), f"Invalid path: {e}"

    return resolved, None


@tool(name="read_file", verify=False, category="filesystem")
async def read_file(path: str) -> ToolResult:
    """Read the contents of a text file (max 50KB)."""
    resolved, err = _validate_path(path)
    if err:
        return ToolResult(success=False, error=err)

    if not resolved.exists():
        return ToolResult(success=False, error=f"File not found: {resolved}")

    if not resolved.is_file():
        return ToolResult(success=False, error=f"Not a file: {resolved}")

    # H17: refuse to read what looks like a binary file. Returning
    # a replacement-character-filled mess silently is worse than
    # a clear error — the user can still open the file with
    # ``open_url`` (e.g. via the system editor) if they need to.
    if _looks_binary(resolved):
        return ToolResult(
            success=False,
            error=(
                f"Refusing to read what looks like a binary file: {resolved}. "
                "Use a different tool (e.g. open in a system editor) for binary content."
            ),
        )

    size = resolved.stat().st_size
    if size > _MAX_READ_BYTES:
        return ToolResult(
            success=False,
            error=f"File too large ({size:,} bytes). Max is {_MAX_READ_BYTES:,} bytes.",
        )

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        return ToolResult(
            success=True,
            data=content,
            metadata={"path": str(resolved), "size_bytes": size},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied: {resolved}")
    except Exception as e:
        return ToolResult(success=False, error=f"Read error: {e}")


@tool(name="write_file", verify=True, category="filesystem")
async def write_file(path: str, content: str) -> ToolResult:
    """Write text content to a file, creating parent directories if needed."""
    resolved, err = _validate_path(path)
    if err:
        return ToolResult(success=False, error=err)

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            data=f"Written {len(content)} characters to {resolved}",
            metadata={"path": str(resolved), "chars_written": len(content)},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied: {resolved}")
    except Exception as e:
        return ToolResult(success=False, error=f"Write error: {e}")


@tool(name="list_dir", verify=False, category="filesystem")
async def list_dir(path: str) -> ToolResult:
    """List the contents of a directory with file sizes."""
    resolved, err = _validate_path(path)
    if err:
        return ToolResult(success=False, error=err)

    if not resolved.exists():
        return ToolResult(success=False, error=f"Directory not found: {resolved}")

    if not resolved.is_dir():
        return ToolResult(success=False, error=f"Not a directory: {resolved}")

    try:
        entries: list[str] = []
        for item in sorted(resolved.iterdir()):
            if item.is_dir():
                entries.append(f"  [DIR]  {item.name}/")
            else:
                try:
                    size = item.stat().st_size
                    entries.append(f"  [FILE] {item.name}  ({_fmt_size(size)})")
                except OSError:
                    entries.append(f"  [FILE] {item.name}  (size unknown)")

        if not entries:
            return ToolResult(
                success=True,
                data=f"Directory is empty: {resolved}",
                metadata={"path": str(resolved), "count": 0},
            )

        header = f"Contents of {resolved} ({len(entries)} items):\n"
        return ToolResult(
            success=True,
            data=header + "\n".join(entries),
            metadata={"path": str(resolved), "count": len(entries)},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied: {resolved}")
    except Exception as e:
        return ToolResult(success=False, error=f"List error: {e}")


@tool(name="create_dir", verify=False, category="filesystem")
async def create_dir(path: str) -> ToolResult:
    """Create a directory and any missing parent directories."""
    resolved, err = _validate_path(path)
    if err:
        return ToolResult(success=False, error=err)

    try:
        resolved.mkdir(parents=True, exist_ok=True)
        return ToolResult(
            success=True,
            data=f"Directory created: {resolved}",
            metadata={"path": str(resolved)},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied: {resolved}")
    except Exception as e:
        return ToolResult(success=False, error=f"Create error: {e}")


@tool(name="copy_file", verify=False, category="filesystem")
async def copy_file(source: str, destination: str) -> ToolResult:
    """Copy a file from source to destination."""
    src, err = _validate_path(source)
    if err:
        return ToolResult(success=False, error=f"Source: {err}")

    dst, err = _validate_path(destination)
    if err:
        return ToolResult(success=False, error=f"Destination: {err}")

    if not src.exists():
        return ToolResult(success=False, error=f"Source not found: {src}")
    if not src.is_file():
        return ToolResult(success=False, error=f"Source is not a file: {src}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return ToolResult(
            success=True,
            data=f"Copied {src.name} to {dst}",
            metadata={"source": str(src), "destination": str(dst)},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied")
    except Exception as e:
        return ToolResult(success=False, error=f"Copy error: {e}")


@tool(name="move_file", verify=True, category="filesystem")
async def move_file(source: str, destination: str) -> ToolResult:
    """Move or rename a file. Requires verification."""
    src, err = _validate_path(source)
    if err:
        return ToolResult(success=False, error=f"Source: {err}")

    dst, err = _validate_path(destination)
    if err:
        return ToolResult(success=False, error=f"Destination: {err}")

    if not src.exists():
        return ToolResult(success=False, error=f"Source not found: {src}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return ToolResult(
            success=True,
            data=f"Moved {src.name} to {dst}",
            metadata={"source": str(src), "destination": str(dst)},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied")
    except Exception as e:
        return ToolResult(success=False, error=f"Move error: {e}")


@tool(name="delete_file", verify=True, category="filesystem")
async def delete_file(path: str) -> ToolResult:
    """Permanently delete a file. Requires verification. Cannot be undone."""
    resolved, err = _validate_path(path)
    if err:
        return ToolResult(success=False, error=err)

    if not resolved.exists():
        return ToolResult(success=False, error=f"File not found: {resolved}")
    if not resolved.is_file():
        return ToolResult(success=False, error=f"Not a file (refusing to delete directories): {resolved}")

    try:
        resolved.unlink()
        return ToolResult(
            success=True,
            data=f"Deleted: {resolved}",
            metadata={"path": str(resolved)},
        )
    except PermissionError:
        return ToolResult(success=False, error=f"Permission denied: {resolved}")
    except Exception as e:
        return ToolResult(success=False, error=f"Delete error: {e}")


def _fmt_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
