"""Notes tools — save, retrieve, and list personal notes.

Notes are stored as individual .md files in data/notes/ with YAML
frontmatter for metadata (title, timestamp). This flat-file approach
keeps notes human-readable and prepares them for Phase 4 memory
integration.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from tools.registry import ToolResult, tool


def _get_notes_dir() -> Path:
    """Get or create the notes directory."""
    project_root = Path(__file__).resolve().parent.parent
    notes_dir = project_root / "data" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    return notes_dir


def _slugify(title: str) -> str:
    """Convert a note title to a safe filename slug.

    Example: "Grocery List" -> "grocery_list"
    """
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)       # Remove non-alphanum
    slug = re.sub(r"[\s-]+", "_", slug)         # Spaces/hyphens to underscores
    slug = slug.strip("_")
    return slug[:80] or "untitled"              # Cap length


@tool(name="save_note", verify=False, category="notes")
async def save_note(title: str, content: str) -> ToolResult:
    """Save a note with a title. Creates or overwrites the note."""
    if not title.strip():
        return ToolResult(success=False, error="Note title cannot be empty.")
    if not content.strip():
        return ToolResult(success=False, error="Note content cannot be empty.")

    notes_dir = _get_notes_dir()
    slug = _slugify(title)
    note_path = notes_dir / f"{slug}.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # YAML frontmatter + content
    note_text = (
        f"---\n"
        f"title: \"{title.strip()}\"\n"
        f"created: {now}\n"
        f"---\n\n"
        f"{content.strip()}\n"
    )

    try:
        note_path.write_text(note_text, encoding="utf-8")
        return ToolResult(
            success=True,
            data=f"Note \"{title.strip()}\" saved.",
            metadata={"path": str(note_path), "slug": slug},
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to save note: {e}")


@tool(name="get_note", verify=False, category="notes")
async def get_note(title: str) -> ToolResult:
    """Retrieve a note by its title."""
    if not title.strip():
        return ToolResult(success=False, error="Note title cannot be empty.")

    notes_dir = _get_notes_dir()
    slug = _slugify(title)
    note_path = notes_dir / f"{slug}.md"

    if not note_path.exists():
        # Try fuzzy match — find notes with similar slugs
        available = [f.stem for f in notes_dir.glob("*.md")]
        suggestion = ""
        if available:
            suggestion = f" Available notes: {', '.join(available[:5])}"
        return ToolResult(
            success=False,
            error=f"Note \"{title.strip()}\" not found.{suggestion}",
        )

    try:
        raw = note_path.read_text(encoding="utf-8")

        # Strip YAML frontmatter for clean reading
        content = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()

        return ToolResult(
            success=True,
            data=content,
            metadata={"title": title.strip(), "path": str(note_path)},
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to read note: {e}")


@tool(name="list_notes", verify=False, category="notes")
async def list_notes() -> ToolResult:
    """List all saved notes with their titles."""
    notes_dir = _get_notes_dir()

    note_files = sorted(notes_dir.glob("*.md"))

    if not note_files:
        return ToolResult(success=True, data="No notes saved yet.")

    entries: list[str] = []
    for nf in note_files:
        try:
            raw = nf.read_text(encoding="utf-8", errors="replace")
            # Extract title from frontmatter
            title = nf.stem.replace("_", " ").title()
            if raw.startswith("---"):
                for line in raw.split("\n"):
                    if line.startswith("title:"):
                        title = line.split(":", 1)[1].strip().strip('"')
                        break
            size = nf.stat().st_size
            entries.append(f"  • {title}  ({size} bytes)")
        except Exception:
            entries.append(f"  • {nf.stem}  (unreadable)")

    return ToolResult(
        success=True,
        data=f"Notes ({len(entries)}):\n" + "\n".join(entries),
        metadata={"count": len(entries)},
    )
