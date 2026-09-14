"""App launcher and URL opener tools for Linux.

Uses xdg-open and standard subprocess calls.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.parse
import webbrowser

import httpx

from tools.registry import ToolResult, tool

APP_ALIASES: dict[str, list[str]] = {
    "google chrome": ["google-chrome-stable", "google-chrome", "com.google.Chrome", "chromium", "chromium-browser"],
    "chrome": ["google-chrome-stable", "google-chrome", "com.google.Chrome", "chromium"],
    "chromium": ["chromium", "chromium-browser", "google-chrome-stable", "google-chrome"],
    "firefox": ["firefox", "org.mozilla.firefox"],
    "code": ["code", "codium"],
    "vscode": ["code", "codium"],
    "vs code": ["code", "codium"],
    "terminal": ["gnome-terminal", "x-terminal-emulator", "org.gnome.Ptyxis", "konsole", "alacritty", "kitty"],
    "calculator": ["gnome-calculator", "kcalc", "xcalc"],
    "telegram": ["telegram-desktop", "telegram"],
    "whatsapp": ["whatsapp-for-linux", "whatsapp"],
    "spotify": ["spotify"],
    "vlc": ["vlc"],
    "files": ["nautilus", "thunar", "dolphin", "pcmanfm"],
}


def _find_desktop_entry(query: str) -> str | None:
    """Find a .desktop file whose Name matches query."""
    import glob
    q = query.lower().strip()
    search_dirs = [
        "/usr/share/applications",
        "/usr/local/share/applications",
        os.path.expanduser("~/.local/share/applications"),
    ]
    for d in search_dirs:
        for path in glob.glob(f"{d}/*.desktop"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("Name="):
                            name_val = line[5:].strip().lower()
                            if q == name_val or q in name_val:
                                desktop_id = os.path.basename(path)[:-8]  # strip .desktop
                                return desktop_id
            except Exception:
                continue
    return None


@tool(name="launch_app", verify=False, category="apps")
async def launch_app(app_name: str) -> ToolResult:
    """Launch an application by name or path."""
    if not app_name.strip():
        return ToolResult(success=False, error="Application name cannot be empty.")

    app = app_name.strip()
    clean = app.lower()

    try:
        # Build candidate binary names (exact name first, then aliases)
        candidates: list[str] = [app]
        if clean != app:
            candidates.append(clean)
        if clean in APP_ALIASES:
            for alias in APP_ALIASES[clean]:
                if alias not in candidates:
                    candidates.append(alias)
        for variant in [clean.replace(" ", "-"), clean.replace(" ", ""), clean.replace(" ", "_")]:
            if variant not in candidates:
                candidates.append(variant)

        # 1. Try finding in PATH
        for c in candidates:
            resolved = shutil.which(c)
            if resolved:
                subprocess.Popen([resolved])
                return ToolResult(
                    success=True,
                    data=f"Launched: {app} (via {resolved})",
                    metadata={"method": "popen_resolved", "path": resolved},
                )

        # 2. Try desktop launcher (gtk-launch / gio)
        desktop_id = _find_desktop_entry(app)
        if desktop_id:
            if shutil.which("gtk-launch"):
                subprocess.Popen(["gtk-launch", desktop_id])
                return ToolResult(
                    success=True,
                    data=f"Launched: {app} (via gtk-launch {desktop_id})",
                    metadata={"method": "gtk_launch", "desktop_id": desktop_id},
                )

        # 3. If path or URI exists, try xdg-open
        if os.path.exists(app) or app.startswith(("http://", "https://")):
            subprocess.Popen(["xdg-open", app])
            return ToolResult(
                success=True,
                data=f"Opened: {app}",
                metadata={"method": "xdg_open"},
            )

        # 4. Web app fallback if native binary is not installed on Linux
        WEB_FALLBACKS = {
            "whatsapp": "https://web.whatsapp.com",
            "spotify": "https://open.spotify.com",
            "youtube": "https://youtube.com",
            "chatgpt": "https://chatgpt.com",
            "telegram": "https://web.telegram.org",
            "twitter": "https://x.com",
            "x": "https://x.com",
            "gmail": "https://mail.google.com",
            "reddit": "https://reddit.com",
        }
        tokens = set(re.findall(r"[a-z0-9]+", clean))
        for key, fallback_url in WEB_FALLBACKS.items():
            if key == clean or key in tokens:
                webbrowser.open(fallback_url)
                return ToolResult(
                    success=True,
                    data=f"Opened {key.capitalize()} in web browser.",
                    metadata={"method": "web_fallback", "url": fallback_url},
                )

        return ToolResult(
            success=False,
            error=f"Application '{app}' not found in PATH or desktop entries."
        )

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to launch {app}: {e}")


@tool(name="open_url", verify=False, category="apps")
async def open_url(url: str) -> ToolResult:
    """Open a URL in the default web browser."""
    if not url.strip():
        return ToolResult(success=False, error="URL cannot be empty.")

    target = url.strip()
    if not target.startswith(("http://", "https://", "file://")):
        target = "https://" + target

    try:
        webbrowser.open(target)
        return ToolResult(
            success=True,
            data=f"Opened: {target}",
            metadata={"url": target},
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to open URL: {e}")


@tool(name="send_whatsapp", verify=False, category="apps")
async def send_whatsapp(message: str, target: str) -> ToolResult:
    """Open a WhatsApp chat with a pre-filled message for the user to review and send."""
    if not message.strip():
        return ToolResult(success=False, error="Message content cannot be empty.")
    if not target.strip():
        return ToolResult(success=False, error="Target contact or number cannot be empty.")

    target = target.strip()
    clean_target = target.replace("+", "").replace(" ", "").replace("-", "")
    is_number = clean_target.isdigit() and len(clean_target) > 5

    encoded_msg = urllib.parse.quote(message.strip())
    if is_number:
        fallback_url = f"https://wa.me/{clean_target}?text={encoded_msg}"
    else:
        fallback_url = f"https://web.whatsapp.com/send?text={encoded_msg}"
        
    try:
        webbrowser.open(fallback_url)
        return ToolResult(
            success=True,
            data=(
                f"Opened WhatsApp Web with the message pre-filled for "
                f"'{target}'. Review and press Enter to send."
            ),
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to open WhatsApp: {e}")


@tool(name="send_telegram", verify=False, category="apps")
async def send_telegram(message: str, target: str) -> ToolResult:
    """Open a Telegram chat with a pre-filled message for the user to review and send."""
    if not message.strip():
        return ToolResult(success=False, error="Message content cannot be empty.")
    if not target.strip():
        return ToolResult(success=False, error="Target contact cannot be empty.")

    target = target.strip()
    encoded_msg = urllib.parse.quote(message.strip())
    
    user = target
    if user.startswith("@"):
        user = user[1:]

    try:
        if user:
            url = f"https://t.me/{user}?text={encoded_msg}"
        else:
            url = f"https://t.me/share/url?url=&text={encoded_msg}"
            
        webbrowser.open(url)
        return ToolResult(
            success=True,
            data=(
                f"Opened Telegram with the message pre-filled for "
                f"'{target}'. Review and press Enter to send."
            ),
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to open Telegram: {e}")


async def _resolve_youtube_video_id(query: str) -> str | None:
    """Scrape the top YouTube video ID for a search query directly."""
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(search_url, headers=headers)
            if resp.status_code == 200:
                matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
                if matches:
                    return matches[0]
    except Exception:
        pass
    return None


@tool(name="play_youtube", verify=False, category="apps")
async def play_youtube(video_title: str) -> ToolResult:
    """Search for a video, track, or song on YouTube and immediately play it directly in the user's browser.

    Use this tool whenever the user asks to:
    - "play [song/artist/video]" (e.g. "play My Bad", "play Bohemian Rhapsody")
    - "play [title] on YouTube" or "play music"
    - "play the first song that comes"
    """
    if not video_title.strip():
        return ToolResult(success=False, error="Video title cannot be empty.")

    query = video_title.strip()
    
    # 1. Attempt to resolve direct top video ID for instant autoplay
    video_id = await _resolve_youtube_video_id(query)

    if video_id:
        watch_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
        try:
            webbrowser.open(watch_url)
            return ToolResult(
                success=True,
                data=f"Playing '{query}' on YouTube now (URL: {watch_url}). Playback started automatically.",
                metadata={"url": watch_url, "video_id": video_id, "query": query, "autoplay": True},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to launch browser: {e}",
            )

    # 2. Fallback to YouTube search results if direct ID scraping was unavailable
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
    
    try:
        webbrowser.open(search_url)
        return ToolResult(
            success=True,
            data=f"Opened YouTube search results for '{query}'.",
            metadata={"url": search_url, "query": query, "method": "search_results"},
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Failed to open YouTube search page: {e}",
        )


@tool(name="browser_search", verify=False, category="apps")
async def browser_search(query: str, engine: str = "google") -> ToolResult:
    """Search Google, YouTube, or DuckDuckGo directly in the user's web browser, or navigate to a website.

    Use this tool whenever the user asks to:
    - "search for X on Google" or "Google X"
    - "search for X on YouTube" or "open YouTube and search X"
    - "search for X in browser"
    - "open website X" or "open Google Chrome and search for X"

    Args:
        query: What to search for, or a site name/url (e.g. "YouTube", "best laptops", "reddit").
        engine: "google" (default), "youtube", or "duckduckgo".
    """
    if not query.strip():
        return ToolResult(success=False, error="Search query cannot be empty.")

    clean_q = query.strip()
    clean_engine = (engine or "google").lower().strip()

    if clean_q.startswith(("http://", "https://")):
        target_url = clean_q
    elif "." in clean_q and " " not in clean_q and not clean_q.endswith("."):
        target_url = "https://" + clean_q
    elif clean_q.lower() in ("youtube", "open youtube", "go to youtube"):
        target_url = "https://www.youtube.com"
    elif clean_engine == "youtube":
        encoded = urllib.parse.quote(clean_q)
        target_url = f"https://www.youtube.com/results?search_query={encoded}"
    elif clean_engine == "duckduckgo":
        encoded = urllib.parse.quote(clean_q)
        target_url = f"https://duckduckgo.com/?q={encoded}"
    else:
        encoded = urllib.parse.quote(clean_q)
        target_url = f"https://www.google.com/search?q={encoded}"

    try:
        webbrowser.open(target_url)
        return ToolResult(
            success=True,
            data=f"Navigated browser to: {target_url}",
            metadata={"query": clean_q, "engine": clean_engine, "url": target_url},
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to open browser search: {e}")

