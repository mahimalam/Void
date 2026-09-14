"""Web search and reading tools.

Uses duckduckgo-search for highly reliable web searching and BeautifulSoup for 
extracting clean, readable text from web pages.
"""

from __future__ import annotations

from tools.registry import ToolResult, tool


@tool(name="web_search", verify=False, category="web")
async def web_search(query: str, max_results: int = 5, timelimit: str | None = None) -> ToolResult:
    """Search the public web for current information. Returns titles, snippets, and URLs.

    USE THIS TOOL WHEN:
    - The user asks for any information that may have changed after your training
      cutoff: sports scores / schedules / fixtures (FIFA, cricket, IPL, Premier
      League, Champions League, etc.), current news, weather, stock prices,
      exchange rates, recent events, "who won", "what's the latest", "what
      happened today", product releases, ticket prices, traffic, etc.
    - The user explicitly asks you to "search", "google", or "look up" something.
    - You are uncertain whether a fact is still current.

    DO NOT USE THIS TOOL FOR:
    - Static knowledge (math, history of well-established facts, grammar, code patterns).
    - Anything that lives on the user's own computer (use read_file, list_dir, get_note, etc.).
    - Tasks that have a dedicated local tool (opening apps, playing music, taking notes).

    If the search snippets don't fully answer the question, follow up with the
    read_webpage tool on the most relevant URL.

    Args:
        query: The search query. Be specific — include team names, dates, or
               product names. Examples: "Liverpool vs Chelsea kickoff time
               June 22 2026", "Bitcoin price USD today", "Dhaka weather
               forecast tomorrow".
        max_results: How many search results to return (default 5; use 3-5
                     for narrow queries, up to 10 for broad ones).
        timelimit: Optional recency filter. Use 'd' for "today" / live
                   events, 'w' for "this week" / recent news, 'm' for "this
                   month", 'y' for "this year". Omit for evergreen queries.
    """
    if not query.strip():
        return ToolResult(success=False, error="Search query cannot be empty.")

    import httpx
    import asyncio
    import datetime
    
    from core.config import load_config
    try:
        cfg = load_config()
        tavily_keys = cfg.search.tavily_api_key
        if isinstance(tavily_keys, str): tavily_keys = [tavily_keys]
        elif not tavily_keys: tavily_keys = []
        
        serper_keys = cfg.search.serper_api_key
        if isinstance(serper_keys, str): serper_keys = [serper_keys]
        elif not serper_keys: serper_keys = []
    except Exception:
        tavily_keys = []
        serper_keys = []

    formatted_results = []
    
    # 1. Serper API
    for serper_key in serper_keys:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                headers = {
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json"
                }
                payload = {"q": query, "num": max_results}
                if timelimit == 'd': payload["tbs"] = "qdr:d"
                elif timelimit == 'w': payload["tbs"] = "qdr:w"
                elif timelimit == 'm': payload["tbs"] = "qdr:m"
                elif timelimit == 'y': payload["tbs"] = "qdr:y"

                resp = await client.post("https://google.serper.dev/search", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("organic", [])
                
                for idx, r in enumerate(results, 1):
                    formatted_results.append(f"{idx}. {r.get('title')}\nURL: {r.get('link')}\nSnippet: {r.get('snippet')}")
                
                if formatted_results:
                    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
                    header = f"Current System Time: {now_str}\nSearch results from Serper for \"{query}\":\n\n"
                    return ToolResult(success=True, data=header + "\n\n".join(formatted_results), metadata={"source": "serper"})
        except Exception:
            pass

    # 2. Tavily API
    for tavily_key in tavily_keys:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                time_range = "day" if timelimit == 'd' else "week" if timelimit == 'w' else "month" if timelimit == 'm' else "year" if timelimit == 'y' else None
                
                payload = {
                    "api_key": tavily_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results
                }
                if time_range:
                    payload["time_range"] = time_range
                    
                resp = await client.post("https://api.tavily.com/search", json=payload)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                
                for idx, r in enumerate(results, 1):
                    formatted_results.append(f"{idx}. {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content')}")
                
                if formatted_results:
                    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
                    header = f"Current System Time: {now_str}\nSearch results from Tavily for \"{query}\":\n\n"
                    return ToolResult(success=True, data=header + "\n\n".join(formatted_results), metadata={"source": "tavily"})
        except Exception:
            pass

    # 3. DuckDuckGo HTML Fallback
    try:
        from duckduckgo_search import DDGS
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query.strip(), max_results=max_results, timelimit=timelimit))
        results = await asyncio.to_thread(_search)
        
        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            url = item.get("href", "No URL")
            snippet = item.get("body", "No snippet available.")
            formatted_results.append(f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}")
            
        if formatted_results:
            now_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
            header = f"Current System Time: {now_str}\nSearch results from DDG for \"{query}\":\n\n"
            return ToolResult(success=True, data=header + "\n\n".join(formatted_results), metadata={"source": "ddg"})
    except Exception:
        pass
        
    return ToolResult(
        success=False, 
        error=f"All search backends failed or returned no results for \"{query}\". Try providing valid API keys in config.yaml."
    )


@tool(name="read_webpage", verify=False, category="web")
async def read_webpage(url: str) -> ToolResult:
    """Read the full readable text content of a specific webpage URL.

    USE THIS TOOL WHEN:
    - You just ran web_search and the snippets don't contain enough detail
      to answer the user's question. Pick the most relevant URL from the
      search results and call read_webpage on it for the full article.
    - The user asks you to read a specific webpage or article they named.
    - You need to verify a fact beyond what a search snippet shows.

    The page text is cleaned of ads, navigation, and scripts, then returned
    as markdown (or plain text if the page isn't HTML). Result is truncated
    to 15,000 characters for safety.

    Args:
        url: A full http(s) URL. Must start with "http://" or "https://".
    """
    if not url.strip().startswith("http"):
        return ToolResult(success=False, error="Invalid URL. Must start with http or https.")

    import httpx
    
    # 1. Try Jina Reader API
    jina_url = f"https://r.jina.ai/{url.strip()}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/markdown,text/plain",
        "X-Return-Format": "markdown"
    }
    
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(jina_url, headers=headers)
            if resp.status_code == 200:
                clean_text = resp.text.strip()
                if clean_text:
                    MAX_CHARS = 15000
                    if len(clean_text) > MAX_CHARS:
                        clean_text = clean_text[:MAX_CHARS] + "\n\n...[Content truncated due to length]..."
                    return ToolResult(success=True, data=f"Content of {url} (via Jina Reader):\n\n{clean_text}", metadata={"source": "jina"})
    except Exception:
        pass # Fallback to local BeautifulSoup
        
    # 2. Local BeautifulSoup Fallback
    try:
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url.strip(), headers=headers)
            resp.raise_for_status()
            
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ToolResult(success=False, error=f"Cannot read non-text content type: {content_type}")
                
            html = resp.text
            soup = BeautifulSoup(html, "lxml")
            
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "meta"]):
                element.extract()
                
            for table in soup.find_all("table"):
                markdown_table = []
                rows = table.find_all("tr")
                for i, row in enumerate(rows):
                    cols = row.find_all(["th", "td"])
                    if not cols:
                        continue
                    row_text = "| " + " | ".join(col.get_text(strip=True).replace("\n", " ") for col in cols) + " |"
                    markdown_table.append(row_text)
                    if i == 0 and cols and cols[0].name == "th":
                        separator = "| " + " | ".join("---" for _ in cols) + " |"
                        markdown_table.append(separator)
                
                if markdown_table:
                    md_text = "\n\n" + "\n".join(markdown_table) + "\n\n"
                    table.replace_with(md_text)
                
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            MAX_CHARS = 15000
            if len(clean_text) > MAX_CHARS:
                clean_text = clean_text[:MAX_CHARS] + "\n\n...[Content truncated due to length]..."
                
            if not clean_text.strip():
                return ToolResult(success=False, error="Webpage returned no readable text.")
                
            return ToolResult(success=True, data=f"Content of {url} (via local fallback):\n\n{clean_text}", metadata={"source": "local"})

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to read webpage: {e}")
