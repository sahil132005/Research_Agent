"""
tools.py — External tool integrations.

Currently contains:
  web_search(queries, max_results_per_query) — DuckDuckGo search (no API key needed).

Adding new tools: create a new function here and call it from the relevant step in steps.py.
The output of every tool must be a plain Python structure (list / dict) so it can be
stored in the agent's state and passed to subsequent LLM steps.
"""

from duckduckgo_search import DDGS


def web_search(queries: list[str], max_results_per_query: int = 4) -> list[dict]:
    """
    Execute one or more DuckDuckGo text searches and return de-duplicated results.

    Args:
        queries:               List of search query strings (produced by Step 1).
        max_results_per_query: Max results to fetch per individual query.

    Returns:
        List of result dicts, each with keys:
          - query   : the search string that produced this result
          - title   : page title
          - url     : page URL
          - snippet : short text excerpt from the page

    On per-query failure the error is printed and that query is skipped —
    the chain continues with whatever results were already collected.
    """
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    with DDGS() as ddgs:
        for query in queries:
            try:
                raw = list(ddgs.text(query, max_results=max_results_per_query))
                for r in raw:
                    url = r.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(
                            {
                                "query":   query,
                                "title":   r.get("title", ""),
                                "url":     url,
                                "snippet": r.get("body", ""),
                            }
                        )
            except Exception as exc:
                # Non-fatal: log and keep going
                print(f"  [Tool Warning] Search failed for '{query}': {exc}")

    return all_results
