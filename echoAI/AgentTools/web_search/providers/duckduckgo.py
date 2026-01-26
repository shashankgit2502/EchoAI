from datetime import datetime
from typing import List

import httpx

from ..interfaces import WebSearchProvider
from ..models import WebSearchQuery, WebSearchResult


class DuckDuckGoSearchProvider(WebSearchProvider):
    async def search(self, query: WebSearchQuery) -> List[WebSearchResult]:
        params = {
            "q": query.query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.duckduckgo.com/", params=params)
            resp.raise_for_status()
            data = resp.json()

        results: List[WebSearchResult] = []
        for topic in data.get("RelatedTopics", [])[: query.max_results]:
            if "Text" in topic and "FirstURL" in topic:
                results.append(
                    WebSearchResult(
                        title=topic["Text"],
                        url=topic["FirstURL"],
                        snippet=topic["Text"],
                        source="duckduckgo",
                        retrieved_at=datetime.utcnow(),
                    )
                )
        return results
