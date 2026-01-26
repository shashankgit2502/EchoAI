from datetime import datetime
from typing import List

import httpx

from ..interfaces import WebSearchProvider
from ..models import WebSearchQuery, WebSearchResult


class GoogleSearchProvider(WebSearchProvider):
    def __init__(self, api_key: str, cx: str):
        self.api_key = api_key
        self.cx = cx
        self.endpoint = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: WebSearchQuery) -> List[WebSearchResult]:
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query.query,
            "num": min(query.max_results, 10),
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(self.endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: List[WebSearchResult] = []
        for item in data.get("items", []):
            results.append(
                WebSearchResult(
                    title=item.get("title") or "",
                    url=item.get("link") or "",
                    snippet=item.get("snippet") or "",
                    source="google",
                    retrieved_at=datetime.utcnow(),
                )
            )
        return results
