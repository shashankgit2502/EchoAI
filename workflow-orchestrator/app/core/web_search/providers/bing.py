from datetime import datetime
from typing import List

import httpx

from ..interfaces import WebSearchProvider
from ..models import WebSearchQuery, WebSearchResult


class BingSearchProvider(WebSearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"

    async def search(self, query: WebSearchQuery) -> List[WebSearchResult]:
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query.query,
            "count": query.max_results,
            "responseFilter": "Webpages",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(self.endpoint, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: List[WebSearchResult] = []
        for item in data.get("webPages", {}).get("value", []):
            results.append(
                WebSearchResult(
                    title=item.get("name") or "",
                    url=item.get("url") or "",
                    snippet=item.get("snippet") or "",
                    source="bing",
                    retrieved_at=datetime.utcnow(),
                )
            )
        return results
