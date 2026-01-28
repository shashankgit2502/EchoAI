from typing import Dict, List, Set, Optional

from .interfaces import WebSearchProvider
from .models import WebSearchQuery, WebSearchResult


class WebSearchService:
    """
    Web search service that supports multiple search providers.

    Can be instantiated with no arguments for tool executor compatibility,
    or with explicit providers for custom configuration.
    """

    def __init__(
        self,
        providers: Optional[Dict[str, WebSearchProvider]] = None,
        allowed_providers: Optional[Set[str]] = None
    ):
        # If no providers given, initialize with defaults
        if providers is None:
            import os
            from .providers.bing import BingSearchProvider

            # Bing API key from env or placeholder
            bing_api_key = os.environ.get("BING_API_KEY", "XYZ")

            providers = {
                "bing": BingSearchProvider(api_key=bing_api_key),
            }

        if allowed_providers is None:
            allowed_providers = {"bing"}

        self.providers = providers
        self.allowed_providers = allowed_providers

    async def search(self, input_data: Dict) -> Dict:
        """
        Perform web search. Accepts dict input for tool executor compatibility.

        Args:
            input_data: Dict with keys:
                - query (str): Search query string (required)
                - provider (str): Search provider - google, bing, duckduckgo (default: duckduckgo)
                - max_results (int): Maximum results to return (default: 10)

        Returns:
            Dict with query, provider, results list, and total_results count
        """
        # Convert dict input to WebSearchQuery
        query_str = input_data.get("query")
        if not query_str:
            raise ValueError("Missing required field: 'query'")

        provider = input_data.get("provider", "duckduckgo")
        max_results = input_data.get("max_results", 10)

        query = WebSearchQuery(
            query=query_str,
            provider=provider,
            max_results=max_results
        )

        # Validate provider
        if query.provider not in self.allowed_providers:
            raise PermissionError(f"Provider '{query.provider}' is not allowed")

        provider_instance = self.providers.get(query.provider)
        if not provider_instance:
            raise ValueError(f"Provider '{query.provider}' not registered")

        # Execute search
        results = await provider_instance.search(query)

        # Convert results to dict for tool executor
        return {
            "query": query_str,
            "provider": provider,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "position": idx + 1
                }
                for idx, r in enumerate(results)
            ],
            "total_results": len(results)
        }
