from typing import Dict, List, Set

from .interfaces import WebSearchProvider
from .models import WebSearchQuery, WebSearchResult


class WebSearchService:
    def __init__(self, providers: Dict[str, WebSearchProvider], allowed_providers: Set[str]):
        self.providers = providers
        self.allowed_providers = allowed_providers

    async def search(self, query: WebSearchQuery) -> List[WebSearchResult]:
        if query.provider not in self.allowed_providers:
            raise PermissionError(f"Provider '{query.provider}' is not allowed")
        provider = self.providers.get(query.provider)
        if not provider:
            raise ValueError(f"Provider '{query.provider}' not registered")
        return await provider.search(query)
