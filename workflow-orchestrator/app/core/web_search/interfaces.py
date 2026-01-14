from abc import ABC, abstractmethod
from typing import List

from .models import WebSearchQuery, WebSearchResult


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: WebSearchQuery) -> List[WebSearchResult]:
        raise NotImplementedError
