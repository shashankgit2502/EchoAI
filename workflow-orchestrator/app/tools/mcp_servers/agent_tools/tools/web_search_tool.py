import os

from app.core.tools_ext.web_search.service import WebSearchService
from app.core.tools_ext.web_search.models import WebSearchQuery
from app.core.tools_ext.web_search.providers.bing import BingSearchProvider
from app.core.tools_ext.web_search.providers.duckduckgo import DuckDuckGoSearchProvider
from app.core.tools_ext.web_search.providers.google import GoogleSearchProvider

bing_provider = BingSearchProvider(api_key=os.environ.get("BING_API_KEY", ""))
duck_provider = DuckDuckGoSearchProvider()

google_provider = GoogleSearchProvider(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    cx=os.environ.get("GOOGLE_CX", ""),
)

web_search_service = WebSearchService(
    providers={
        "bing": bing_provider,
        "duckduckgo": duck_provider,
        "google": google_provider,
    },
    allowed_providers={"bing"},
)


async def handle_web_search(arguments: dict):
    query = WebSearchQuery(**arguments)
    results = await web_search_service.search(query)
    return [r.model_dump() for r in results]
