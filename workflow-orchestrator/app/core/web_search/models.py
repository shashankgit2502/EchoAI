from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field

SearchProvider = Literal["bing", "google", "duckduckgo"]


class WebSearchQuery(BaseModel):
    provider: SearchProvider = Field(..., description="Search provider to use")
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=50)


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: SearchProvider
    retrieved_at: datetime
