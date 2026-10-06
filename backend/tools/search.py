"""
Web Search Tool Module
======================
Wraps TavilyClient to provide internet search functionality to DeepAgents.
"""

from typing import Literal
from tavily import TavilyClient
from backend.config import settings

tavily_client = None
if settings.TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize TavilyClient: {e}")


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a live web search for current information, documentation, and news."""
    if not tavily_client:
        return {
            "error": "TAVILY_API_KEY is not configured. Web search unavailable.",
            "query": query,
            "results": [],
        }
    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as exc:
        return {"error": str(exc), "query": query, "results": []}
