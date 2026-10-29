"""Construct the tool set from Settings. Phase 3 executor should not new-up keys by hand."""

from __future__ import annotations

from typing import Any

from agent.cache import JsonCache
from agent.config import Settings
from agent.tools.base import BaseTool
from agent.tools.fetch import FetchPageTool
from agent.tools.github_stats import GithubRepoStatsTool
from agent.tools.scholarly import ScholarlySearchTool
from agent.tools.search import WebSearchTool
from agent.tracing import Tracer


def build_cache(settings: Settings) -> JsonCache:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return JsonCache(settings.cache_dir, enabled=settings.cache_enabled)


def build_tools(
    settings: Settings,
    tracer: Tracer | None = None,
    cache: JsonCache | None = None,
) -> dict[str, BaseTool[Any, Any]]:
    cache = cache if cache is not None else build_cache(settings)
    timeout = settings.tool_timeout_s
    return {
        "web_search": WebSearchTool(
            tracer=tracer,
            timeout_s=timeout,
            cache=cache,
            exa_api_key=settings.exa_api_key,
            tavily_api_key=settings.tavily_api_key,
        ),
        "fetch_page": FetchPageTool(
            tracer=tracer,
            timeout_s=timeout,
            cache=cache,
            fixture_root=settings.fixtures_dir,
        ),
        "github_repo_stats": GithubRepoStatsTool(
            tracer=tracer,
            timeout_s=timeout,
            cache=cache,
            token=settings.github_token,
        ),
        "scholarly_search": ScholarlySearchTool(
            tracer=tracer,
            timeout_s=timeout,
            cache=cache,
            s2_api_key=settings.semantic_scholar_api_key,
        ),
    }
