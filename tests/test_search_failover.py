from __future__ import annotations

import httpx
import pytest

from agent.tools.base import ToolError
from agent.tools.search import SearchHit, SearchInput, WebSearchTool
from agent.tracing import Tracer


async def test_exa_failure_falls_back_to_tavily(public_dns, tmp_path) -> None:
    async def boom(_payload):
        raise TimeoutError("exa down")

    async def tavily(_payload):
        return [
            SearchHit(
                title="Qdrant docs",
                url="https://qdrant.tech/documentation/",
                snippet="vector db",
                provider="tavily",
            )
        ]

    tool = WebSearchTool(
        tracer=Tracer.create(tmp_path, "s1"),
        timeout_s=5,
        provider_fns=[("exa", boom), ("tavily", tavily)],
    )
    out = await tool.run(SearchInput(query="qdrant vs milvus"))
    assert out.provider_used == "tavily"
    assert out.failed_providers[0].startswith("exa:")
    assert out.results[0].url.endswith("/documentation/")


async def test_all_providers_fail() -> None:
    async def boom(_payload):
        raise RuntimeError("nope")

    tool = WebSearchTool(timeout_s=5, provider_fns=[("exa", boom), ("tavily", boom)])
    tool.max_attempts = 1
    with pytest.raises(ToolError, match="all search providers failed"):
        await tool.run(SearchInput(query="x"))


async def test_malformed_search_hit_skipped(public_dns) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "bad", "url": "javascript:alert(1)"},
                    {"title": "ok", "url": "https://weaviate.io/developers/weaviate", "content": "docs"},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = WebSearchTool(timeout_s=5, client=client, tavily_api_key="tvly-test", exa_api_key=None)
    out = await tool.run(SearchInput(query="weaviate"))
    assert out.provider_used == "tavily"
    assert len(out.results) == 1
    assert "weaviate" in out.results[0].url
    await client.aclose()
