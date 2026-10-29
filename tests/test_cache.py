from agent.cache import JsonCache
from agent.tools.search import SearchHit, SearchInput, WebSearchTool


def test_cache_disabled_never_stores(tmp_path) -> None:
    cache = JsonCache(tmp_path, enabled=False)
    cache.set("k", {"a": 1})
    assert cache.get("k") is None


async def test_search_uses_disk_cache(tmp_path, public_dns) -> None:
    calls = {"n": 0}

    async def provider(_payload):
        calls["n"] += 1
        return [
            SearchHit(
                title="Docs",
                url="https://docs.example.com/",
                snippet="ok",
                provider="exa",
            )
        ]

    cache = JsonCache(tmp_path / "c", enabled=True)
    tool = WebSearchTool(timeout_s=5, cache=cache, provider_fns=[("exa", provider)])
    payload = SearchInput(query="vector db")
    first = await tool.run(payload)
    second = await tool.run(payload)
    assert calls["n"] == 1
    assert first.results[0].url == second.results[0].url
    cache.close()
