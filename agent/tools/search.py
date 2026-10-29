"""Web search: Exa primary, Tavily then DuckDuckGo on error/timeout/empty."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent.cache import JsonCache
from agent.safety import assert_public_http_url
from agent.tools.base import USER_AGENT, BaseTool, ToolError
from agent.tracing import Tracer

ProviderFn = Callable[["SearchInput"], Awaitable["list[SearchHit]"]]


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str

    @field_validator("url")
    @classmethod
    def http_url(cls, value: str) -> str:
        if not value.startswith("http://") and not value.startswith("https://"):
            raise ValueError("search hit url must be http(s)")
        return value


class SearchInput(BaseModel):
    query: str = Field(min_length=1)
    num_results: int = Field(default=5, ge=1, le=10)


class SearchOutput(BaseModel):
    results: list[SearchHit]
    provider_used: str
    failed_providers: list[str] = Field(default_factory=list)


def _ddg_unwrap(href: str) -> str:
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return unquote(qs["uddg"][0])
    return href


class WebSearchTool(BaseTool[SearchInput, SearchOutput]):
    name = "web_search"
    max_attempts = 2

    def __init__(
        self,
        tracer: Tracer | None = None,
        timeout_s: float | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: JsonCache | None = None,
        exa_api_key: str | None = None,
        tavily_api_key: str | None = None,
        provider_fns: list[tuple[str, ProviderFn]] | None = None,
    ) -> None:
        super().__init__(tracer=tracer, timeout_s=timeout_s)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s or self.timeout_s,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        self.cache = cache
        self.exa_api_key = exa_api_key
        self.tavily_api_key = tavily_api_key
        self.provider_fns = provider_fns

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _execute(self, payload: SearchInput) -> SearchOutput:
        cache_key = f"web_search:{payload.query}:{payload.num_results}"
        if self.cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                if self.tracer:
                    self.tracer.totals.cache_hits += 1
                return SearchOutput.model_validate(hit)

        providers = self.provider_fns or self._default_providers()
        failed: list[str] = []
        last_error = "no search providers configured"
        for name, fn in providers:
            try:
                results = await fn(payload)
                usable: list[SearchHit] = []
                for item in results:
                    try:
                        assert_public_http_url(item.url, tool=self.name)
                    except ToolError:
                        continue
                    usable.append(item)
                    if len(usable) >= payload.num_results:
                        break
                if usable:
                    out = SearchOutput(results=usable, provider_used=name, failed_providers=failed)
                    if self.cache:
                        self.cache.set(cache_key, out.model_dump(mode="json"))
                    return out
                failed.append(f"{name}:empty")
            except Exception as exc:
                failed.append(f"{name}:{exc}")
                last_error = str(exc)
                continue
        raise ToolError(
            f"all search providers failed ({'; '.join(failed) or last_error})",
            tool=self.name,
            retryable=True,
        )

    def _default_providers(self) -> list[tuple[str, ProviderFn]]:
        ordered: list[tuple[str, ProviderFn]] = []
        if self.exa_api_key:
            ordered.append(("exa", self._exa))
        if self.tavily_api_key:
            ordered.append(("tavily", self._tavily))
        ordered.append(("duckduckgo", self._duckduckgo))
        return ordered

    async def _exa(self, payload: SearchInput) -> list[SearchHit]:
        response = await self._client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": self.exa_api_key or "", "Content-Type": "application/json"},
            json={"query": payload.query, "numResults": payload.num_results, "type": "auto"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        hits: list[SearchHit] = []
        for row in data.get("results") or []:
            url = row.get("url") or row.get("id")
            if not url:
                continue
            try:
                hits.append(
                    SearchHit(
                        title=row.get("title") or url,
                        url=url,
                        snippet=(row.get("text") or row.get("summary") or "")[:400],
                        provider="exa",
                    )
                )
            except ValidationError:
                continue
        return hits

    async def _tavily(self, payload: SearchInput) -> list[SearchHit]:
        response = await self._client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.tavily_api_key,
                "query": payload.query,
                "max_results": payload.num_results,
                "include_answer": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        hits: list[SearchHit] = []
        for row in data.get("results") or []:
            url = row.get("url")
            if not url:
                continue
            try:
                hits.append(
                    SearchHit(
                        title=row.get("title") or url,
                        url=url,
                        snippet=(row.get("content") or "")[:400],
                        provider="tavily",
                    )
                )
            except ValidationError:
                continue
        return hits

    async def _duckduckgo(self, payload: SearchInput) -> list[SearchHit]:
        # Instant-answer JSON is crude but has no API key. HTML is the real fallback.
        hits = await self._ddg_json(payload)
        if hits:
            return hits
        return await self._ddg_html(payload)

    async def _ddg_json(self, payload: SearchInput) -> list[SearchHit]:
        response = await self._client.get(
            "https://api.duckduckgo.com/",
            params={"q": payload.query, "format": "json", "no_html": 1, "no_redirect": 1},
        )
        response.raise_for_status()
        data = response.json()
        hits: list[SearchHit] = []
        abstract = data.get("AbstractURL")
        if abstract:
            hits.append(
                SearchHit(
                    title=data.get("Heading") or abstract,
                    url=abstract,
                    snippet=data.get("Abstract") or "",
                    provider="duckduckgo",
                )
            )
        for row in data.get("RelatedTopics") or []:
            if not isinstance(row, dict):
                continue
            url = row.get("FirstURL")
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=(row.get("Text") or url)[:120],
                    url=url,
                    snippet=row.get("Text") or "",
                    provider="duckduckgo",
                )
            )
        return hits

    async def _ddg_html(self, payload: SearchInput) -> list[SearchHit]:
        response = await self._client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": payload.query},
        )
        response.raise_for_status()
        html = response.text
        hits: list[SearchHit] = []
        # Intentionally small parser — DDG HTML is a fallback, not a product surface.
        for raw in _findall_hrefs(html):
            url = _ddg_unwrap(raw)
            if not url.startswith("http"):
                continue
            hits.append(SearchHit(title=url, url=url, snippet="", provider="duckduckgo"))
            if len(hits) >= payload.num_results:
                break
        return hits


def _findall_hrefs(html: str) -> list[str]:
    return re.findall(r'href="(https?://[^"]+|//duckduckgo.com/l/\?[^"]+)"', html)
