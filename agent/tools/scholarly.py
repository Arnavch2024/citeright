"""Papers via Semantic Scholar, then arXiv. Stretch tool for architecture/RAG queries."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import httpx
from pydantic import BaseModel, Field

from agent.cache import JsonCache
from agent.tools.base import USER_AGENT, BaseTool, ToolError
from agent.tracing import Tracer

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


class ScholarlyInput(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=10)


class ScholarlyPaper(BaseModel):
    title: str
    url: str
    year: int | None = None
    abstract: str = ""
    source: str
    citation_count: int | None = None


class ScholarlyOutput(BaseModel):
    papers: list[ScholarlyPaper]
    provider_used: str
    failed_providers: list[str] = Field(default_factory=list)


class ScholarlySearchTool(BaseTool[ScholarlyInput, ScholarlyOutput]):
    name = "scholarly_search"
    max_attempts = 2

    def __init__(
        self,
        tracer: Tracer | None = None,
        timeout_s: float | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: JsonCache | None = None,
        s2_api_key: str | None = None,
    ) -> None:
        super().__init__(tracer=tracer, timeout_s=timeout_s)
        self._owns_client = client is None
        headers = {"User-Agent": USER_AGENT}
        if s2_api_key:
            headers["x-api-key"] = s2_api_key
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s or self.timeout_s,
            headers=headers,
            follow_redirects=True,
        )
        self.cache = cache

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _execute(self, payload: ScholarlyInput) -> ScholarlyOutput:
        cache_key = f"scholarly_search:{payload.query}:{payload.limit}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.tracer:
                    self.tracer.totals.cache_hits += 1
                return ScholarlyOutput.model_validate(cached)

        failed: list[str] = []
        for name, fn in (("semanticscholar", self._s2), ("arxiv", self._arxiv)):
            try:
                papers = await fn(payload)
                if papers:
                    out = ScholarlyOutput(papers=papers[: payload.limit], provider_used=name, failed_providers=failed)
                    if self.cache:
                        self.cache.set(cache_key, out.model_dump(mode="json"))
                    return out
                failed.append(f"{name}:empty")
            except Exception as exc:
                failed.append(f"{name}:{exc}")
        raise ToolError(
            f"scholarly search failed ({'; '.join(failed)})",
            tool=self.name,
            retryable=True,
        )

    async def _s2(self, payload: ScholarlyInput) -> list[ScholarlyPaper]:
        response = await self._client.get(
            S2_SEARCH,
            params={
                "query": payload.query,
                "limit": payload.limit,
                "fields": "title,year,url,abstract,citationCount,externalIds",
            },
        )
        response.raise_for_status()
        data = response.json()
        papers: list[ScholarlyPaper] = []
        for row in data.get("data") or []:
            url = row.get("url")
            if not url:
                ext = row.get("externalIds") or {}
                arxiv_id = ext.get("ArXiv")
                url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            if not url:
                continue
            papers.append(
                ScholarlyPaper(
                    title=row.get("title") or url,
                    url=url,
                    year=row.get("year"),
                    abstract=(row.get("abstract") or "")[:800],
                    source="semanticscholar",
                    citation_count=row.get("citationCount"),
                )
            )
        return papers

    async def _arxiv(self, payload: ScholarlyInput) -> list[ScholarlyPaper]:
        response = await self._client.get(
            ARXIV_API,
            params={"search_query": f"all:{payload.query}", "start": 0, "max_results": payload.limit},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        papers: list[ScholarlyPaper] = []
        for entry in root.findall(f"{ATOM}entry"):
            title = (entry.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
            link = entry.findtext(f"{ATOM}id") or ""
            summary = (entry.findtext(f"{ATOM}summary") or "").strip()
            published = entry.findtext(f"{ATOM}published") or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            if not link:
                continue
            papers.append(
                ScholarlyPaper(
                    title=title or link,
                    url=link,
                    year=year,
                    abstract=summary[:800],
                    source="arxiv",
                )
            )
        return papers
