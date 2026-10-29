"""EXECUTE: run planned tools, fetch pages, score, select passages.

Search snippets are not evidence. After web_search we fetch top hits (capped)
so CHECK can count real independent domains.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.budget import RunBudget
from agent.config import Settings
from agent.llm import LLMClient, LLMError
from agent.retrieval.base import select_passages
from agent.safety import wrap_untrusted
from agent.schemas import (
    Passage,
    RelevanceBatch,
    ResearchPlan,
    ScoredDocument,
    Source,
    SubQuestionPlan,
    ToolName,
)
from agent.scoring import score_document, select_sources
from agent.tools.base import BaseTool, ToolError
from agent.tools.fetch import FetchInput, FetchOutput
from agent.tools.github_stats import GithubRepoInput, GithubRepoStats, parse_owner_repo
from agent.tools.scholarly import ScholarlyInput, ScholarlyOutput
from agent.tools.search import SearchHit, SearchInput, SearchOutput
from agent.tracing import Tracer

CHEAP_FILTER_SYSTEM = """Select which search-result URLs are worth fetching for the sub-question.
Keep official docs, repos, and papers. Drop obvious forums/spam. Return URLs only from the list."""


@dataclass
class SubQuestionResult:
    sub_id: str
    question: str
    documents: list[ScoredDocument] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    results: list[SubQuestionResult]
    documents: list[ScoredDocument]
    retrieved_ids: set[str]
    retrieved_urls: set[str]
    failures: list[str]
    next_source_i: int


class Executor:
    def __init__(
        self,
        settings: Settings,
        tools: dict[str, BaseTool[Any, Any]],
        budget: RunBudget,
        llm: LLMClient | None = None,
        tracer: Tracer | None = None,
        source_metadata_map: dict[str, Source] | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.budget = budget
        self.llm = llm
        self.tracer = tracer
        self.source_metadata_map = source_metadata_map or {}
        self.reference_time = reference_time
        self._source_i = 0
        self._seen_urls: set[str] = set()

    def _sid(self) -> str:
        self._source_i += 1
        return f"s{self._source_i}"

    async def run_plan(self, plan: ResearchPlan, start_i: int = 0) -> ExecutionResult:
        self._source_i = start_i
        # Independent sub-questions run concurrently; tools inside one sub-q stay ordered.
        sem = asyncio.Semaphore(3)

        async def one(sq: SubQuestionPlan) -> SubQuestionResult:
            async with sem:
                return await self._run_subquestion(sq)

        parts = await asyncio.gather(*[one(sq) for sq in plan.sub_questions])
        docs: list[ScoredDocument] = []
        failures: list[str] = []
        for part in parts:
            docs.extend(part.documents)
            failures.extend(part.failures)
        # WHY: the no-filter eval ablation isolates the effect of source scoring/dedup.
        kept = select_sources(docs) if self.settings.source_filter_enabled else docs
        kept_ids = {d.source_id for d in kept}
        # Passages stay attached to each sub-question; documents are the global scored set.
        return ExecutionResult(
            results=list(parts),
            documents=kept,
            retrieved_ids={d.source_id for d in docs},
            retrieved_urls={d.url for d in docs},
            failures=failures,
            next_source_i=self._source_i,
        )

    async def _run_subquestion(self, sq: SubQuestionPlan) -> SubQuestionResult:
        out = SubQuestionResult(sub_id=sq.id, question=sq.question)
        hits: list[SearchHit] = []
        for choice in sq.tools:
            name = str(choice.name)
            hint = (choice.query_hint or sq.question).strip()
            out.tools_used.append(name)
            try:
                if name == ToolName.WEB_SEARCH:
                    hits.extend(await self._search(hint))
                elif name == ToolName.FETCH_PAGE:
                    doc = await self._fetch(hint)
                    if doc:
                        out.documents.append(doc)
                elif name == ToolName.GITHUB_REPO_STATS:
                    doc = await self._github(hint)
                    if doc:
                        out.documents.append(doc)
                elif name == ToolName.SCHOLARLY_SEARCH:
                    out.documents.extend(await self._scholarly(hint))
                else:
                    out.failures.append(f"{sq.id}: unknown tool {name}")
            except ToolError as exc:
                out.failures.append(f"{sq.id}:{name}:{exc.reason}")

        fetch_urls = await self._urls_to_fetch(sq.question, hits)
        for url in fetch_urls:
            try:
                doc = await self._fetch(url)
                if doc:
                    out.documents.append(doc)
                    if "fetch_page" not in out.tools_used:
                        out.tools_used.append("fetch_page")
            except ToolError as exc:
                out.failures.append(f"{sq.id}:fetch:{exc.reason}")

        for doc in out.documents:
            out.passages.extend(
                select_passages(
                    sq.question,
                    doc.text,
                    self.settings.passages_k,
                    source_id=doc.source_id,
                    backend=self.settings.retrieval_backend,
                )
            )
        return out

    async def _search(self, query: str) -> list[SearchHit]:
        tool = self.tools.get("web_search")
        if tool is None or not self.budget.can_call_tool():
            return []
        self.budget.record_tool()
        result = await tool.run(SearchInput(query=query, num_results=5))
        assert isinstance(result, SearchOutput)
        return result.results

    async def _urls_to_fetch(self, question: str, hits: list[SearchHit]) -> list[str]:
        urls = [h.url for h in hits if h.url not in self._seen_urls]
        if not urls:
            return []
        if self.llm and len(urls) > 3:
            try:
                listing = "\n".join(f"- {h.title} :: {h.url} :: {h.snippet[:160]}" for h in hits)
                batch = await self.llm.generate_structured(
                    CHEAP_FILTER_SYSTEM,
                    f"Sub-question: {question}\nHits:\n{listing}",
                    RelevanceBatch,
                    "cheap",
                )
                allowed = set(urls)
                picked = [u for u in batch.keep_urls if u in allowed]
                if picked:
                    urls = picked
            except LLMError:
                pass
        return urls[:4]

    async def _fetch(self, url: str) -> ScoredDocument | None:
        if not (url.startswith("http://") or url.startswith("https://") or url.startswith("file:")):
            return None
        if url in self._seen_urls:
            return None
        tool = self.tools.get("fetch_page")
        if tool is None or not self.budget.can_call_tool():
            return None
        self.budget.record_tool()
        self._seen_urls.add(url)
        result = await tool.run(FetchInput(url=url))
        assert isinstance(result, FetchOutput)
        doc = score_document(
            source_id=self._sid(),
            url=result.final_url or result.url,
            title=result.title,
            text=result.text,
            injection_flagged=result.injection_flagged,
            reference_time=self.reference_time,
        )
        return self._restore_source_metadata(doc)

    async def _github(self, hint: str) -> ScoredDocument | None:
        tool = self.tools.get("github_repo_stats")
        if tool is None or not self.budget.can_call_tool():
            return None
        repo = hint
        try:
            owner, name = parse_owner_repo(hint)
            repo = f"{owner}/{name}"
        except ToolError:
            # Hint might be a sentence; last path-like token often still parses.
            for token in hint.replace(",", " ").split():
                try:
                    owner, name = parse_owner_repo(token)
                    repo = f"{owner}/{name}"
                    break
                except ToolError:
                    continue
        self.budget.record_tool()
        stats = await tool.run(GithubRepoInput(repo=repo))
        assert isinstance(stats, GithubRepoStats)
        text = (
            f"# {stats.repo}\n\n"
            "GitHub repository statistics retrieved via the GitHub REST API "
            "for source-grounded comparison (stars, license, activity).\n\n"
            f"URL: {stats.url}\n"
            f"Stars: {stats.stars}\n"
            f"License: {stats.license}\n"
            f"Open issues: {stats.open_issues}\n"
            f"Last commit: {stats.last_commit_at}\n"
            f"Latest release: {stats.latest_release}\n"
            f"Description: {stats.description}\n"
        )
        self._seen_urls.add(stats.url)
        doc = score_document(
            source_id=self._sid(),
            url=stats.url,
            title=stats.repo,
            text=text,
            published_at=stats.last_commit_at,
            reference_time=self.reference_time,
        )
        return self._restore_source_metadata(doc)

    async def _scholarly(self, query: str) -> list[ScoredDocument]:
        tool = self.tools.get("scholarly_search")
        if tool is None or not self.budget.can_call_tool():
            return []
        self.budget.record_tool()
        result = await tool.run(ScholarlyInput(query=query, limit=5))
        assert isinstance(result, ScholarlyOutput)
        docs: list[ScoredDocument] = []
        for paper in result.papers:
            if paper.url in self._seen_urls:
                continue
            self._seen_urls.add(paper.url)
            year = None
            if paper.year:
                from datetime import datetime, timezone

                year = datetime(paper.year, 1, 1, tzinfo=timezone.utc)
            text = f"# {paper.title}\n\nYear: {paper.year}\n\n{paper.abstract}\n"
            doc = score_document(
                source_id=self._sid(),
                url=paper.url,
                title=paper.title,
                text=text,
                published_at=year,
                reference_time=self.reference_time,
            )
            docs.append(self._restore_source_metadata(doc))
        return docs

    def _restore_source_metadata(self, doc: ScoredDocument) -> ScoredDocument:
        """Reuse original source identity/timestamp so replay prompts stay stable."""
        source = self.source_metadata_map.get(doc.url)
        if source is None:
            return doc
        return doc.model_copy(
            update={
                "source_id": source.id,
                "url": source.url,
                "title": source.title,
                "retrieved_at": source.retrieved_at,
                "domain_type": source.domain_type,
            }
        )


def evidence_for_prompt(results: list[SubQuestionResult], docs: list[ScoredDocument]) -> str:
    """Passages wrapped as untrusted data. The model must treat this as quoted evidence."""
    by_id = {d.source_id: d for d in docs}
    blocks: list[str] = []
    for part in results:
        blocks.append(f"### Sub-question {part.sub_id}: {part.question}")
        used: list[Passage] = []
        for p in part.passages:
            if p.source_id in by_id:
                used.append(p)
        if not used:
            for doc in part.documents:
                if doc.source_id in by_id:
                    used.append(Passage(source_id=doc.source_id, text=doc.text[:1200], score=doc.score))
        if not used:
            blocks.append("(no retrieved evidence)")
            continue
        for p in used[:8]:
            meta = by_id.get(p.source_id)
            heading = f" {p.heading}" if p.heading else ""
            header = f"[{p.source_id}] {meta.url if meta else ''}{heading}"
            blocks.append(header + "\n" + wrap_untrusted(p.text[:1500]))
    return "\n\n".join(blocks)
