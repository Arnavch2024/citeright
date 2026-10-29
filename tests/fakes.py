"""Shared fake tools for executor/orchestrator tests. No network."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.tools.base import BaseTool
from agent.tools.fetch import FetchInput, FetchOutput
from agent.tools.github_stats import GithubRepoInput, GithubRepoStats
from agent.tools.scholarly import ScholarlyInput, ScholarlyOutput, ScholarlyPaper
from agent.tools.search import SearchHit, SearchInput, SearchOutput

LONG = ("Primary-source documentation suitable for RAG comparison. " * 20)


class FakeSearch(BaseTool[SearchInput, SearchOutput]):
    name = "web_search"

    def __init__(self, hits: list[SearchHit], **kwargs) -> None:
        super().__init__(**kwargs)
        self.hits = hits

    async def _execute(self, payload: SearchInput) -> SearchOutput:
        return SearchOutput(results=self.hits, provider_used="fake")


class FakeFetch(BaseTool[FetchInput, FetchOutput]):
    name = "fetch_page"

    async def _execute(self, payload: FetchInput) -> FetchOutput:
        return FetchOutput(
            url=payload.url,
            final_url=payload.url,
            title=payload.url,
            text=LONG + payload.url,
        )


class FakeGithub(BaseTool[GithubRepoInput, GithubRepoStats]):
    name = "github_repo_stats"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    async def _execute(self, payload: GithubRepoInput) -> GithubRepoStats:
        self.calls += 1
        return GithubRepoStats(
            repo=payload.repo,
            url=f"https://github.com/{payload.repo}",
            stars=100,
            license="Apache-2.0",
            open_issues=3,
            last_commit_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            latest_release="v1.0.0",
            description="vector database " + LONG,
        )


class FakeScholarly(BaseTool[ScholarlyInput, ScholarlyOutput]):
    name = "scholarly_search"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    async def _execute(self, payload: ScholarlyInput) -> ScholarlyOutput:
        self.calls += 1
        return ScholarlyOutput(
            papers=[
                ScholarlyPaper(
                    title="Multi-agent customer support",
                    url="https://arxiv.org/abs/2401.00001",
                    year=2024,
                    abstract=LONG,
                    source="arxiv",
                )
            ],
            provider_used="fake",
        )
