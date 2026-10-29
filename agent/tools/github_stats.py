"""GitHub REST stats. Token is optional; unauthenticated is fine for evals."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from agent.cache import JsonCache
from agent.tools.base import USER_AGENT, BaseTool, ToolError
from agent.tracing import Tracer

GITHUB_API = "https://api.github.com"


class GithubRepoInput(BaseModel):
    repo: str = Field(min_length=3, description="owner/name or GitHub URL")


class GithubRepoStats(BaseModel):
    repo: str
    url: str
    stars: int
    license: str | None = None
    open_issues: int = 0
    last_commit_at: datetime | None = None
    latest_release: str | None = None
    description: str | None = None


def parse_owner_repo(raw: str) -> tuple[str, str]:
    text = raw.strip().rstrip("/")
    if text.startswith("http://") or text.startswith("https://"):
        path = urlparse(text).path.strip("/")
        parts = path.split("/")
    else:
        parts = text.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ToolError(f"cannot parse GitHub repo from {raw!r}", tool="github_repo_stats", retryable=False)
    owner, name = parts[0], parts[1].removesuffix(".git")
    return owner, name


class GithubRepoStatsTool(BaseTool[GithubRepoInput, GithubRepoStats]):
    name = "github_repo_stats"

    def __init__(
        self,
        tracer: Tracer | None = None,
        timeout_s: float | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: JsonCache | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(tracer=tracer, timeout_s=timeout_s)
        self._owns_client = client is None
        headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s or self.timeout_s,
            headers=headers,
            follow_redirects=True,
        )
        self.cache = cache
        self.token = token

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _execute(self, payload: GithubRepoInput) -> GithubRepoStats:
        owner, name = parse_owner_repo(payload.repo)
        repo = f"{owner}/{name}"
        cache_key = f"github_repo_stats:{repo}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.tracer:
                    self.tracer.totals.cache_hits += 1
                return GithubRepoStats.model_validate(cached)

        repo_json = await self._get_json(f"{GITHUB_API}/repos/{owner}/{name}")
        last_commit_at = await self._last_commit(owner, name)
        latest_release = await self._latest_release(owner, name)
        license_info = repo_json.get("license") or {}
        out = GithubRepoStats(
            repo=repo,
            url=repo_json.get("html_url") or f"https://github.com/{repo}",
            stars=int(repo_json.get("stargazers_count") or 0),
            license=license_info.get("spdx_id") if isinstance(license_info, dict) else None,
            open_issues=int(repo_json.get("open_issues_count") or 0),
            last_commit_at=last_commit_at,
            latest_release=latest_release,
            description=repo_json.get("description"),
        )
        if self.cache:
            self.cache.set(cache_key, out.model_dump(mode="json"))
        return out

    async def _get_json(self, url: str) -> dict:
        response = await self._client.get(url)
        if response.status_code == 404:
            raise ToolError(f"GitHub 404 for {url}", tool=self.name, retryable=False)
        if response.status_code == 403:
            raise ToolError("GitHub rate limited or forbidden", tool=self.name, retryable=True)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ToolError("malformed GitHub JSON object", tool=self.name, retryable=False)
        return data

    async def _last_commit(self, owner: str, name: str) -> datetime | None:
        response = await self._client.get(
            f"{GITHUB_API}/repos/{owner}/{name}/commits",
            params={"per_page": 1},
        )
        if response.status_code >= 400:
            return None
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return None
        date = (
            rows[0].get("commit", {}).get("committer", {}).get("date")
            or rows[0].get("commit", {}).get("author", {}).get("date")
        )
        if not date:
            return None
        return datetime.fromisoformat(date.replace("Z", "+00:00"))

    async def _latest_release(self, owner: str, name: str) -> str | None:
        response = await self._client.get(f"{GITHUB_API}/repos/{owner}/{name}/releases/latest")
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            return None
        data = response.json()
        if not isinstance(data, dict):
            return None
        return data.get("tag_name") or data.get("name")
