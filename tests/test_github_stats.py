from __future__ import annotations

import httpx
import pytest

from agent.tools.base import ToolError
from agent.tools.github_stats import GithubRepoInput, GithubRepoStatsTool, parse_owner_repo


def test_parse_repo() -> None:
    assert parse_owner_repo("https://github.com/qdrant/qdrant") == ("qdrant", "qdrant")
    assert parse_owner_repo("weaviate/weaviate.git") == ("weaviate", "weaviate")


async def test_github_stats_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/qdrant/qdrant/commits"):
            return httpx.Response(
                200,
                json=[{"commit": {"committer": {"date": "2026-01-02T00:00:00Z"}}}],
            )
        if path.endswith("/releases/latest"):
            return httpx.Response(200, json={"tag_name": "v1.13.0"})
        if path.endswith("/repos/qdrant/qdrant"):
            return httpx.Response(
                200,
                json={
                    "html_url": "https://github.com/qdrant/qdrant",
                    "stargazers_count": 12345,
                    "open_issues_count": 42,
                    "license": {"spdx_id": "Apache-2.0"},
                    "description": "vector db",
                },
            )
        return httpx.Response(404, json={"message": "nope"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = GithubRepoStatsTool(timeout_s=5, client=client)
    out = await tool.run(GithubRepoInput(repo="qdrant/qdrant"))
    assert out.stars == 12345
    assert out.license == "Apache-2.0"
    assert out.latest_release == "v1.13.0"
    assert out.last_commit_at is not None
    await client.aclose()


async def test_github_malformed_object() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = GithubRepoStatsTool(timeout_s=5, client=client)
    tool.max_attempts = 1
    with pytest.raises(ToolError, match="malformed"):
        await tool.run(GithubRepoInput(repo="foo/bar"))
    await client.aclose()
