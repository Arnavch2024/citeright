from agent.budget import RunBudget
from agent.config import Settings
from agent.executor import Executor
from agent.schemas import ResearchPlan, SubQuestionPlan, ToolChoice, ToolName
from agent.tools.search import SearchHit
from tests.fakes import FakeFetch, FakeGithub, FakeScholarly, FakeSearch


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="fake",
        cache_enabled=False,
        cache_dir=tmp_path / "c",
        outputs_dir=tmp_path / "o",
        traces_dir=tmp_path / "t",
    )


async def test_github_path_not_used_for_papers(tmp_path) -> None:
    settings = _settings(tmp_path)
    gh = FakeGithub()
    sch = FakeScholarly()
    tools = {
        "web_search": FakeSearch([]),
        "fetch_page": FakeFetch(),
        "github_repo_stats": gh,
        "scholarly_search": sch,
    }
    plan = ResearchPlan(
        assumptions=[],
        sub_questions=[
            SubQuestionPlan(
                id="q1",
                question="papers on multi-agent support",
                tools=[ToolChoice(name=ToolName.SCHOLARLY_SEARCH, reason="papers", query_hint="multi-agent")],
                sufficient_evidence="arxiv",
            )
        ],
        sufficient_evidence_overall="papers",
    )
    exe = Executor(settings, tools, RunBudget(settings))
    result = await exe.run_plan(plan)
    assert sch.calls == 1
    assert gh.calls == 0
    assert any("arxiv.org" in d.url for d in result.documents)


async def test_github_path_for_repo_stats(tmp_path) -> None:
    settings = _settings(tmp_path)
    gh = FakeGithub()
    tools = {
        "web_search": FakeSearch([]),
        "fetch_page": FakeFetch(),
        "github_repo_stats": gh,
        "scholarly_search": FakeScholarly(),
    }
    plan = ResearchPlan(
        assumptions=[],
        sub_questions=[
            SubQuestionPlan(
                id="q1",
                question="qdrant stars",
                tools=[ToolChoice(name=ToolName.GITHUB_REPO_STATS, reason="stats", query_hint="qdrant/qdrant")],
                sufficient_evidence="github",
            )
        ],
        sufficient_evidence_overall="repo",
    )
    result = await Executor(settings, tools, RunBudget(settings)).run_plan(plan)
    assert gh.calls == 1
    assert result.documents[0].url.startswith("https://github.com/qdrant/qdrant")


async def test_search_auto_fetches(tmp_path) -> None:
    settings = _settings(tmp_path)
    tools = {
        "web_search": FakeSearch(
            [
                SearchHit(title="A", url="https://docs.alpha.test/x", snippet="a", provider="fake"),
                SearchHit(title="B", url="https://docs.beta.test/x", snippet="b", provider="fake"),
            ]
        ),
        "fetch_page": FakeFetch(),
        "github_repo_stats": FakeGithub(),
        "scholarly_search": FakeScholarly(),
    }
    plan = ResearchPlan(
        assumptions=[],
        sub_questions=[
            SubQuestionPlan(
                id="q1",
                question="compare engines",
                tools=[ToolChoice(name=ToolName.WEB_SEARCH, reason="web", query_hint="engines")],
                sufficient_evidence="docs",
            )
        ],
        sufficient_evidence_overall="two docs",
    )
    result = await Executor(settings, tools, RunBudget(settings)).run_plan(plan)
    hosts = {d.domain for d in result.documents}
    assert "docs.alpha.test" in hosts
    assert "docs.beta.test" in hosts
