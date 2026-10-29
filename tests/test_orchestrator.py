from agent.config import Settings
from agent.llm import FakeLLM
from agent.orchestrator import Orchestrator
from agent.schemas import (
    ConfidenceLevel,
    Finding,
    ResearchPlan,
    SubQuestionPlan,
    SynthesisDraft,
    ToolChoice,
    ToolName,
)
from agent.tools.search import SearchHit
from agent.tracing import Tracer
from tests.fakes import FakeFetch, FakeGithub, FakeScholarly, FakeSearch


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="fake",
        cache_enabled=False,
        cache_dir=tmp_path / "c",
        outputs_dir=tmp_path / "o",
        traces_dir=tmp_path / "t",
        max_replans=1,
    )


def _tools(hits: list[SearchHit] | None = None):
    hits = hits or [
        SearchHit(title="A", url="https://docs.alpha.test/x", snippet="a", provider="fake"),
        SearchHit(title="B", url="https://docs.beta.test/x", snippet="b", provider="fake"),
    ]
    return {
        "web_search": FakeSearch(hits),
        "fetch_page": FakeFetch(),
        "github_repo_stats": FakeGithub(),
        "scholarly_search": FakeScholarly(),
    }


def _plan_web() -> ResearchPlan:
    return ResearchPlan(
        assumptions=["Interpret compare as docs + licensing."],
        sub_questions=[
            SubQuestionPlan(
                id="q1",
                question="What do official docs say?",
                tools=[ToolChoice(name=ToolName.WEB_SEARCH, reason="docs", query_hint="vector db docs")],
                sufficient_evidence="two official domains",
            )
        ],
        sufficient_evidence_overall="two domains",
    )


async def test_happy_path_writes_outputs(tmp_path) -> None:
    settings = _settings(tmp_path)
    tracer = Tracer.create(settings.traces_dir, "run-happy")
    llm = FakeLLM(settings, tracer)
    llm.set("ResearchPlan", _plan_web())
    llm.set(
        "SynthesisDraft",
        SynthesisDraft(
            short_answer="Two doc sites retrieved.",
            key_findings=[Finding(claim="Docs exist for both.", source_ids=["s1"])],
            confidence=ConfidenceLevel.HIGH,
            confidence_rationale="two domains",
            assumptions_and_limitations=[],
            next_steps=["Read API pages."],
        ),
    )
    orch = Orchestrator(settings, llm=llm, tools=_tools(), tracer=tracer)
    answer = await orch.run("Compare two vector databases")
    assert answer.key_findings
    assert answer.sources
    assert all(s.id.startswith("s") for s in answer.sources)
    assert (settings.outputs_dir / f"{answer.metadata.run_id}.json").exists()
    assert (settings.outputs_dir / f"{answer.metadata.run_id}.md").exists()
    assert answer.metadata.replan_used is False


async def test_invented_citation_is_sanitized(tmp_path) -> None:
    settings = _settings(tmp_path)
    tracer = Tracer.create(settings.traces_dir, "run-cite")
    llm = FakeLLM(settings, tracer)
    llm.set("ResearchPlan", _plan_web())
    llm.set(
        "SynthesisDraft",
        SynthesisDraft(
            short_answer="Invented cite",
            key_findings=[Finding(claim="Secret fact.", source_ids=["s99"])],
            confidence=ConfidenceLevel.HIGH,
            confidence_rationale="overconfident",
            assumptions_and_limitations=[],
            next_steps=[],
        ),
    )
    orch = Orchestrator(settings, llm=llm, tools=_tools(), tracer=tracer)
    answer = await orch.run("Compare two vector databases")
    cited = {i for f in answer.key_findings for i in f.source_ids}
    assert "s99" not in cited
    assert answer.confidence is not ConfidenceLevel.HIGH or answer.metadata.errors


async def test_replan_flag_when_single_domain(tmp_path) -> None:
    settings = _settings(tmp_path)
    tracer = Tracer.create(settings.traces_dir, "run-replan")
    llm = FakeLLM(settings, tracer)
    plans = iter(
        [
            _plan_web(),
            ResearchPlan(
                assumptions=[],
                sub_questions=[
                    SubQuestionPlan(
                        id="q2",
                        question="github stars",
                        tools=[
                            ToolChoice(
                                name=ToolName.GITHUB_REPO_STATS,
                                reason="repo",
                                query_hint="qdrant/qdrant",
                            )
                        ],
                        sufficient_evidence="github",
                    )
                ],
                sufficient_evidence_overall="add repo",
            ),
        ]
    )
    llm.set("ResearchPlan", lambda _s, _u, _sc: next(plans))
    llm.set(
        "SynthesisDraft",
        SynthesisDraft(
            short_answer="Partial.",
            key_findings=[Finding(claim="Not found broadly.", source_ids=[])],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_rationale="thin",
            assumptions_and_limitations=[],
            next_steps=["Fetch more domains."],
        ),
    )
    same_host = [
        SearchHit(title="A", url="https://docs.alpha.test/a", snippet="a", provider="fake"),
        SearchHit(title="B", url="https://docs.alpha.test/b", snippet="b", provider="fake"),
    ]
    orch = Orchestrator(settings, llm=llm, tools=_tools(same_host), tracer=tracer)
    answer = await orch.run("Only one vendor?")
    assert answer.metadata.replan_used is True
