from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent.schemas import (
    ConfidenceLevel,
    Finding,
    ResearchAnswer,
    ResearchPlan,
    RunMetadata,
    Source,
    SubQuestionPlan,
    ToolChoice,
    ToolName,
)


def _meta() -> RunMetadata:
    return RunMetadata(
        run_id="abc123",
        latency_ms=12.5,
        provider="fake",
        cheap_model="fake-cheap",
        strong_model="fake-strong",
    )


def test_research_answer_roundtrip() -> None:
    answer = ResearchAnswer(
        question="What is X?",
        short_answer="Not found with high confidence.",
        key_findings=[Finding(claim="No independent sources located.", source_ids=[])],
        sources=[
            Source(
                id="s1",
                url="https://example.com/docs",
                title="Example",
                retrieved_at=datetime.now(timezone.utc),
            )
        ],
        confidence=ConfidenceLevel.LOW,
        confidence_rationale="Single domain.",
        assumptions_and_limitations=["Ambiguous product name."],
        next_steps=["Search official docs."],
        metadata=_meta(),
    )
    loaded = ResearchAnswer.model_validate_json(answer.model_dump_json())
    assert loaded.confidence is ConfidenceLevel.LOW
    assert loaded.sources[0].id == "s1"


def test_source_rejects_non_http_url() -> None:
    with pytest.raises(ValidationError):
        Source(id="s1", url="javascript:alert(1)", title="x", retrieved_at=datetime.now(timezone.utc))


def test_plan_requires_a_tool() -> None:
    with pytest.raises(ValidationError):
        SubQuestionPlan(id="q1", question="stars?", tools=[], sufficient_evidence="github stats")

    plan = ResearchPlan(
        assumptions=["Interpret 'top' as GitHub stars."],
        sub_questions=[
            SubQuestionPlan(
                id="q1",
                question="What are current GitHub stars?",
                tools=[ToolChoice(name=ToolName.GITHUB_REPO_STATS, reason="primary metric")],
                sufficient_evidence="stars + last commit from GitHub",
            )
        ],
        sufficient_evidence_overall="two independent domains",
    )
    assert plan.sub_questions[0].tools[0].name is ToolName.GITHUB_REPO_STATS
