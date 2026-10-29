"""Canonical contracts for the whole pipeline.

Keeping every stage on Pydantic models means the CLI, validator, traces, and
eval harness share one source of truth — and the LLM adapter can pass these
types straight into provider structured-output APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11; assignment targets 3.11+ but this machine may be 3.10
    class StrEnum(str, Enum):
        def __str__(self) -> str:  # pragma: no cover
            return str(self.value)


class ToolName(StrEnum):
    WEB_SEARCH = "web_search"
    FETCH_PAGE = "fetch_page"
    GITHUB_REPO_STATS = "github_repo_stats"
    SCHOLARLY_SEARCH = "scholarly_search"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DomainType(StrEnum):
    OFFICIAL_DOCS = "official_docs"
    REPO = "repo"
    PAPER = "paper"
    BLOG = "blog"
    FORUM = "forum"
    NEWS = "news"
    OTHER = "other"


class LLMTier(StrEnum):
    CHEAP = "cheap"
    STRONG = "strong"


class PipelineState(StrEnum):
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    CHECK = "CHECK"
    REPLAN = "REPLAN"
    SYNTHESIZE = "SYNTHESIZE"
    VALIDATE = "VALIDATE"


class Finding(BaseModel):
    """A single claim. source_ids may be empty only for explicit 'not found' claims."""

    claim: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class Source(BaseModel):
    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str
    retrieved_at: datetime
    domain_type: DomainType = DomainType.OTHER

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, value: str) -> str:
        # Stored as str so traces/JSON stay simple; still reject obvious junk.
        if not (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("file://")
        ):
            raise ValueError("url must be http(s) or a local file fixture")
        return value


class RunMetadata(BaseModel):
    run_id: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    replan_used: bool = False
    provider: str
    cheap_model: str
    strong_model: str
    errors: list[str] = Field(default_factory=list)
    cache_hits: int = 0
    low_evidence_flag: bool = False
    tool_path: list[str] = Field(default_factory=list)


class ResearchAnswer(BaseModel):
    question: str
    short_answer: str
    key_findings: list[Finding]
    sources: list[Source]
    confidence: ConfidenceLevel
    confidence_rationale: str
    assumptions_and_limitations: list[str]
    next_steps: list[str]
    metadata: RunMetadata


class ToolChoice(BaseModel):
    name: ToolName
    reason: str
    query_hint: str | None = None


class SubQuestionPlan(BaseModel):
    id: str
    question: str
    tools: list[ToolChoice] = Field(min_length=1)
    sufficient_evidence: str


class ResearchPlan(BaseModel):
    """PLAN output. Assumptions are required whenever the question is underspecified."""

    assumptions: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestionPlan] = Field(min_length=1)
    sufficient_evidence_overall: str


class SynthesisDraft(BaseModel):
    """LLM synthesis without a sources list — the orchestrator attaches retrieved sources only."""

    short_answer: str
    key_findings: list[Finding]
    confidence: ConfidenceLevel
    confidence_rationale: str
    assumptions_and_limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class RelevanceBatch(BaseModel):
    keep_urls: list[str] = Field(default_factory=list)


class Passage(BaseModel):
    source_id: str
    text: str
    score: float
    heading: str | None = None


class ScoredDocument(BaseModel):
    source_id: str
    url: str
    title: str
    domain: str
    domain_type: DomainType
    score: float
    retrieved_at: datetime
    text: str
    thin: bool = False
    injection_flagged: bool = False
    published_at: datetime | None = None


class LLMCallStats(BaseModel):
    provider: str
    model: str
    tier: LLMTier
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class ConfidenceSignals(BaseModel):
    """Code-side inputs. LLM rationale never raises this above the code floor."""

    independent_source_count: int
    subquestions_with_evidence: int
    subquestions_total: int
    tool_failures: int
    replan_used: bool
    injection_flagged: bool = False
    code_level: ConfidenceLevel
    llm_level: ConfidenceLevel
    final_level: ConfidenceLevel


class PartialErrorNote(BaseModel):
    stage: PipelineState
    message: str


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
