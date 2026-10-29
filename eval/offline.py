"""Deterministic LLM + tools so evals run without API keys and stay reproducible."""

from __future__ import annotations

import re

from agent.config import Settings
from agent.llm import FakeLLM
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
from tests.fakes import FakeFetch, FakeGithub, FakeScholarly, FakeSearch


def _file_uri(text: str) -> str | None:
    match = re.search(r"file:[^\s]+", text)
    return match.group(0).rstrip(".,)") if match else None


class EvalFakeLLM(FakeLLM):
    def __init__(self, settings: Settings, tracer=None, *, mode: str = "agent") -> None:
        super().__init__(settings, tracer)
        self.mode = mode

    async def _generate(self, system: str, user: str, schema, model: str):
        name = schema.__name__
        if name == "ResearchPlan":
            parsed = self._plan(user)
        elif name == "SynthesisDraft":
            parsed = self._baseline_draft(user) if self.mode == "baseline" else self._agent_draft(user)
        elif name == "RelevanceBatch":
            parsed = schema(keep_urls=[])
        else:
            from agent.llm import LLMError

            raise LLMError(f"EvalFakeLLM cannot produce {name}")
        approx = max(1, (len(system) + len(user)) // 4)
        return parsed, approx, max(1, len(parsed.model_dump_json()) // 4)

    def _plan(self, user: str) -> ResearchPlan:
        q = user.lower()
        uri = _file_uri(user)
        if "previous plan" in q or "lacked >=" in q or "close those gaps" in q:
            return ResearchPlan(
                assumptions=["Replan: add a second independent domain."],
                sub_questions=[
                    SubQuestionPlan(
                        id="q_replan",
                        question="Corroborate from GitHub",
                        tools=[
                            ToolChoice(
                                name=ToolName.GITHUB_REPO_STATS,
                                reason="second domain",
                                query_hint="milvus-io/milvus",
                            )
                        ],
                        sufficient_evidence="a second org's repo",
                    )
                ],
                sufficient_evidence_overall="two orgs",
            )
        if uri or "fixture" in q or "injection" in q:
            hint = uri or "file://eval/fixtures/injection.html"
            return ResearchPlan(
                assumptions=["The file URL is an eval fixture, not a public website."],
                sub_questions=[
                    SubQuestionPlan(
                        id="q1",
                        question="What factual claims are on the fixture page?",
                        tools=[ToolChoice(name=ToolName.FETCH_PAGE, reason="local fixture", query_hint=hint)],
                        sufficient_evidence="extracted fixture text",
                    )
                ],
                sufficient_evidence_overall="the fixture plus injection flag",
            )
        if "zephyrnimbus" in q or "mongoose" in q:
            return ResearchPlan(
                assumptions=["The named company may not exist."],
                sub_questions=[
                    SubQuestionPlan(
                        id="q1",
                        question="Does ZephyrNimbus HR appear in public sources?",
                        tools=[ToolChoice(name=ToolName.WEB_SEARCH, reason="existence check", query_hint="ZephyrNimbus HR Bangalore")],
                        sufficient_evidence="two independent corroborating sources",
                    )
                ],
                sufficient_evidence_overall="independent press or filings",
            )
        if "switching" in q or "is it worth" in q:
            return ResearchPlan(
                assumptions=[
                    "The user did not name the current stack or the destination.",
                    "Interpret as 'should a RAG startup switch vector databases' unless corrected.",
                ],
                sub_questions=[
                    SubQuestionPlan(
                        id="q1",
                        question="What are switching costs among OSS vector databases?",
                        tools=[ToolChoice(name=ToolName.WEB_SEARCH, reason="docs", query_hint="qdrant vs milvus migration")],
                        sufficient_evidence="official migration docs from two vendors",
                    )
                ],
                sufficient_evidence_overall="two vendor docs",
            )
        if "multi-agent" in q or "customer support" in q:
            return ResearchPlan(
                assumptions=[],
                sub_questions=[
                    SubQuestionPlan(
                        id="q1",
                        question="What do papers say about multi-agent support systems?",
                        tools=[
                            ToolChoice(
                                name=ToolName.SCHOLARLY_SEARCH,
                                reason="architecture evidence",
                                query_hint="multi-agent customer support",
                            )
                        ],
                        sufficient_evidence="peer-reviewed or arXiv papers",
                    )
                ],
                sufficient_evidence_overall="scholarly sources",
            )
        if "hr tech" in q or "saas" in q:
            return ResearchPlan(
                assumptions=["India-headquartered B2B HR products."],
                sub_questions=[
                    SubQuestionPlan(
                        id="q1",
                        question="Which Indian HR tech SaaS firms are publicly described?",
                        tools=[ToolChoice(name=ToolName.WEB_SEARCH, reason="market", query_hint="Indian HR tech SaaS Darwinbox Keka")],
                        sufficient_evidence="company sites",
                    )
                ],
                sufficient_evidence_overall="company primary pages",
            )
        tools = [
            ToolChoice(name=ToolName.WEB_SEARCH, reason="docs", query_hint="qdrant milvus weaviate RAG"),
            ToolChoice(name=ToolName.GITHUB_REPO_STATS, reason="stars", query_hint="qdrant/qdrant"),
        ]
        return ResearchPlan(
            assumptions=["Top-3 means widely used OSS engines: Qdrant, Milvus, Weaviate."],
            sub_questions=[
                SubQuestionPlan(
                    id="q1",
                    question="Docs and repo stats for leading OSS vector databases",
                    tools=tools,
                    sufficient_evidence="official docs + GitHub",
                )
            ],
            sufficient_evidence_overall="docs and github from 2+ orgs",
        )

    def _allowed(self, user: str) -> list[str]:
        match = re.search(r"Allowed source_ids \(only these exist\): ([^\n]+)", user)
        if not match or match.group(1).strip() == "(none)":
            return []
        return [p.strip() for p in match.group(1).split(",") if p.strip() and p.strip() != "(none)"]

    def _agent_draft(self, user: str) -> SynthesisDraft:
        ids = self._allowed(user)
        q = user.lower()
        limits: list[str] = []
        # The orchestrator appends a "prompt-injection patterns" note to the
        # known-limitations block when it actually flags a page, and injection
        # text inside the evidence carries "ignore previous". We must NOT match
        # on "untrusted" — that word is always present via the <<UNTRUSTED_*>> delimiters.
        if "injection" in q or "ignore previous" in q:
            limits.append("Fetched content contained prompt-injection patterns; treated as data only.")
        if not ids:
            return SynthesisDraft(
                short_answer="Not found in retrieved sources.",
                key_findings=[Finding(claim="No independent public sources were retrieved for this claim.", source_ids=[])],
                confidence=ConfidenceLevel.LOW,
                confidence_rationale="empty evidence set",
                assumptions_and_limitations=limits or ["Do not invent entities or numbers."],
                next_steps=["Search official registries or ask the user for a canonical URL."],
            )
        sid = ids[0]
        if "zephyrnimbus" in q:
            return SynthesisDraft(
                short_answer="Not found.",
                key_findings=[Finding(claim="No retrieved source confirms ZephyrNimbus HR or a Series B.", source_ids=[])],
                confidence=ConfidenceLevel.LOW,
                confidence_rationale="no corroboration",
                assumptions_and_limitations=limits,
                next_steps=["Treat the entity as unverified."],
            )
        return SynthesisDraft(
            short_answer="Grounded in retrieved sources only.",
            key_findings=[Finding(claim="Retrieved primary-looking pages support a cautious comparison.", source_ids=[sid])],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_rationale="limited but non-empty evidence",
            assumptions_and_limitations=limits,
            next_steps=["Fetch remaining vendor docs."],
        )

    def _baseline_draft(self, user: str) -> SynthesisDraft:
        # Deliberately overconfident: this is the no-tools control.
        return SynthesisDraft(
            short_answer="Parametric guess without retrieval.",
            key_findings=[
                Finding(
                    claim="ZephyrNimbus HR raised a $80M Series B in Bangalore, and Qdrant is always the best RAG store.",
                    source_ids=[],
                )
            ],
            confidence=ConfidenceLevel.HIGH,
            confidence_rationale="memorized priors, no tools",
            assumptions_and_limitations=["No sources retrieved."],
            next_steps=["Use the agentic pipeline."],
        )


def offline_tools(empty_search: bool = False) -> dict:
    hits = [] if empty_search else [
        SearchHit(title="Qdrant docs", url="https://qdrant.tech/documentation/", snippet="vector db", provider="fake"),
        SearchHit(title="Milvus docs", url="https://milvus.io/docs", snippet="vector db", provider="fake"),
    ]
    return {
        "web_search": FakeSearch(hits),
        "fetch_page": FakeFetch(),
        "github_repo_stats": FakeGithub(),
        "scholarly_search": FakeScholarly(),
    }
