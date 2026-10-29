"""Explicit state machine: PLAN → EXECUTE → CHECK → REPLAN → SYNTHESIZE → VERIFY → VALIDATE."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from agent.budget import RunBudget
from agent.check import insufficient_subquestions
from agent.confidence import assess_confidence, rationale
from agent.conflicts import detect_conflicts
from agent.config import Settings
from agent.executor import ExecutionResult, Executor, evidence_for_prompt
from agent.executor import SubQuestionResult
from agent.llm import LLMClient, LLMError, build_llm
from agent.planner import plan_research, replan_research
from agent.report import render_markdown
from agent.schemas import (
    ConfidenceLevel,
    Conflict,
    Finding,
    PipelineState,
    ResearchAnswer,
    ResearchPlan,
    RunMetadata,
    Source,
    ScoredDocument,
    SynthesisDraft,
    utcnow,
)
from agent.scoring import select_sources
from agent.synthesizer import synthesize
from agent.tools.factory import build_cache, build_tools
from agent.tracing import Tracer, new_run_id
from agent.validator import repair_draft, sanitize_answer, validate_answer
from agent.verifier import verify_findings


def _sources_from_docs(docs: list[ScoredDocument]) -> list[Source]:
    return [
        Source(
            id=d.source_id,
            url=d.url,
            title=d.title,
            retrieved_at=d.retrieved_at,
            domain_type=d.domain_type,
        )
        for d in docs
    ]


def _evidence_by_source(
    results: list[SubQuestionResult], docs: list[ScoredDocument]
) -> dict[str, str]:
    """Map source_id -> passage text for the verifier. Fall back to doc text.

    The verifier sees only retrieved passages for cited source_ids, never a full
    page body or any source that was filtered out.
    """
    by_id = {d.source_id: d for d in docs}
    out: dict[str, str] = {}
    for part in results:
        for p in part.passages:
            if p.source_id not in by_id:
                continue
            prefix = (p.heading + ": ") if p.heading else ""
            chunk = prefix + p.text
            out[p.source_id] = out.get(p.source_id, "") + ("\n" if out.get(p.source_id) else "") + chunk
    return out


def _missing_subquestions(plan: ResearchPlan, exe: ExecutionResult) -> list[str]:
    """Sub-question text with no retrieved documents — for abstention notes."""
    missing: list[str] = []
    for sq, res in zip(plan.sub_questions, exe.results):
        if not res.documents:
            missing.append(sq.question)
    return missing


def _answer_from_draft(
    question: str,
    draft: SynthesisDraft,
    docs: list[ScoredDocument],
    meta: RunMetadata,
    extra_limits: list[str],
    *,
    verified_ratio: float = 0.0,
    conflicts: list[Conflict] | None = None,
) -> ResearchAnswer:
    limits = list(draft.assumptions_and_limitations) + extra_limits
    findings = draft.key_findings or [Finding(claim="Not found in retrieved sources.", source_ids=[])]
    return ResearchAnswer(
        question=question,
        short_answer=draft.short_answer or "Not found.",
        key_findings=findings,
        sources=_sources_from_docs(docs),
        confidence=draft.confidence,
        confidence_rationale=draft.confidence_rationale,
        assumptions_and_limitations=limits,
        next_steps=draft.next_steps or ["Gather additional primary sources."],
        metadata=meta,
        verified_ratio=verified_ratio,
        conflicts=conflicts or [],
    )


def save_outputs(answer: ResearchAnswer, outputs_dir: Path) -> tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / f"{answer.metadata.run_id}.json"
    md_path = outputs_dir / f"{answer.metadata.run_id}.md"
    json_path.write_text(answer.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(answer), encoding="utf-8")
    return json_path, md_path


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        tools: dict | None = None,
        tracer: Tracer | None = None,
        source_metadata_map: dict[str, Source] | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer or Tracer.create(settings.traces_dir)
        self.llm = llm or build_llm(settings, self.tracer)
        self.source_metadata_map = source_metadata_map or {}
        self.reference_time = reference_time
        self.cache = build_cache(settings)
        self.tools = tools if tools is not None else build_tools(settings, self.tracer, self.cache)
        self._owns_tools = tools is None

    def _verifier_llm(self) -> tuple[LLMClient, bool, str | None]:
        """Build the verifier LLM.

        Uses VERIFIER_PROVIDER if set and its key is available, so the verifier
        differs from the generator (reduces self-confirmation bias). Otherwise the
        generator's LLM is reused and verifier_same_provider=True is recorded.
        Returns (llm, same_provider, note).
        """
        vp = self.settings.verifier_provider
        if not vp or vp == self.settings.llm_provider:
            return self.llm, True, None
        try:
            v_settings = self.settings.model_copy(update={"llm_provider": vp})
            return build_llm(v_settings, self.tracer), False, None
        except LLMError as exc:
            # Other provider unavailable -> degrade to same provider, surface it.
            return self.llm, True, f"verifier provider {vp!r} unavailable ({exc.reason}); used the generator's provider"

    async def aclose(self) -> None:
        if self._owns_tools:
            for tool in self.tools.values():
                close = getattr(tool, "aclose", None)
                if close:
                    await close()
        self.cache.close()

    async def run(self, question: str) -> ResearchAnswer:
        started = time.perf_counter()
        reference_time = self.reference_time or utcnow()
        budget = RunBudget(self.settings)
        errors: list[str] = []
        replan_used = False
        low_evidence = False
        state = PipelineState.PLAN

        try:
            if self.tracer:
                self.tracer.log(
                    state.value,
                    input_preview=question,
                    extra={
                        "question": question,
                        "reference_time": reference_time.isoformat(),
                        "replay_config": {
                            "llm_provider": self.settings.llm_provider,
                            "openai_cheap_model": self.settings.openai_cheap_model,
                            "openai_strong_model": self.settings.openai_strong_model,
                            "gemini_cheap_model": self.settings.gemini_cheap_model,
                            "gemini_strong_model": self.settings.gemini_strong_model,
                            "max_replans": self.settings.max_replans,
                            "retrieval_backend": self.settings.retrieval_backend,
                            "passages_k": self.settings.passages_k,
                            "verifier_enabled": self.settings.verifier_enabled,
                            "verifier_provider": self.settings.verifier_provider,
                            "min_verified_ratio": self.settings.min_verified_ratio,
                            "min_coverage": self.settings.min_coverage,
                            "source_filter_enabled": self.settings.source_filter_enabled,
                        },
                    },
                )
            plan = await plan_research(self.llm, question)

            state = PipelineState.EXECUTE
            if self.tracer:
                self.tracer.log(state.value, extra={"sub_questions": len(plan.sub_questions)})
            executor = Executor(
                self.settings, self.tools, budget, self.llm, self.tracer,
                source_metadata_map=self.source_metadata_map,
                reference_time=reference_time,
            )
            exe = await executor.run_plan(plan)

            state = PipelineState.CHECK
            by_sub = {r.sub_id: r.documents for r in exe.results}
            gaps = insufficient_subquestions(by_sub)
            if self.tracer:
                self.tracer.log(state.value, extra={"gaps": gaps})

            if gaps and self.settings.max_replans >= 1:
                state = PipelineState.REPLAN
                replan_used = True
                if self.tracer:
                    self.tracer.log(state.value, extra={"gaps": gaps})
                try:
                    gap_labels = [
                        next((r.question for r in exe.results if r.sub_id == gid), gid) for gid in gaps
                    ]
                    plan2 = await replan_research(self.llm, question, plan, gap_labels)
                    exe2 = await executor.run_plan(plan2, start_i=exe.next_source_i)
                    exe = _merge_execution(exe, exe2, filter_sources=self.settings.source_filter_enabled)
                    plan = _merge_plans(plan, plan2)
                except LLMError as exc:
                    errors.append(f"replan failed: {exc.reason}")
                by_sub = {r.sub_id: r.documents for r in exe.results}
                if insufficient_subquestions(by_sub):
                    low_evidence = True
                    errors.append("low_evidence: some sub-questions still have <2 independent domains")

            tool_path = [name for part in exe.results for name in part.tools_used]
            extra_limits = list(plan.assumptions)
            extra_limits.extend(exe.failures)
            if any(d.injection_flagged for d in exe.documents):
                extra_limits.append("Fetched pages contained prompt-injection patterns; treated as data only.")
            if low_evidence:
                extra_limits.append("Evidence check failed after one replan; confidence is capped.")

            state = PipelineState.SYNTHESIZE
            allowed = [d.source_id for d in exe.documents]
            evidence = evidence_for_prompt(exe.results, exe.documents)
            if self.tracer:
                self.tracer.log(state.value, extra={"allowed_ids": allowed})
            draft = await synthesize(self.llm, question, plan, evidence, allowed, extra_limits)

            # VERIFY: check each claim against its cited passages (cheap, no tools).
            verifier_same_provider = False
            verifier_ran = False
            verified_ratio = 0.0
            claims_dropped = 0
            all_cited_findings_dropped = False
            evidence_by_source = _evidence_by_source(exe.results, exe.documents)
            if self.settings.verifier_enabled:
                state = PipelineState.VERIFY
                v_llm, verifier_same_provider, v_note = self._verifier_llm()
                if v_note:
                    extra_limits.append(v_note)
                if verifier_same_provider:
                    extra_limits.append(
                        "Verifier used the generator's provider (verifier_same_provider=true); provider independence is unavailable."
                    )
                if self.tracer:
                    self.tracer.log(
                        state.value,
                        extra={
                            "same_provider": verifier_same_provider,
                            "findings": len(draft.key_findings),
                            "verifiable": sum(1 for f in draft.key_findings if f.source_ids),
                        },
                    )
                outcome = await verify_findings(v_llm, draft.key_findings, evidence_by_source)
                had_cited_findings = any(f.source_ids for f in draft.key_findings)
                draft = draft.model_copy(update={"key_findings": outcome.findings_after})
                verifier_ran = outcome.verifier_ran
                verified_ratio = outcome.verified_ratio
                claims_dropped = outcome.claims_dropped
                all_cited_findings_dropped = (
                    had_cited_findings and not outcome.findings_after
                )
                if claims_dropped:
                    short_answer = (
                        "Not enough cited evidence to answer."
                        if all_cited_findings_dropped
                        else "Some claims were removed because their cited evidence did not support them."
                    )
                    draft = draft.model_copy(update={"short_answer": short_answer})
                if outcome.error:
                    extra_limits.append(f"verifier failed: {outcome.error}")
                if not outcome.findings_after:
                    extra_limits.append("All verifiable claims were dropped by verification.")
                    errors.append("partial_result: verifier dropped every cited finding")
                    low_evidence = True

            # CONFLICTS: surface disagreements across the surviving sources.
            conflicts = await detect_conflicts(self.llm, draft.key_findings, evidence_by_source)
            if conflicts:
                if self.tracer:
                    self.tracer.log("CONFLICTS", extra={"conflicts": len(conflicts)})
                extra_limits.append(
                    f"Conflicting evidence found in {len(conflicts)} topic(s); confidence lowered accordingly."
                )

            covered = sum(1 for r in exe.results if r.documents)
            signals = assess_confidence(
                docs=exe.documents,
                subquestions_with_evidence=covered,
                subquestions_total=max(1, len(plan.sub_questions)),
                tool_failures=len(exe.failures),
                replan_used=replan_used,
                llm_level=draft.confidence,
                verified_ratio=verified_ratio,
                claims_dropped=claims_dropped,
                conflicts_found=len(conflicts),
                verifier_ran=verifier_ran,
                min_verified_ratio=self.settings.min_verified_ratio,
                min_coverage=self.settings.min_coverage,
            )
            draft = draft.model_copy(
                update={
                    "confidence": signals.final_level,
                    "confidence_rationale": f"{draft.confidence_rationale} | {rationale(signals)}",
                }
            )

            # WHY: one note ties a LOW result to the missing evidence and claim attrition.
            missing_subs = _missing_subquestions(plan, exe)
            coverage = covered / max(1, len(plan.sub_questions))
            ratio_failed = verifier_ran and verified_ratio < self.settings.min_verified_ratio
            coverage_failed = coverage < self.settings.min_coverage
            if ratio_failed or coverage_failed:
                extra_limits.append(
                    f"Abstaining: verified ratio {verified_ratio:.0%} "
                    f"(minimum {self.settings.min_verified_ratio:.0%}); coverage {coverage:.0%} "
                    f"(minimum {self.settings.min_coverage:.0%}); sub-questions lacking sources: "
                    f"{'; '.join(missing_subs) if missing_subs else 'none'}; "
                    f"{claims_dropped} claim(s) dropped."
                )

            meta = _meta(
                self,
                started,
                replan_used,
                low_evidence,
                errors,
                budget,
                tool_path,
                claims_dropped=claims_dropped,
                conflicts_found=len(conflicts),
                verifier_same_provider=verifier_same_provider,
                verifier_ran=verifier_ran,
                verified_ratio=verified_ratio,
            )
            answer = _answer_from_draft(
                question, draft, exe.documents, meta, extra_limits,
                verified_ratio=verified_ratio, conflicts=conflicts,
            )
            if all_cited_findings_dropped:
                # WHY: reuse the established last-resort sanitizer when no grounded claim survives.
                retrieved_ids = set(exe.retrieved_ids) | {d.source_id for d in exe.documents}
                retrieved_urls = set(exe.retrieved_urls) | {d.url for d in exe.documents}
                answer = sanitize_answer(answer, retrieved_ids, retrieved_urls).model_copy(
                    update={
                        "confidence": ConfidenceLevel.LOW,
                        "assumptions_and_limitations": answer.assumptions_and_limitations
                        + ["Returned a partial result after verification removed every cited claim."],
                    }
                )

            state = PipelineState.VALIDATE
            retrieved_ids = set(exe.retrieved_ids) | {d.source_id for d in exe.documents}
            retrieved_urls = set(exe.retrieved_urls) | {d.url for d in exe.documents}
            problems = validate_answer(answer, retrieved_ids, retrieved_urls)
            if problems:
                if self.tracer:
                    self.tracer.log(state.value, error="; ".join(problems))
                try:
                    draft2 = await repair_draft(self.llm, draft, problems, allowed)
                    draft2 = draft2.model_copy(
                        update={
                            "confidence": min_conf(draft2.confidence, signals.final_level),
                        }
                    )
                    answer = _answer_from_draft(
                        question, draft2, exe.documents, meta, extra_limits,
                        verified_ratio=verified_ratio, conflicts=conflicts,
                    )
                    problems = validate_answer(answer, retrieved_ids, retrieved_urls)
                except LLMError as exc:
                    errors.append(f"repair failed: {exc.reason}")
                if problems:
                    answer = sanitize_answer(answer, retrieved_ids, retrieved_urls)
                    errors.append("partial_result: " + "; ".join(problems))
                    answer = answer.model_copy(
                        update={
                            "confidence": ConfidenceLevel.LOW,
                            "metadata": _meta(
                                self, started, replan_used, True, errors, budget, tool_path,
                                claims_dropped=claims_dropped, conflicts_found=len(conflicts),
                                verifier_same_provider=verifier_same_provider,
                                verifier_ran=verifier_ran,
                                verified_ratio=verified_ratio,
                            ),
                            "assumptions_and_limitations": answer.assumptions_and_limitations
                            + ["Returned a partial result after validation failed."],
                        }
                    )
            else:
                answer = answer.model_copy(
                    update={
                        "metadata": _meta(
                            self, started, replan_used, low_evidence, errors, budget, tool_path,
                            claims_dropped=claims_dropped, conflicts_found=len(conflicts),
                            verifier_same_provider=verifier_same_provider,
                            verifier_ran=verifier_ran,
                            verified_ratio=verified_ratio,
                        )
                    }
                )

            if self.tracer:
                self.tracer.log(
                    "SOURCES",
                    extra={"sources": [source.model_dump(mode="json") for source in answer.sources]},
                )
            save_outputs(answer, self.settings.outputs_dir)
            return answer
        except Exception as exc:
            errors.append(f"{state.value}: {exc}")
            if self.tracer:
                self.tracer.log(state.value, error=str(exc))
            meta = _meta(self, started, replan_used, True, errors, budget, [])
            answer = ResearchAnswer(
                question=question,
                short_answer="Not found — the pipeline failed before a grounded answer could be produced.",
                key_findings=[Finding(claim="Not found due to pipeline error.", source_ids=[])],
                sources=[],
                confidence=ConfidenceLevel.LOW,
                confidence_rationale="; ".join(errors),
                assumptions_and_limitations=errors,
                next_steps=["Retry with cache enabled or a different provider."],
                metadata=meta,
            )
            save_outputs(answer, self.settings.outputs_dir)
            return answer


def min_conf(a: ConfidenceLevel, b: ConfidenceLevel) -> ConfidenceLevel:
    from agent.confidence import min_level

    return min_level(a, b)


def _merge_execution(
    a: ExecutionResult, b: ExecutionResult, *, filter_sources: bool = True
) -> ExecutionResult:
    combined = a.documents + b.documents
    docs = select_sources(combined) if filter_sources else combined
    return ExecutionResult(
        results=a.results + b.results,
        documents=docs,
        retrieved_ids=a.retrieved_ids | b.retrieved_ids,
        retrieved_urls=a.retrieved_urls | b.retrieved_urls,
        failures=a.failures + b.failures,
        next_source_i=b.next_source_i,
    )


def _merge_plans(a: ResearchPlan, b: ResearchPlan) -> ResearchPlan:
    return ResearchPlan(
        assumptions=list(dict.fromkeys(a.assumptions + b.assumptions)),
        sub_questions=a.sub_questions + b.sub_questions,
        sufficient_evidence_overall=b.sufficient_evidence_overall or a.sufficient_evidence_overall,
    )


def _meta(
    orch: Orchestrator,
    started: float,
    replan_used: bool,
    low_evidence: bool,
    errors: list[str],
    budget: RunBudget,
    tool_path: list[str],
    *,
    claims_dropped: int = 0,
    conflicts_found: int = 0,
    verifier_same_provider: bool = False,
    verifier_ran: bool = False,
    verified_ratio: float = 0.0,
) -> RunMetadata:
    totals = orch.tracer.totals if orch.tracer else None
    return RunMetadata(
        run_id=orch.tracer.run_id if orch.tracer else new_run_id(),
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=totals.input_tokens if totals else 0,
        output_tokens=totals.output_tokens if totals else 0,
        tool_calls=budget.tool_calls,
        replan_used=replan_used,
        provider=orch.settings.llm_provider,
        cheap_model=orch.settings.cheap_model(),
        strong_model=orch.settings.strong_model(),
        errors=list(errors),
        cache_hits=totals.cache_hits if totals else 0,
        low_evidence_flag=low_evidence,
        tool_path=tool_path,
        claims_dropped=claims_dropped,
        conflicts_found=conflicts_found,
        verifier_same_provider=verifier_same_provider,
        verifier_ran=verifier_ran,
        verified_ratio=verified_ratio,
    )


async def run_research(question: str, settings: Settings | None = None) -> ResearchAnswer:
    from agent.config import load_settings

    settings = settings or load_settings()
    orch = Orchestrator(settings)
    try:
        return await orch.run(question)
    finally:
        await orch.aclose()
