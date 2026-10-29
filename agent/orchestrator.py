"""Explicit state machine: PLAN → EXECUTE → CHECK → (REPLAN) → SYNTHESIZE → VALIDATE."""

from __future__ import annotations

import time
from pathlib import Path

from agent.budget import RunBudget
from agent.check import insufficient_subquestions
from agent.confidence import assess_confidence, rationale
from agent.config import Settings
from agent.executor import ExecutionResult, Executor, evidence_for_prompt
from agent.llm import LLMClient, LLMError, build_llm
from agent.planner import plan_research, replan_research
from agent.report import render_markdown
from agent.schemas import (
    ConfidenceLevel,
    Finding,
    PipelineState,
    ResearchAnswer,
    ResearchPlan,
    RunMetadata,
    Source,
    SynthesisDraft,
)
from agent.scoring import select_sources
from agent.synthesizer import synthesize
from agent.tools.factory import build_cache, build_tools
from agent.tracing import Tracer, new_run_id
from agent.validator import repair_draft, sanitize_answer, validate_answer


def _sources_from_docs(docs) -> list[Source]:
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


def _answer_from_draft(
    question: str,
    draft: SynthesisDraft,
    docs,
    meta: RunMetadata,
    extra_limits: list[str],
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
    ) -> None:
        self.settings = settings
        self.tracer = tracer or Tracer.create(settings.traces_dir)
        self.llm = llm or build_llm(settings, self.tracer)
        self.cache = build_cache(settings)
        self.tools = tools if tools is not None else build_tools(settings, self.tracer, self.cache)
        self._owns_tools = tools is None

    async def aclose(self) -> None:
        if self._owns_tools:
            for tool in self.tools.values():
                close = getattr(tool, "aclose", None)
                if close:
                    await close()
        self.cache.close()

    async def run(self, question: str) -> ResearchAnswer:
        started = time.perf_counter()
        budget = RunBudget(self.settings)
        errors: list[str] = []
        replan_used = False
        low_evidence = False
        state = PipelineState.PLAN

        try:
            if self.tracer:
                self.tracer.log(state.value, input_preview=question)
            plan = await plan_research(self.llm, question)

            state = PipelineState.EXECUTE
            if self.tracer:
                self.tracer.log(state.value, extra={"sub_questions": len(plan.sub_questions)})
            executor = Executor(self.settings, self.tools, budget, self.llm, self.tracer)
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
                    exe = _merge_execution(exe, exe2)
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

            covered = sum(1 for r in exe.results if r.documents)
            signals = assess_confidence(
                docs=exe.documents,
                subquestions_with_evidence=covered,
                subquestions_total=max(1, len(plan.sub_questions)),
                tool_failures=len(exe.failures),
                replan_used=replan_used,
                llm_level=draft.confidence,
            )
            draft = draft.model_copy(
                update={
                    "confidence": signals.final_level,
                    "confidence_rationale": f"{draft.confidence_rationale} | {rationale(signals)}",
                }
            )

            meta = _meta(self, started, replan_used, low_evidence, errors, budget, tool_path)
            answer = _answer_from_draft(question, draft, exe.documents, meta, extra_limits)

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
                    answer = _answer_from_draft(question, draft2, exe.documents, meta, extra_limits)
                    problems = validate_answer(answer, retrieved_ids, retrieved_urls)
                except LLMError as exc:
                    errors.append(f"repair failed: {exc.reason}")
                if problems:
                    answer = sanitize_answer(answer, retrieved_ids, retrieved_urls)
                    errors.append("partial_result: " + "; ".join(problems))
                    answer = answer.model_copy(
                        update={
                            "confidence": ConfidenceLevel.LOW,
                            "metadata": _meta(self, started, replan_used, True, errors, budget, tool_path),
                            "assumptions_and_limitations": answer.assumptions_and_limitations
                            + ["Returned a partial result after validation failed."],
                        }
                    )
            else:
                answer = answer.model_copy(
                    update={"metadata": _meta(self, started, replan_used, low_evidence, errors, budget, tool_path)}
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


def _merge_execution(a: ExecutionResult, b: ExecutionResult) -> ExecutionResult:
    docs = select_sources(a.documents + b.documents)
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
    )


async def run_research(question: str, settings: Settings | None = None) -> ResearchAnswer:
    from agent.config import load_settings

    settings = settings or load_settings()
    orch = Orchestrator(settings)
    try:
        return await orch.run(question)
    finally:
        await orch.aclose()
