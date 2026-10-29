"""Single-shot LLM, no tools. Comparison control for the agentic pipeline."""

from __future__ import annotations

import time

from agent.llm import LLMClient
from agent.schemas import Finding, ResearchAnswer, RunMetadata, SynthesisDraft

SYSTEM = """Answer the research question from parameters only. You have NO tools and NO sources.
If you are unsure, say not found. Do not invent URLs. Structured output only.
"""


async def run_baseline(llm: LLMClient, question: str, *, provider: str, cheap: str, strong: str) -> ResearchAnswer:
    started = time.perf_counter()
    draft = await llm.generate_structured(SYSTEM, question, SynthesisDraft, "strong")
    latency = (time.perf_counter() - started) * 1000
    stats = llm.last_call
    return ResearchAnswer(
        question=question,
        short_answer=draft.short_answer,
        key_findings=draft.key_findings or [Finding(claim="Not found.", source_ids=[])],
        sources=[],
        confidence=draft.confidence,
        confidence_rationale=draft.confidence_rationale + " | baseline (no tools)",
        assumptions_and_limitations=draft.assumptions_and_limitations + ["Baseline: no retrieval."],
        next_steps=draft.next_steps,
        metadata=RunMetadata(
            run_id="baseline",
            latency_ms=latency,
            input_tokens=stats.input_tokens if stats else 0,
            output_tokens=stats.output_tokens if stats else 0,
            tool_calls=0,
            provider=provider,
            cheap_model=cheap,
            strong_model=strong,
            tool_path=[],
        ),
    )
