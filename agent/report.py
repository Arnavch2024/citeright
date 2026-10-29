"""Render a ResearchAnswer to Markdown. JSON is the source of truth; this is for humans."""

from __future__ import annotations

from agent.schemas import ResearchAnswer


def render_markdown(answer: ResearchAnswer) -> str:
    findings = "\n".join(
        f"- {f.claim} ({', '.join(f.source_ids) or 'no source'})" for f in answer.key_findings
    )
    sources = "\n".join(f"- `{s.id}` [{s.title}]({s.url})" for s in answer.sources)
    limits = "\n".join(f"- {x}" for x in answer.assumptions_and_limitations) or "- (none)"
    next_steps = "\n".join(f"- {x}" for x in answer.next_steps) or "- (none)"
    meta = answer.metadata
    return f"""# Research report

**Question:** {answer.question}

**Short answer:** {answer.short_answer}

## Key findings

{findings or '- (none)'}

## Sources

{sources or '- (none)'}

## Confidence

**{answer.confidence}** — {answer.confidence_rationale}

## Assumptions and limitations

{limits}

## Next steps

{next_steps}

## Run metadata

- run_id: `{meta.run_id}`
- latency_ms: {meta.latency_ms:.0f}
- tokens: {meta.input_tokens} in / {meta.output_tokens} out
- tool_calls: {meta.tool_calls}
- replan_used: {meta.replan_used}
- cache_hits: {meta.cache_hits}
- errors: {'; '.join(meta.errors) or '(none)'}
"""
