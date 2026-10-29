"""PLAN / REPLAN. Strong tier, structured. No tools — it only names tools for the executor."""

from __future__ import annotations

from agent.llm import LLMClient
from agent.schemas import ResearchPlan

SYSTEM = """You are the planner for a research agent. You do not browse the web.
Return a structured plan only.

Tools you may name:
- web_search: open web (companies, comparisons, recent facts). Always pair with fetch_page.
- fetch_page: retrieve a known URL (docs, blog). Use after search or when the URL is known.
- github_repo_stats: GitHub stars/license/issues/release. query_hint must be owner/name or a github.com URL.
- scholarly_search: papers (arXiv / Semantic Scholar). Use for architecture, methods, academic claims.

Rules:
- 2–4 sub-questions. Different question types MUST use different tool paths
  (repos → github_repo_stats; papers → scholarly_search; market/startups → web_search+fetch_page).
- If the user question is ambiguous, write explicit assumptions rather than guessing silently.
- Prefer primary sources (official docs, repos, papers) in sufficient_evidence.
- query_hint is the concrete query or URL/repo the executor should pass to the tool.
"""


async def plan_research(llm: LLMClient, question: str, *, extra: str = "") -> ResearchPlan:
    user = f"Research question:\n{question}"
    if extra:
        user += f"\n\nAdditional planner context:\n{extra}"
    return await llm.generate_structured(SYSTEM, user, ResearchPlan, "strong")


async def replan_research(
    llm: LLMClient,
    question: str,
    prior: ResearchPlan,
    gaps: list[str],
) -> ResearchPlan:
    extra = (
        "Previous plan JSON:\n"
        f"{prior.model_dump_json()}\n\n"
        "These sub-questions lacked >=2 independent-domain sources:\n"
        + "\n".join(f"- {g}" for g in gaps)
        + "\nPlan ONLY replacement sub-questions that close those gaps. "
        "Switch tools if the first path failed (e.g. add github_repo_stats or scholarly_search)."
    )
    return await plan_research(llm, question, extra=extra)
