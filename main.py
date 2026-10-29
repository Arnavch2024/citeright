"""CLI: `research` runs the state machine; `check` verifies config + tool wiring."""

from __future__ import annotations

import asyncio
import json

import typer

from agent.config import load_settings
from agent.orchestrator import run_research
from agent.schemas import ResearchAnswer, ResearchPlan
from agent.tools.factory import build_tools
from agent.tracing import Tracer, new_run_id

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("research")
def research(
    question: str = typer.Argument(..., help="Research question"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable disk cache"),
) -> None:
    settings = load_settings()
    if no_cache:
        settings = settings.model_copy(update={"cache_enabled": False})
    answer = asyncio.run(run_research(question, settings))
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(f"**{answer.confidence}** — {answer.short_answer}"))
        Console().print(f"saved outputs/{answer.metadata.run_id}.json and .md")
    except Exception:
        typer.echo(answer.model_dump_json(indent=2))


@app.command("check")
def check() -> None:
    settings = load_settings()
    run_id = new_run_id()
    tracer = Tracer.create(settings.traces_dir, run_id)
    tracer.log("check", extra={"provider": settings.llm_provider})
    payload = {
        "ok": True,
        "phase": 3,
        "tools": list(build_tools(settings, tracer).keys()),
        "run_id": run_id,
        "settings": settings.redacted_dict(),
        "schemas": {
            "ResearchPlan": list(ResearchPlan.model_json_schema()["properties"].keys()),
            "ResearchAnswer": list(ResearchAnswer.model_json_schema()["properties"].keys()),
        },
        "trace": str(tracer.path),
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    app()
