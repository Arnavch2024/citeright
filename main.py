"""CLI: `research` runs the state machine; `check` verifies config + tool wiring."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

import typer

from agent.config import Settings, load_settings
from agent.cache import JsonCache
from agent.llm import LLMClient, LLMError, T
from agent.orchestrator import Orchestrator, run_research
from agent.schemas import ResearchAnswer, ResearchPlan, Source
from agent.tools.factory import build_tools
from agent.tracing import Tracer, new_run_id

app = typer.Typer(add_completion=False, no_args_is_help=True)


class ReplayCache:
    """Read-only cache facade that fails closed instead of making network calls."""

    def __init__(self, cache: JsonCache) -> None:
        self.cache = cache
        self.missing: list[str] = []

    def get(self, key: str) -> Any:
        value = self.cache.get(key)
        if value is None:
            self.missing.append(key)
            raise LLMError(f"replay cache entry missing: {key}")
        return value

    def set(self, key: str, value: Any, expire: int = 0) -> None:
        return None

    def close(self) -> None:
        self.cache.close()


class TraceReplayLLM(LLMClient):
    """Return recorded structured outputs in order, keyed by schema type."""

    def __init__(self, settings: Settings, tracer: Tracer, events: list[dict[str, Any]]) -> None:
        super().__init__(settings, tracer)
        self.outputs: dict[str, deque[Any]] = defaultdict(deque)
        self.by_input: dict[str, Any] = {}
        self.missing: list[str] = []
        for event in events:
            if event.get("step") == "llm" and event.get("extra", {}).get("output") is not None:
                extra = event["extra"]
                if extra.get("input_hash"):
                    self.by_input[extra["input_hash"]] = extra["output"]
                else:
                    self.outputs[extra["schema"]].append(extra["output"])

    async def _generate(self, system: str, user: str, schema: type[T], model: str) -> tuple[T, int, int]:
        key = hashlib.sha256(f"{system}\0{user}".encode("utf-8")).hexdigest()
        raw = self.by_input.get(key)
        if raw is None and self.outputs[schema.__name__]:
            raw = self.outputs[schema.__name__].popleft()  # older traces lack input hashes
        if raw is None:
            self.missing.append(schema.__name__)
            raise LLMError(f"trace has no saved output for schema {schema.__name__}")
        return schema.model_validate(raw), 0, 0


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


@app.command("replay")
def replay(run_id: str = typer.Argument(..., help="Run id whose JSONL trace should be replayed")) -> None:
    """Replay cached tools and recorded LLM outputs without network access."""
    settings = load_settings()
    trace_path = settings.traces_dir / f"{run_id}.jsonl"
    if not trace_path.exists():
        raise typer.BadParameter(f"trace not found: {trace_path}")
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    question_event = next((e for e in events if e.get("step") == "PLAN" and e.get("input_preview")), None)
    if question_event is None:
        raise typer.BadParameter("trace does not contain the original question")
    question = question_event.get("extra", {}).get("question") or question_event["input_preview"]
    replay_config = question_event.get("extra", {}).get("replay_config")
    if replay_config:
        settings = settings.model_copy(update=replay_config)
    sources_event = next((e for e in events if e.get("step") == "SOURCES"), None)
    saved_sources = (sources_event or {}).get("extra", {}).get("sources")
    if saved_sources is None:
        answer_path = settings.outputs_dir / f"{run_id}.json"
        if answer_path.exists():
            saved_sources = json.loads(answer_path.read_text(encoding="utf-8")).get("sources", [])
    source_metadata_map = {
        source["url"]: Source.model_validate(source) for source in saved_sources or []
    }
    reference_time_raw = question_event.get("extra", {}).get("reference_time")
    reference_time = datetime.fromisoformat(reference_time_raw) if reference_time_raw else None
    replay_tracer = Tracer.create(settings.traces_dir, f"replay-{run_id}-{new_run_id()}")
    replay_llm = TraceReplayLLM(settings, replay_tracer, events)
    cache = ReplayCache(JsonCache(settings.cache_dir, enabled=settings.cache_enabled))
    tools = build_tools(settings, replay_tracer, cache)  # type: ignore[arg-type]
    orch = Orchestrator(
        settings, llm=replay_llm, tools=tools, tracer=replay_tracer,
        source_metadata_map=source_metadata_map, reference_time=reference_time,
    )
    try:
        answer = asyncio.run(orch.run(question))
        if replay_llm.missing:
            raise typer.BadParameter("trace is missing saved LLM outputs for: " + ", ".join(replay_llm.missing))
        if cache.missing:
            missing = ", ".join(dict.fromkeys(cache.missing))
            raise typer.BadParameter(f"replay cache entries missing: {missing}")
        typer.echo(answer.model_dump_json(indent=2))
    except Exception as exc:
        raise typer.BadParameter(f"replay failed: {exc}") from exc
    finally:
        asyncio.run(orch.aclose())
        for tool in tools.values():
            close = getattr(tool, "aclose", None)
            if close:
                asyncio.run(close())
        cache.close()
        orch.cache.close()


if __name__ == "__main__":
    app()
