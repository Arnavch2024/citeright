"""JSONL run traces. One file per run_id so evals can grep a single research job.

Inputs are truncated: full page HTML in a trace is both noisy and a secret-leak
risk (cookies, tokens in fetched pages).
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from agent.schemas import utcnow


TRACE_PREVIEW_CHARS = 800


class TraceEvent(BaseModel):
    run_id: str
    step: str
    ts: str
    tool: str | None = None
    model: str | None = None
    input_preview: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def truncate(text: str | None, limit: int = TRACE_PREVIEW_CHARS) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + f"…[+{len(compact) - limit} chars]"


@dataclass
class TokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    cache_hits: int = 0

    def add_llm(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out
        self.llm_calls += 1

    def estimate_usd(self, cheap_in: float, cheap_out: float, strong_in: float, strong_out: float, *, strong: bool) -> float:
        # Eval helper: caller says whether this batch was cheap or strong.
        inp_rate = strong_in if strong else cheap_in
        out_rate = strong_out if strong else cheap_out
        return (self.input_tokens / 1_000_000) * inp_rate + (self.output_tokens / 1_000_000) * out_rate


@dataclass
class Tracer:
    run_id: str
    path: Path
    totals: TokenTotals = field(default_factory=TokenTotals)

    @classmethod
    def create(cls, traces_dir: Path, run_id: str | None = None) -> "Tracer":
        traces_dir.mkdir(parents=True, exist_ok=True)
        rid = run_id or new_run_id()
        return cls(run_id=rid, path=traces_dir / f"{rid}.jsonl")

    def log(
        self,
        step: str,
        *,
        tool: str | None = None,
        model: str | None = None,
        input_preview: str | None = None,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            run_id=self.run_id,
            step=step,
            ts=utcnow().isoformat(),
            tool=tool,
            model=model,
            input_preview=truncate(input_preview),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            extra=extra or {},
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    @contextmanager
    def span(self, step: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Time a block and log once, including any exception message."""
        started = time.perf_counter()
        bag: dict[str, Any] = {}
        error: str | None = None
        try:
            yield bag
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            self.log(
                step,
                latency_ms=latency_ms,
                error=error,
                input_preview=bag.get("input_preview", kwargs.get("input_preview")),
                tool=kwargs.get("tool"),
                model=kwargs.get("model") or bag.get("model"),
                input_tokens=bag.get("input_tokens"),
                output_tokens=bag.get("output_tokens"),
                extra={k: v for k, v in {**kwargs, **bag}.items() if k not in {"input_preview", "model", "input_tokens", "output_tokens"}},
            )
