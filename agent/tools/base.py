"""Shared tool contract: typed I/O, timeout, backoff, JSONL trace.

Retries live here so individual tools do not invent their own flake handling.
Non-retryable errors (SSRF, 404, bad input) fail fast — retrying those only
hides bugs and wastes the wall-clock budget.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from agent.tracing import Tracer

USER_AGENT = "research-agent/0.2 (+take-home assignment)"

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class ToolError(Exception):
    def __init__(self, reason: str, *, tool: str, retryable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.tool = tool
        self.retryable = retryable

    def __str__(self) -> str:
        kind = "retryable" if self.retryable else "fatal"
        return f"{self.tool} [{kind}]: {self.reason}"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ToolError):
        return exc.retryable
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    return False


class BaseTool(ABC, Generic[InT, OutT]):
    name: str
    timeout_s: float = 20.0
    max_attempts: int = 3

    def __init__(self, tracer: Tracer | None = None, timeout_s: float | None = None) -> None:
        self.tracer = tracer
        if timeout_s is not None:
            self.timeout_s = timeout_s

    @abstractmethod
    async def _execute(self, payload: InT) -> OutT:
        raise NotImplementedError

    def _preview(self, payload: InT) -> str:
        return payload.model_dump_json()

    async def run(self, payload: InT) -> OutT:
        started = time.perf_counter()
        error: str | None = None
        if self.tracer:
            self.tracer.totals.tool_calls += 1
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential(multiplier=0.4, min=0.4, max=6),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    return await asyncio.wait_for(self._execute(payload), timeout=self.timeout_s)
            raise ToolError("retry loop exited without result", tool=self.name)
        except ValidationError as exc:
            error = f"malformed tool output: {exc}"
            raise ToolError(error, tool=self.name, retryable=False) from exc
        except (TimeoutError, asyncio.TimeoutError) as exc:
            error = f"timed out after {self.timeout_s}s"
            raise ToolError(error, tool=self.name, retryable=True) from exc
        except ToolError as exc:
            error = exc.reason
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise ToolError(error, tool=self.name, retryable=_is_retryable(exc)) from exc
        finally:
            if self.tracer:
                self.tracer.log(
                    "tool",
                    tool=self.name,
                    input_preview=self._preview(payload),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=error,
                )


class EmptyInput(BaseModel):
    """Placeholder so tests can exercise BaseTool without a real payload."""

    note: str = ""


class EmptyOutput(BaseModel):
    ok: bool = True
    detail: str = ""
