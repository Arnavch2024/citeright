"""Provider-agnostic structured generation.

We call native parse helpers (OpenAI Responses `text_format`, Gemini
`response_schema`) so the schema is enforced by the provider, not by regexing
JSON out of prose. Tool-use / chat.parse is a fallback for older OpenAI SDKs.

Gemini replaces Anthropic: same structured-output contract, free AI Studio
tier, Flash for extraction and Pro for planning/synthesis.

Cheap vs strong is a routing concern, not a prompt trick: extraction and
relevance filtering should not spend a frontier model.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from agent.config import Settings
from agent.schemas import LLMCallStats, LLMTier
from agent.tracing import Tracer

T = TypeVar("T", bound=BaseModel)
Tier = Literal["cheap", "strong"]


class LLMError(Exception):
    """Structured generation failed after provider/SDK handling."""

    def __init__(self, reason: str, *, retryable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


def _usage_tokens(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    inp = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or getattr(usage, "prompt_token_count", None)
        or 0
    )
    out = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or getattr(usage, "candidates_token_count", None)
        or 0
    )
    return int(inp), int(out)


class LLMClient(ABC):
    last_call: LLMCallStats | None = None

    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        self.settings = settings
        self.tracer = tracer

    @abstractmethod
    async def _generate(self, system: str, user: str, schema: type[T], model: str) -> tuple[T, int, int]:
        raise NotImplementedError

    async def generate_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        tier: Tier,
    ) -> T:
        model = self.settings.model_for_tier(tier)
        started = time.perf_counter()
        error: str | None = None
        inp = out = 0
        try:
            parsed, inp, out = await asyncio.wait_for(
                self._generate(system, user, schema, model),
                timeout=self.settings.per_call_timeout_s,
            )
            return parsed
        except (TimeoutError, asyncio.TimeoutError) as exc:
            error = f"timeout after {self.settings.per_call_timeout_s}s"
            raise LLMError(error, retryable=True) from exc
        except LLMError as exc:
            error = exc.reason
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise LLMError(error, retryable=True) from exc
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            self.last_call = LLMCallStats(
                provider=self.settings.llm_provider,
                model=model,
                tier=LLMTier(tier),
                latency_ms=latency_ms,
                input_tokens=inp,
                output_tokens=out,
            )
            if self.tracer:
                self.tracer.totals.add_llm(inp, out)
                self.tracer.log(
                    "llm",
                    model=model,
                    input_preview=f"tier={tier} schema={schema.__name__} user={user}",
                    latency_ms=latency_ms,
                    input_tokens=inp,
                    output_tokens=out,
                    error=error,
                    extra={"tier": tier, "schema": schema.__name__},
                )


class OpenAIAdapter(LLMClient):
    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        super().__init__(settings, tracer)
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is missing")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.per_call_timeout_s,
        )

    async def _generate(self, system: str, user: str, schema: type[T], model: str) -> tuple[T, int, int]:
        # Prefer Responses.parse (current SDK). Fall back to chat.completions.parse.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        parse = getattr(self._client.responses, "parse", None)
        if callable(parse):
            response = await parse(
                model=model,
                input=messages,
                text_format=schema,
                max_output_tokens=self.settings.max_tokens_per_call,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise LLMError("OpenAI returned no parsed output (refusal or empty)")
            return parsed, *_usage_tokens(getattr(response, "usage", None))

        chat_parse = getattr(self._client.chat.completions, "parse", None)
        if callable(chat_parse):
            completion = await chat_parse(
                model=model,
                messages=messages,
                response_format=schema,
                max_tokens=self.settings.max_tokens_per_call,
            )
            message = completion.choices[0].message
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                raise LLMError(getattr(message, "refusal", None) or "OpenAI chat parse returned empty")
            return parsed, *_usage_tokens(getattr(completion, "usage", None))

        # Last resort: force a function call whose arguments ARE the schema.
        return await self._tool_use_structured(system, user, schema, model)

    async def _tool_use_structured(
        self, system: str, user: str, schema: type[T], model: str
    ) -> tuple[T, int, int]:
        tool_name = f"emit_{schema.__name__}"
        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Return a {schema.__name__} object.",
                        "parameters": schema.model_json_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            max_tokens=self.settings.max_tokens_per_call,
        )
        choice = response.choices[0].message
        calls = choice.tool_calls or []
        if not calls:
            raise LLMError("OpenAI tool-use fallback produced no tool call")
        payload = json.loads(calls[0].function.arguments)
        return schema.model_validate(payload), *_usage_tokens(getattr(response, "usage", None))


class GeminiAdapter(LLMClient):
    """Google AI Studio / Gemini. Free tier; native Pydantic response_schema."""

    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        super().__init__(settings, tracer)
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing")
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def _generate(self, system: str, user: str, schema: type[T], model: str) -> tuple[T, int, int]:
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user,
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "max_output_tokens": self.settings.max_tokens_per_call,
            },
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None and getattr(response, "text", None):
            parsed = schema.model_validate_json(response.text)
        if parsed is None:
            raise LLMError("Gemini returned no parsed structured output")
        if not isinstance(parsed, schema):
            parsed = schema.model_validate(parsed)
        usage = getattr(response, "usage_metadata", None)
        return parsed, *_usage_tokens(usage)


class FakeLLM(LLMClient):
    """Deterministic backend for tests. Maps schema class name -> canned model or factory."""

    def __init__(
        self,
        settings: Settings,
        tracer: Tracer | None = None,
        canned: dict[str, BaseModel | Callable[[str, str, type[BaseModel]], BaseModel]] | None = None,
    ) -> None:
        super().__init__(settings, tracer)
        self.canned = canned or {}

    def set(self, schema_name: str, value: BaseModel | Callable[..., BaseModel]) -> None:
        self.canned[schema_name] = value

    async def _generate(self, system: str, user: str, schema: type[T], model: str) -> tuple[T, int, int]:
        item = self.canned.get(schema.__name__)
        if item is None:
            raise LLMError(f"FakeLLM has no canned response for {schema.__name__}")
        raw = item(system, user, schema) if callable(item) else item
        parsed = schema.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
        # Token counts are synthetic so tests can still assert tracing.
        approx = max(1, (len(system) + len(user)) // 4)
        return parsed, approx, max(1, len(parsed.model_dump_json()) // 4)


def build_llm(settings: Settings, tracer: Tracer | None = None, fake: FakeLLM | None = None) -> LLMClient:
    if settings.llm_provider == "fake":
        return fake or FakeLLM(settings, tracer)
    if settings.llm_provider == "gemini":
        return GeminiAdapter(settings, tracer)
    if settings.llm_provider == "openai":
        return OpenAIAdapter(settings, tracer)
    raise LLMError(f"Unknown provider {settings.llm_provider}")


def validate_or_raise(schema: type[T], data: Any) -> T:
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise LLMError(f"schema validation failed: {exc}") from exc
