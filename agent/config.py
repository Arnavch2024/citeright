"""Runtime caps and secrets. Loaded from the environment, not hardcoded.

Hard caps exist so a stuck search/LLM call cannot burn budget. Model names are
env-driven because providers rotate IDs faster than this assignment's code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


ProviderName = Literal["openai", "gemini", "fake"]
RetrievalBackend = Literal["bm25", "embeddings"]


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    llm_provider: ProviderName = "openai"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_cheap_model: str = "gpt-4o-mini"
    openai_strong_model: str = "gpt-4o"
    gemini_cheap_model: str = "gemini-2.5-flash"
    gemini_strong_model: str = "gemini-2.5-pro"

    exa_api_key: str | None = None
    tavily_api_key: str | None = None
    github_token: str | None = None
    semantic_scholar_api_key: str | None = None

    max_tool_calls: int = Field(default=20, ge=1)
    max_replans: int = Field(default=1, ge=0, le=1)
    max_tokens_per_call: int = Field(default=4096, ge=256)
    per_call_timeout_s: float = Field(default=45.0, gt=0)
    total_wall_clock_s: float = Field(default=180.0, gt=0)
    tool_timeout_s: float = Field(default=20.0, gt=0)

    cache_enabled: bool = True
    cache_dir: Path = Path(".cache")

    retrieval_backend: RetrievalBackend = "bm25"
    passages_k: int = Field(default=5, ge=1, le=20)

    # Trust layer: claim verification, conflict detection, abstention thresholds.
    verifier_enabled: bool = True
    verifier_provider: str | None = None  # different provider than the generator, if set
    min_verified_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    min_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    source_filter_enabled: bool = True  # False => --no-filter skips scoring/dedup

    traces_dir: Path = Path("traces")
    outputs_dir: Path = Path("outputs")
    fixtures_dir: Path = Path("eval/fixtures")

    # Rough USD / 1M tokens for eval cost — Gemini Flash is ~this; Pro is higher.
    cheap_input_usd_per_m: float = 0.10
    cheap_output_usd_per_m: float = 0.40
    strong_input_usd_per_m: float = 1.25
    strong_output_usd_per_m: float = 10.00

    def model_for_tier(self, tier: str) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_cheap_model if tier == "cheap" else self.gemini_strong_model
        if self.llm_provider == "fake":
            return f"fake-{tier}"
        return self.openai_cheap_model if tier == "cheap" else self.openai_strong_model

    def cheap_model(self) -> str:
        return self.model_for_tier("cheap")

    def strong_model(self) -> str:
        return self.model_for_tier("strong")

    def redacted_dict(self) -> dict[str, object]:
        """Safe to print: secrets become bool presence flags."""
        data = self.model_dump(mode="json")
        for key in (
            "openai_api_key",
            "gemini_api_key",
            "exa_api_key",
            "tavily_api_key",
            "github_token",
            "semantic_scholar_api_key",
        ):
            data[key] = bool(data.get(key))
        return data


def load_settings(env_file: Path | None = None) -> Settings:
    # WHY: dotenv is optional in CI; tests pass env vars directly.
    load_dotenv(env_file or Path(".env"), override=False)
    provider = _env("LLM_PROVIDER", "openai")
    if provider not in {"openai", "gemini", "fake"}:
        raise ValueError(f"LLM_PROVIDER must be openai|gemini|fake, got {provider!r}")
    retrieval = _env("RETRIEVAL_BACKEND", "bm25")
    if retrieval not in {"bm25", "embeddings"}:
        raise ValueError(f"RETRIEVAL_BACKEND must be bm25|embeddings, got {retrieval!r}")
    vprovider = _env("VERIFIER_PROVIDER", "")
    if vprovider and vprovider not in {"openai", "gemini", "fake"}:
        raise ValueError(f"VERIFIER_PROVIDER must be openai|gemini|fake, got {vprovider!r}")
    return Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        openai_api_key=_env("OPENAI_API_KEY") or None,
        gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY") or None,
        openai_cheap_model=_env("OPENAI_CHEAP_MODEL", "gpt-4o-mini"),
        openai_strong_model=_env("OPENAI_STRONG_MODEL", "gpt-4o"),
        gemini_cheap_model=_env("GEMINI_CHEAP_MODEL", "gemini-2.5-flash"),
        gemini_strong_model=_env("GEMINI_STRONG_MODEL", "gemini-2.5-pro"),
        exa_api_key=_env("EXA_API_KEY") or None,
        tavily_api_key=_env("TAVILY_API_KEY") or None,
        github_token=_env("GITHUB_TOKEN") or None,
        semantic_scholar_api_key=_env("SEMANTIC_SCHOLAR_API_KEY") or None,
        max_tool_calls=_env_int("MAX_TOOL_CALLS", 20),
        max_replans=_env_int("MAX_REPLANS", 1),
        max_tokens_per_call=_env_int("MAX_TOKENS_PER_CALL", 4096),
        per_call_timeout_s=_env_float("PER_CALL_TIMEOUT_S", 45.0),
        total_wall_clock_s=_env_float("TOTAL_WALL_CLOCK_S", 180.0),
        tool_timeout_s=_env_float("TOOL_TIMEOUT_S", 20.0),
        cache_enabled=_env_bool("CACHE_ENABLED", True),
        cache_dir=Path(_env("CACHE_DIR", ".cache")),
        retrieval_backend=retrieval,  # type: ignore[arg-type]
        passages_k=_env_int("PASSAGES_K", 5),
        verifier_enabled=_env_bool("VERIFIER_ENABLED", True),
        verifier_provider=_env("VERIFIER_PROVIDER") or None,
        min_verified_ratio=_env_float("MIN_VERIFIED_RATIO", 0.7),
        min_coverage=_env_float("MIN_COVERAGE", 0.5),
        source_filter_enabled=_env_bool("SOURCE_FILTER_ENABLED", True),
        traces_dir=Path(_env("TRACES_DIR", "traces")),
        outputs_dir=Path(_env("OUTPUTS_DIR", "outputs")),
        fixtures_dir=Path(_env("FIXTURES_DIR", "eval/fixtures")),
    )
