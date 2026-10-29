import hashlib

import pytest

from agent.config import Settings
from agent.schemas import ResearchPlan
from agent.tracing import Tracer
from main import ReplayCache, TraceReplayLLM
from agent.cache import JsonCache
from agent.llm import LLMError


async def test_trace_replay_llm_returns_saved_output_deterministically(tmp_path) -> None:
    settings = Settings(llm_provider="fake", traces_dir=tmp_path / "traces")
    tracer = Tracer.create(settings.traces_dir, "replay-test")
    plan = ResearchPlan(assumptions=[], sub_questions=[{
        "id": "q1", "question": "q", "tools": [{"name": "web_search", "reason": "r"}],
        "sufficient_evidence": "s",
    }], sufficient_evidence_overall="s")
    key = hashlib.sha256("system\0user".encode("utf-8")).hexdigest()
    events = [{"step": "llm", "extra": {
        "schema": "ResearchPlan", "input_hash": key, "output": plan.model_dump(mode="json")
    }}]
    llm = TraceReplayLLM(settings, tracer, events)
    first = await llm.generate_structured("system", "user", ResearchPlan, "strong")
    assert first == plan
    second_llm = TraceReplayLLM(settings, tracer, events)
    second = await second_llm.generate_structured("system", "user", ResearchPlan, "strong")
    assert second == first
    with pytest.raises(LLMError, match="no saved output"):
        await llm.generate_structured("system", "different user", ResearchPlan, "strong")


def test_replay_cache_reports_missing_entries(tmp_path) -> None:
    replay_cache = ReplayCache(JsonCache(tmp_path / "empty", enabled=True))
    with pytest.raises(LLMError, match="replay cache entry missing"):
        replay_cache.get("web_search:q:5")
    assert replay_cache.missing == ["web_search:q:5"]
    replay_cache.close()
