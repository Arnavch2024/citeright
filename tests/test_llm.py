import pytest
from pydantic import BaseModel

from agent.config import Settings
from agent.llm import FakeLLM, LLMError, build_llm
from agent.tracing import Tracer


class ToySchema(BaseModel):
    label: str
    n: int


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake")


async def test_fake_llm_returns_typed_model_and_traces(settings, tmp_path) -> None:
    tracer = Tracer.create(tmp_path, "run1")
    llm = FakeLLM(settings, tracer)
    llm.set("ToySchema", ToySchema(label="ok", n=2))
    result = await llm.generate_structured("sys", "user question", ToySchema, "cheap")
    assert result == ToySchema(label="ok", n=2)
    assert llm.last_call is not None
    assert llm.last_call.input_tokens > 0
    lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "ToySchema" in lines[0]


async def test_fake_llm_missing_canned_raises(settings) -> None:
    llm = FakeLLM(settings)
    with pytest.raises(LLMError):
        await llm.generate_structured("s", "u", ToySchema, "strong")


def test_build_llm_fake(settings) -> None:
    client = build_llm(settings)
    assert isinstance(client, FakeLLM)


def test_build_openai_requires_key() -> None:
    with pytest.raises(LLMError):
        build_llm(Settings(llm_provider="openai", openai_api_key=None))


def test_build_gemini_requires_key() -> None:
    with pytest.raises(LLMError):
        build_llm(Settings(llm_provider="gemini", gemini_api_key=None))
