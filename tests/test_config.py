from agent.config import load_settings


def test_load_settings_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("MAX_TOOL_CALLS", "7")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.llm_provider == "fake"
    assert settings.max_tool_calls == 7
    assert settings.cache_enabled is False
    assert settings.redacted_dict()["openai_api_key"] is True
    assert "sk-test" not in str(settings.redacted_dict())


def test_model_for_tier(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_CHEAP_MODEL", "flash-test")
    monkeypatch.setenv("GEMINI_STRONG_MODEL", "pro-test")
    settings = load_settings(env_file=tmp_path / "none.env")
    assert settings.model_for_tier("cheap") == "flash-test"
    assert settings.model_for_tier("strong") == "pro-test"
