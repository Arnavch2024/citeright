from agent.config import Settings
from agent.tools.factory import build_tools


def test_build_tools_names(tmp_path) -> None:
    settings = Settings(cache_dir=tmp_path / "c", cache_enabled=False)
    tools = build_tools(settings)
    assert set(tools) == {"web_search", "fetch_page", "github_repo_stats", "scholarly_search"}
