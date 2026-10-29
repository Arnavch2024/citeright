from pathlib import Path

import pytest

from agent.safety import resolve_fixture_file
from agent.tools.base import ToolError
from agent.tools.fetch import FetchInput, FetchPageTool

FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures"


async def test_fixture_fetch_flags_injection() -> None:
    uri = (FIXTURES / "injection.html").resolve().as_uri()
    tool = FetchPageTool(timeout_s=5, jina_enabled=False, fixture_root=FIXTURES)
    out = await tool.run(FetchInput(url=uri))
    assert out.injection_flagged is True
    assert "acme" in out.text.lower() or "hnsw" in out.text.lower() or "Apache" in out.text


def test_fixture_escape_blocked(tmp_path) -> None:
    uri = (tmp_path / "secret.html").resolve().as_uri()
    with pytest.raises(ToolError):
        resolve_fixture_file(uri, FIXTURES)
