import pytest

from agent.safety import assert_public_http_url
from agent.tools.base import ToolError


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com/x",
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://2130706433/",  # 127.0.0.1 as an integer host
    ],
)
def test_ssrf_guard_blocks(url: str) -> None:
    with pytest.raises(ToolError) as exc:
        assert_public_http_url(url)
    assert exc.value.retryable is False


def test_ssrf_blocks_dns_to_private(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host, *args, **kwargs):
        return [(0, 0, 0, "", ("10.1.2.3", 0))]

    monkeypatch.setattr("agent.safety.socket.getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ToolError, match="non-public"):
        assert_public_http_url("http://evil.example/")


def test_ssrf_allows_public_dns(public_dns) -> None:
    assert_public_http_url("https://docs.qdrant.tech/overview/")
