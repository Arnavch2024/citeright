from __future__ import annotations

import pytest


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map every hostname to a public address so unit tests never hit real DNS."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(0, 0, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr("agent.safety.socket.getaddrinfo", fake_getaddrinfo)
