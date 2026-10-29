from __future__ import annotations

import httpx

from agent.tools.scholarly import ScholarlyInput, ScholarlySearchTool

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Multi-Agent Customer Support</title>
    <published>2024-01-01T00:00:00Z</published>
    <summary>A paper about multi-agent architectures.</summary>
  </entry>
</feed>
"""


async def test_scholarly_falls_back_to_arxiv() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "semanticscholar.org" in request.url.host:
            return httpx.Response(503, text="down")
        return httpx.Response(200, text=ARXIV_XML)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = ScholarlySearchTool(timeout_s=5, client=client)
    out = await tool.run(ScholarlyInput(query="multi-agent customer support"))
    assert out.provider_used == "arxiv"
    assert out.papers[0].title.startswith("Multi-Agent")
    assert "arxiv.org" in out.papers[0].url
    await client.aclose()
