"""Fetch + extract. SSRF first, then trafilatura, then Jina Reader if empty."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin

import httpx
import trafilatura
from pydantic import BaseModel, Field

from agent.cache import JsonCache
from agent.safety import assert_public_http_url, resolve_fixture_file, scan_injection, strip_injection_markers
from agent.tools.base import USER_AGENT, BaseTool, ToolError
from agent.tracing import Tracer

MAX_CHARS = 20_000
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5
JINA_PREFIX = "https://r.jina.ai/"


class FetchInput(BaseModel):
    url: str = Field(min_length=8)


class FetchOutput(BaseModel):
    url: str
    final_url: str
    title: str
    text: str
    truncated: bool = False
    injection_flagged: bool = False
    injection_patterns: list[str] = Field(default_factory=list)
    extractor: str = "trafilatura"


def extract_markdown(html: str) -> tuple[str, str]:
    """Return (title, markdown). Tables stay as markdown via trafilatura."""
    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title if metadata and metadata.title else "") or ""
    text = trafilatura.extract(
        html,
        include_tables=True,
        include_comments=False,
        output_format="markdown",
        favor_recall=True,
    ) or ""
    return title.strip(), text.strip()


def _truncate(text: str, limit: int = MAX_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n[truncated]", True


class FetchPageTool(BaseTool[FetchInput, FetchOutput]):
    name = "fetch_page"
    max_attempts = 2

    def __init__(
        self,
        tracer: Tracer | None = None,
        timeout_s: float | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        cache: JsonCache | None = None,
        extract_fn: Callable[[str], tuple[str, str]] | None = None,
        jina_enabled: bool = True,
        fixture_root: Path | None = None,
    ) -> None:
        super().__init__(tracer=tracer, timeout_s=timeout_s)
        self._owns_client = client is None
        # follow_redirects=False: we validate every hop against the SSRF guard.
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s or self.timeout_s,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
            max_redirects=0,
        )
        self.cache = cache
        self.extract_fn = extract_fn or extract_markdown
        self.jina_enabled = jina_enabled
        self.fixture_root = fixture_root

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _execute(self, payload: FetchInput) -> FetchOutput:
        url = str(payload.url)
        if url.startswith("file:"):
            if self.fixture_root is None:
                raise ToolError("file URLs are disabled", tool=self.name, retryable=False)
            path = resolve_fixture_file(url, self.fixture_root)
            html = path.read_text(encoding="utf-8", errors="replace")
            return self._from_text(url, url, html, extractor="fixture")
        assert_public_http_url(url, tool=self.name)
        cache_key = f"fetch_page:{url}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.tracer:
                    self.tracer.totals.cache_hits += 1
                return FetchOutput.model_validate(cached)

        html, final_url = await self._get_html(url)
        title, text = self.extract_fn(html)
        extractor = "trafilatura"
        if not text and self.jina_enabled:
            text = await self._jina(url)
            extractor = "jina"
            if not title:
                title = final_url
        if not text:
            raise ToolError("empty extraction", tool=self.name, retryable=False)
        return self._from_text(
            url, final_url, text, extractor=extractor, already_extracted=True, title=title
        )

    def _from_text(
        self,
        url: str,
        final_url: str,
        raw: str,
        *,
        extractor: str,
        already_extracted: bool = False,
        title: str = "",
    ) -> FetchOutput:
        if already_extracted:
            text = raw
        else:
            title, text = self.extract_fn(raw)
        if not text:
            raise ToolError("empty extraction", tool=self.name, retryable=False)
        text = strip_injection_markers(text)
        report = scan_injection(text)
        if report.flagged and self.tracer:
            self.tracer.log(
                "safety",
                tool=self.name,
                input_preview=url,
                extra={"patterns": report.patterns},
            )
        text, truncated = _truncate(text)
        out = FetchOutput(
            url=url,
            final_url=final_url,
            title=title or final_url,
            text=text,
            truncated=truncated,
            injection_flagged=report.flagged,
            injection_patterns=report.patterns,
            extractor=extractor,
        )
        if self.cache:
            self.cache.set(f"fetch_page:{url}", out.model_dump(mode="json"))
        return out

    async def _get_html(self, url: str) -> tuple[str, str]:
        current = url
        for _ in range(MAX_REDIRECTS):
            assert_public_http_url(current, tool=self.name)
            response = await self._client.get(current)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ToolError("redirect without Location", tool=self.name, retryable=False)
                current = urljoin(current, location)
                continue
            if response.status_code >= 400:
                retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                raise ToolError(
                    f"HTTP {response.status_code} for {current}",
                    tool=self.name,
                    retryable=retryable,
                )
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text/" not in content_type and content_type:
                # Still try: many docs servers omit a precise type.
                pass
            raw = response.content[: MAX_BYTES + 1]
            if len(raw) > MAX_BYTES:
                raise ToolError("response too large", tool=self.name, retryable=False)
            return raw.decode(response.encoding or "utf-8", errors="replace"), str(response.url)
        raise ToolError("too many redirects", tool=self.name, retryable=False)

    async def _jina(self, url: str) -> str:
        # Jina fetches on their infra; still refuse private targets so we don't
        # launder SSRF through a proxy.
        assert_public_http_url(url, tool=self.name)
        response = await self._client.get(JINA_PREFIX + url)
        if response.status_code >= 400:
            return ""
        return response.text.strip()
