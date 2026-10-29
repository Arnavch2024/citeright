"""Deterministic source ranking. An LLM must not score its own evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from agent.schemas import DomainType, ScoredDocument, utcnow

THIN_CHARS = 280

TYPE_WEIGHT = {
    DomainType.OFFICIAL_DOCS: 1.00,
    DomainType.REPO: 0.92,
    DomainType.PAPER: 0.90,
    DomainType.NEWS: 0.68,
    DomainType.BLOG: 0.52,
    DomainType.FORUM: 0.38,
    DomainType.OTHER: 0.45,
}

_DOCS_HOST_HINTS = ("docs.", "developer.", "dev.", "api.", "learn.")
_DOCS_PATH_HINTS = ("/docs", "/documentation", "/api/", "/reference")
_BLOG_HINTS = ("blog.", "medium.com", "substack.com", "dev.to", "hashnode")
_FORUM_HINTS = ("reddit.com", "news.ycombinator.com", "stackoverflow.com", "stackexchange.com", "discourse.")
_NEWS_HINTS = ("techcrunch.com", "theverge.com", "wired.com", "arstechnica.com", "bbc.", "nytimes.com")
_PAPER_HINTS = ("arxiv.org", "aclanthology.org", "semanticscholar.org", "doi.org", "acm.org", "ieee.org")
_REPO_HINTS = ("github.com", "gitlab.com", "bitbucket.org")


def registrable_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def classify_domain(url: str) -> DomainType:
    host = registrable_host(url)
    path = urlparse(url).path.lower()
    if any(h in host for h in _REPO_HINTS):
        return DomainType.REPO
    if any(h in host for h in _PAPER_HINTS):
        return DomainType.PAPER
    if any(host.startswith(h) or h in host for h in _DOCS_HOST_HINTS) or any(p in path for p in _DOCS_PATH_HINTS):
        return DomainType.OFFICIAL_DOCS
    if any(h in host for h in _FORUM_HINTS):
        return DomainType.FORUM
    if any(h in host for h in _NEWS_HINTS):
        return DomainType.NEWS
    if any(h in host for h in _BLOG_HINTS) or "/blog" in path:
        return DomainType.BLOG
    return DomainType.OTHER


def _recency_factor(published_at: datetime | None) -> float:
    """Unknown dates are slightly discounted; stale pages more so. Never zero — a 2019 RFC still matters."""
    if published_at is None:
        return 0.85
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    days = max(0, (utcnow() - published_at).days)
    if days <= 180:
        return 1.0
    if days <= 365:
        return 0.90
    if days <= 365 * 3:
        return 0.75
    return 0.55


def _length_factor(n: int) -> tuple[float, bool]:
    if n < THIN_CHARS:
        return 0.0, True
    if n < 800:
        return 0.65, False
    return 1.0, False


def _dedupe_key(doc: ScoredDocument) -> str:
    # WHY: collapsing all of github.com would hide a 3-way repo comparison.
    host = doc.domain
    if doc.domain_type == DomainType.REPO:
        parts = urlparse(doc.url).path.strip("/").split("/")
        if len(parts) >= 2:
            return f"repo:{host}/{parts[0]}/{parts[1]}"
    return f"host:{host}"


def score_document(
    *,
    source_id: str,
    url: str,
    title: str,
    text: str,
    injection_flagged: bool = False,
    published_at: datetime | None = None,
) -> ScoredDocument:
    domain = registrable_host(url)
    dtype = classify_domain(url)
    length_f, thin = _length_factor(len(text.strip()))
    score = TYPE_WEIGHT[dtype] * length_f * _recency_factor(published_at)
    if injection_flagged:
        score *= 0.5
    return ScoredDocument(
        source_id=source_id,
        url=url,
        title=title or domain,
        domain=domain,
        domain_type=dtype,
        score=score,
        retrieved_at=utcnow(),
        text=text,
        thin=thin,
        injection_flagged=injection_flagged,
        published_at=published_at,
    )


def select_sources(docs: list[ScoredDocument], *, max_keep: int = 12) -> list[ScoredDocument]:
    """Drop thin pages, keep the best page per domain key, prefer primary sources."""
    usable = [d for d in docs if not d.thin and d.score > 0]
    usable.sort(key=lambda d: d.score, reverse=True)
    seen: set[str] = set()
    picked: list[ScoredDocument] = []
    for doc in usable:
        key = _dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        picked.append(doc)
        if len(picked) >= max_keep:
            break
    return picked
