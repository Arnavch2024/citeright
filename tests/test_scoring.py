from datetime import datetime, timezone, timedelta

from agent.scoring import classify_domain, score_document, select_sources
from agent.schemas import DomainType


def test_classify_primary_sources() -> None:
    assert classify_domain("https://qdrant.tech/documentation/overview/") is DomainType.OFFICIAL_DOCS
    assert classify_domain("https://github.com/milvus-io/milvus") is DomainType.REPO
    assert classify_domain("https://arxiv.org/abs/2401.00001") is DomainType.PAPER
    assert classify_domain("https://medium.com/@someone/post") is DomainType.BLOG
    assert classify_domain("https://news.ycombinator.com/item?id=1") is DomainType.FORUM


def test_drop_thin_and_dedupe_blogs() -> None:
    thin = score_document(source_id="s0", url="https://example.com/x", title="x", text="tiny")
    blog_a = score_document(
        source_id="s1",
        url="https://medium.com/@a/one",
        title="a",
        text="B" * 900,
    )
    blog_b = score_document(
        source_id="s2",
        url="https://medium.com/@b/two",
        title="b",
        text="C" * 4000,
    )
    docs = score_document(
        source_id="s3",
        url="https://docs.weaviate.io/weaviate",
        title="docs",
        text="D" * 4000,
    )
    kept = select_sources([thin, blog_a, blog_b, docs])
    ids = {d.source_id for d in kept}
    assert "s0" not in ids
    assert "s3" in ids
    # One medium.com page only — the longer/higher-scoring one.
    assert ("s1" in ids) ^ ("s2" in ids)


def test_github_repos_not_collapsed() -> None:
    a = score_document(source_id="g1", url="https://github.com/qdrant/qdrant", title="q", text="x" * 500)
    b = score_document(source_id="g2", url="https://github.com/milvus-io/milvus", title="m", text="x" * 500)
    kept = select_sources([a, b])
    assert {d.source_id for d in kept} == {"g1", "g2"}


def test_docs_beat_blog() -> None:
    docs = score_document(source_id="d", url="https://docs.foo.dev/guide", title="d", text="x" * 1000)
    blog = score_document(source_id="b", url="https://medium.com/@x/p", title="b", text="x" * 1000)
    assert docs.score > blog.score


def test_recency_boosts_fresh_pages() -> None:
    now = datetime.now(timezone.utc)
    fresh = score_document(
        source_id="n",
        url="https://example.com/post",
        title="n",
        text="x" * 1000,
        published_at=now - timedelta(days=10),
    )
    stale = score_document(
        source_id="o",
        url="https://example.com/old",
        title="o",
        text="x" * 1000,
        published_at=now - timedelta(days=2000),
    )
    assert fresh.score > stale.score
