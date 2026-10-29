from agent.retrieval.base import build_selector, select_passages, split_heading_chunks

PAGE = """
# Installation
pip install qdrant-client and then create a collection.

# Benchmarks
Independent tests found Qdrant faster than Weaviate on filtered ANN at this scale.

# License
Apache 2.0 applies to the core server.
"""


def test_split_headings() -> None:
    chunks = split_heading_chunks(PAGE)
    headings = [h for h, _ in chunks]
    assert "Benchmarks" in headings


def test_bm25_ranks_relevant_heading() -> None:
    passages = select_passages("which engine is faster qdrant or weaviate", PAGE, k=2, source_id="s1")
    assert passages
    assert passages[0].source_id == "s1"
    assert "faster" in passages[0].text.lower() or passages[0].heading == "Benchmarks"


def test_embeddings_backend_returns_k() -> None:
    selector = build_selector("embeddings")
    passages = selector.select_passages("license apache", PAGE, k=1, source_id="s1")
    assert len(passages) == 1
    assert passages[0].score >= 0
