from __future__ import annotations

from rank_bm25 import BM25Okapi

from agent.retrieval.base import PassageSelector, split_heading_chunks, tokenize
from agent.schemas import Passage


class BM25Selector(PassageSelector):
    name = "bm25"

    def select_passages(
        self, question: str, page_text: str, k: int, source_id: str = ""
    ) -> list[Passage]:
        chunks = split_heading_chunks(page_text)
        if not chunks:
            return []
        corpus = [tokenize(body) for _, body in chunks]
        # Empty token lists make BM25 raise; keep a dummy token so indexing holds.
        corpus = [toks or ["_empty"] for toks in corpus]
        query = tokenize(question) or ["_empty"]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[Passage] = []
        for idx, score in ranked:
            heading, body = chunks[idx]
            out.append(
                Passage(
                    source_id=source_id,
                    text=body,
                    score=float(score),
                    heading=heading,
                )
            )
        return out
