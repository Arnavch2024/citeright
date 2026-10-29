"""In-memory hashed n-gram embeddings + cosine similarity.

No model download: evals can swap this in via RETRIEVAL_BACKEND=embeddings
and compare against BM25 without pulling a 100MB checkpoint.
"""

from __future__ import annotations

import hashlib

import numpy as np

from agent.retrieval.base import PassageSelector, split_heading_chunks, tokenize
from agent.schemas import Passage

DIM = 256


def _embed(tokens: list[str]) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float64)
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class HashEmbeddingSelector(PassageSelector):
    name = "embeddings"

    def select_passages(
        self, question: str, page_text: str, k: int, source_id: str = ""
    ) -> list[Passage]:
        chunks = split_heading_chunks(page_text)
        if not chunks:
            return []
        q = _embed(tokenize(question))
        scored: list[tuple[int, float]] = []
        for i, (_, body) in enumerate(chunks):
            scored.append((i, _cosine(q, _embed(tokenize(body)))))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[Passage] = []
        for idx, score in scored[:k]:
            heading, body = chunks[idx]
            out.append(Passage(source_id=source_id, text=body, score=score, heading=heading))
        return out
