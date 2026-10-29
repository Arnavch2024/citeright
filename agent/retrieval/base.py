"""Passage selection without a vector DB.

Default is BM25 over heading chunks. Embeddings exist only so evals can
compare the two; they are hashed n-grams, not a downloaded model.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from agent.config import RetrievalBackend
from agent.schemas import Passage

HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def split_heading_chunks(page_text: str) -> list[tuple[str | None, str]]:
    """Split markdown-ish text on ATX headings; fall back to paragraphs."""
    text = page_text.strip()
    if not text:
        return []
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [(None, p) for p in paras] or [(None, text)]
    chunks: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            chunks.append((None, preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        heading = match.group(2).strip()
        chunk = f"{heading}\n{body}".strip()
        chunks.append((heading, chunk))
    return chunks


class PassageSelector(ABC):
    name: str

    @abstractmethod
    def select_passages(
        self, question: str, page_text: str, k: int, source_id: str = ""
    ) -> list[Passage]:
        raise NotImplementedError


def build_selector(backend: RetrievalBackend = "bm25") -> PassageSelector:
    if backend == "embeddings":
        from agent.retrieval.embeddings import HashEmbeddingSelector

        return HashEmbeddingSelector()
    from agent.retrieval.bm25 import BM25Selector

    return BM25Selector()


def select_passages(
    question: str,
    page_text: str,
    k: int,
    source_id: str = "",
    backend: RetrievalBackend = "bm25",
) -> list[Passage]:
    return build_selector(backend).select_passages(question, page_text, k, source_id=source_id)
