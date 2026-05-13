"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

Dense embeddings capture semantic similarity well but routinely miss
queries that hinge on rare technical terms, named entities, or exact
phrases — *"BLEU 28.4"*, *"Adam beta_2"*, *"NIST 08 newstest"*. BM25 does
the opposite: strong on lexical matches, weak on paraphrase.

This module adds a thin in-memory BM25 index that pulls its corpus from
``VectorStore.all_chunks()``. At query time we run dense and BM25 in
parallel, then fuse the two ranked lists with Reciprocal Rank Fusion. RRF
is parameter-free in the sense that it doesn't need a tuned weight per
retriever — only a fixed dampening constant ``k`` (literature default 60).

The BM25 index caches its tokenized corpus and is invalidated on every
PDF upload / delete so retrievals stay consistent with the dense store.
For PaperPal's scale (≤ a few thousand chunks) the rebuild is sub-second.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger
from rank_bm25 import BM25Okapi

from .store import Retrieval

if TYPE_CHECKING:
    from .store import VectorStore


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index built lazily from a VectorStore's chunks.

    Call :meth:`invalidate` whenever the underlying corpus changes (PDF
    upload, delete). The next :meth:`query` will rebuild.
    """

    def __init__(self, store: "VectorStore") -> None:
        self._store = store
        self._bm25: BM25Okapi | None = None
        self._meta: list[tuple[str, int, int, str]] = []  # paper_id, page, chunk_idx, text

    def invalidate(self) -> None:
        self._bm25 = None
        self._meta = []

    def _rebuild(self) -> None:
        docs, metas = self._store.all_chunks()
        tokenized: list[list[str]] = []
        meta_rows: list[tuple[str, int, int, str]] = []
        for doc, meta in zip(docs, metas, strict=True):
            tokenized.append(_tokenize(doc))
            meta_rows.append(
                (str(meta["paper_id"]), int(meta["page"]), int(meta["chunk_idx"]), doc)
            )
        self._meta = meta_rows
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info(f"BM25 index rebuilt over {len(self._meta)} chunks")

    def query(
        self,
        question: str,
        *,
        top_k: int,
        paper_ids: list[str] | None = None,
    ) -> list[Retrieval]:
        if self._bm25 is None:
            self._rebuild()
        if self._bm25 is None or not self._meta:
            return []

        scores = self._bm25.get_scores(_tokenize(question))
        indexed = list(enumerate(scores))
        if paper_ids:
            wanted = set(paper_ids)
            indexed = [(i, s) for i, s in indexed if self._meta[i][0] in wanted]
        indexed.sort(key=lambda x: float(x[1]), reverse=True)

        out: list[Retrieval] = []
        for i, score in indexed[:top_k]:
            paper_id, page, chunk_idx, text = self._meta[i]
            out.append(
                Retrieval(
                    paper_id=paper_id,
                    page=page,
                    chunk_idx=chunk_idx,
                    text=text,
                    score=float(score),
                )
            )
        return out


def rrf_fuse(
    rankings: list[list[Retrieval]],
    *,
    top_k: int,
    k: int = 60,
) -> list[Retrieval]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    For each document d and ranking r, contributes ``1 / (k + rank(d, r))``
    to d's fused score. ``k`` dampens the contribution of high ranks;
    Cormack et al. (2009) recommend 60 as a sane default that doesn't
    need tuning per dataset.

    The fused score replaces the per-retriever score on the returned
    Retrieval objects so downstream code (logging, the SSE payload) sees
    a single coherent "relevance" number.
    """
    accumulator: dict[str, tuple[Retrieval, float]] = {}
    for ranking in rankings:
        for rank, r in enumerate(ranking):
            chunk_id = f"{r.paper_id}:{r.page}:{r.chunk_idx}"
            increment = 1.0 / (k + rank + 1)
            if chunk_id in accumulator:
                stored, score = accumulator[chunk_id]
                accumulator[chunk_id] = (stored, score + increment)
            else:
                accumulator[chunk_id] = (r, increment)

    fused = sorted(accumulator.values(), key=lambda x: x[1], reverse=True)
    return [
        Retrieval(
            paper_id=r.paper_id,
            page=r.page,
            chunk_idx=r.chunk_idx,
            text=r.text,
            score=score,
        )
        for r, score in fused[:top_k]
    ]
