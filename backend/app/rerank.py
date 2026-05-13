"""Cross-encoder reranker for retrieved chunks.

Bi-encoders (the sentence-transformer that powers our embedding store) are
fast but limited: they encode the query and each document separately, so they
can only model "similar topics" via cosine distance in a shared space.
Cross-encoders score ``(query, document)`` pairs jointly with a transformer
forward pass per pair — slower, but they understand the relationship deeply.

The standard pattern is to use the bi-encoder for coarse retrieval (top-N
candidates from a huge index) and then rerank with the cross-encoder down to
the final top-k. We retrieve 24 candidates and rerank to the 8 the LLM sees.
"""
from __future__ import annotations

from functools import lru_cache

from loguru import logger
from sentence_transformers import CrossEncoder

from .store import Retrieval


@lru_cache(maxsize=4)
def get_reranker(model_name: str) -> CrossEncoder:
    """Load (and cache) a cross-encoder. Models are ~100–300 MB; first call
    downloads, subsequent calls return the same in-memory instance."""
    logger.info(f"Loading cross-encoder reranker: {model_name}")
    return CrossEncoder(model_name)


def rerank(
    question: str,
    candidates: list[Retrieval],
    *,
    model_name: str,
    top_k: int,
) -> list[Retrieval]:
    """Score ``(question, candidate.text)`` pairs with a cross-encoder and
    return the top ``top_k`` candidates ordered by descending score.

    The original cosine-similarity score is replaced with the cross-encoder
    score in the returned ``Retrieval`` objects. Downstream code treats
    ``score`` as "how relevant is this chunk to the query", so writing the
    cross-encoder score there keeps the semantics clean.
    """
    if not candidates:
        return []
    pairs = [(question, r.text) for r in candidates]
    scores = get_reranker(model_name).predict(pairs)

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    out: list[Retrieval] = []
    for original, score in ranked[:top_k]:
        out.append(
            Retrieval(
                paper_id=original.paper_id,
                page=original.page,
                chunk_idx=original.chunk_idx,
                text=original.text,
                score=float(score),
            )
        )
    return out
