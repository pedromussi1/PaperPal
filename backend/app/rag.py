"""Retrieval-augmented generation: retrieve → assemble prompt → stream LLM.

Provider-agnostic: takes any `LLMProvider` (Ollama, Groq, etc.). The
RAG logic — chunk formatting, prompt assembly, citation rules — is the
same regardless of which model is generating tokens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from .llm import LLMProvider
from .rerank import rerank
from .store import Retrieval, VectorStore

_SYSTEM_PROMPT = """You are PaperPal, a research assistant that answers questions about uploaded research papers.

Rules:
1. Ground every factual claim in the provided <retrieved_chunks>. Do not use outside knowledge.
2. Cite each claim with the page it came from, using the format [paper_id:page]. Place citations inline immediately after the claim.
3. If the retrieved chunks do not contain enough information to answer, say so explicitly. Do not speculate.
4. Be concise. Prefer short, direct answers. Quote sparingly.
5. If multiple chunks support a claim, cite the most relevant one.

Format your response as plain markdown with inline citations like [a1b2c3:7].

Example:

<retrieved_chunks>
[a1b2c3:3] The Transformer architecture relies entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.
[a1b2c3:5] Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation.
</retrieved_chunks>

Question: What is the core mechanism behind the Transformer?

Answer: The Transformer relies entirely on self-attention rather than RNNs or convolution [a1b2c3:3]. Self-attention relates different positions of a single sequence to compute its representation [a1b2c3:5]."""


@dataclass(frozen=True)
class RagResult:
    answer: str
    retrievals: list[Retrieval]


def _format_chunks(retrievals: list[Retrieval]) -> str:
    return "\n\n".join(
        f"[{r.paper_id}:{r.page}] {r.text}" for r in retrievals
    )


def _build_user_message(question: str, retrievals: list[Retrieval]) -> str:
    chunks_text = _format_chunks(retrievals) if retrievals else "(no relevant chunks found)"
    return (
        f"<retrieved_chunks>\n{chunks_text}\n</retrieved_chunks>\n\n"
        f"Question: {question}"
    )


class RagEngine:
    def __init__(
        self,
        store: VectorStore,
        provider: LLMProvider,
        *,
        top_k: int,
        retrieve_top_k: int = 24,
        reranker_model: str = "",
    ) -> None:
        self._store = store
        self._provider = provider
        self._top_k = top_k
        self._retrieve_top_k = retrieve_top_k
        self._reranker_model = reranker_model

    async def aclose(self) -> None:
        await self._provider.aclose()

    def retrieve(
        self,
        question: str,
        *,
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[Retrieval]:
        """Two-stage retrieval when a reranker is configured.

        Stage 1: cosine-similarity search in the bi-encoder index (cheap,
        good for coarse recall).
        Stage 2: cross-encoder reranking on the top ``retrieve_top_k``
        candidates (more accurate, scales with N rather than corpus size).

        When ``reranker_model`` is empty, falls back to single-stage retrieval
        — useful for A/B comparison against the v0.4.0 eval baseline.
        """
        final_k = top_k or self._top_k
        if not self._reranker_model:
            return self._store.query(question, top_k=final_k, paper_ids=paper_ids)

        # Pull a wider candidate pool so the reranker has something to reorder.
        candidate_k = max(self._retrieve_top_k, final_k)
        candidates = self._store.query(question, top_k=candidate_k, paper_ids=paper_ids)
        return rerank(
            question,
            candidates,
            model_name=self._reranker_model,
            top_k=final_k,
        )

    async def stream_answer(
        self,
        question: str,
        retrievals: list[Retrieval],
    ) -> AsyncIterator[str]:
        async for token in self._provider.stream(
            _SYSTEM_PROMPT,
            _build_user_message(question, retrievals),
        ):
            yield token
