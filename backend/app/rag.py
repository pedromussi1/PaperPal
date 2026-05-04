"""Retrieval-augmented generation: retrieve → assemble prompt → stream Ollama.

Uses a local Ollama server (default http://localhost:11434) so the project
stays free and offline. Ollama's KV cache automatically reuses the system-
prompt prefix across requests with the same model.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

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
        *,
        ollama_base_url: str,
        model: str,
        top_k: int,
        request_timeout: float = 120.0,
    ) -> None:
        self._store = store
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._top_k = top_k
        self._client = httpx.AsyncClient(timeout=request_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def retrieve(
        self, question: str, *, paper_ids: list[str] | None = None, top_k: int | None = None
    ) -> list[Retrieval]:
        return self._store.query(
            question,
            top_k=top_k or self._top_k,
            paper_ids=paper_ids,
        )

    async def stream_answer(
        self,
        question: str,
        retrievals: list[Retrieval],
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "stream": True,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, retrievals)},
            ],
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
            },
        }

        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("done"):
                    break
                msg = event.get("message") or {}
                content = msg.get("content")
                if content:
                    yield content
