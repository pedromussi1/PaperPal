"""ChromaDB-backed vector store.

One collection per embedding-model + chunking config so ablations don't
collide. Documents (chunks) carry paper_id, page, and chunk_idx as
metadata so retrieval results can be rendered as page-anchored citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import Embedder
from .ingest import Chunk


@dataclass(frozen=True)
class Retrieval:
    paper_id: str
    page: int
    chunk_idx: int
    text: str
    score: float


class VectorStore:
    """Thin wrapper around a Chroma persistent collection."""

    def __init__(
        self,
        persist_dir: Path,
        *,
        collection_name: str = "papers",
        embedder: Embedder,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = embedder
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedder.model_name, "hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], *, title: str | None = None) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        base_meta: dict[str, str | int] = {}
        if title:
            base_meta["title"] = title
        metadatas = [
            {**base_meta, "paper_id": c.paper_id, "page": c.page, "chunk_idx": c.chunk_idx}
            for c in chunks
        ]
        embeddings = self._embedder.embed(texts).tolist()
        self._collection.upsert(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def delete_paper(self, paper_id: str) -> int:
        """Remove every chunk belonging to `paper_id`. Returns the number of
        chunks that were deleted (0 if the paper wasn't in the index)."""
        existing = self._collection.get(where={"paper_id": paper_id}, include=[])
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def query(
        self,
        question: str,
        *,
        top_k: int = 8,
        paper_ids: list[str] | None = None,
    ) -> list[Retrieval]:
        q_embedding = self._embedder.embed_one(question).tolist()
        where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
        result = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            where=where,
        )

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        retrievals: list[Retrieval] = []
        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            retrievals.append(
                Retrieval(
                    paper_id=str(meta["paper_id"]),
                    page=int(meta["page"]),
                    chunk_idx=int(meta["chunk_idx"]),
                    text=doc,
                    score=1.0 - float(dist),  # cosine similarity from cosine distance
                )
            )
        return retrievals

    def all_chunks(self) -> tuple[list[str], list[dict]]:
        """Return every chunk's text and metadata. Used by callers that
        maintain a parallel index (e.g. BM25) over the same corpus."""
        result = self._collection.get(include=["documents", "metadatas"])
        return (result.get("documents") or [], result.get("metadatas") or [])

    def list_papers(self) -> list[dict[str, int | str | None]]:
        result = self._collection.get(include=["metadatas"])
        metas = result.get("metadatas") or []
        by_paper: dict[str, dict[str, int | str | None]] = {}
        for meta in metas:
            pid = str(meta["paper_id"])
            entry = by_paper.setdefault(
                pid, {"chunks": 0, "max_page": 0, "title": None}
            )
            entry["chunks"] = int(entry["chunks"] or 0) + 1  # type: ignore[arg-type]
            entry["max_page"] = max(int(entry["max_page"] or 0), int(meta["page"]))  # type: ignore[arg-type]
            if entry["title"] is None and meta.get("title"):
                entry["title"] = str(meta["title"])
        return [
            {
                "paper_id": pid,
                "title": v["title"],
                "pages": v["max_page"],
                "chunks": v["chunks"],
            }
            for pid, v in by_paper.items()
        ]
