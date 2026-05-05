"""FastAPI entrypoint.

Endpoints:
    POST /upload      multipart PDF → ingest → return paper_id
    GET  /docs/list   list ingested papers
    POST /query       question → SSE stream of answer tokens (final event = full payload)
    GET  /healthz     liveness check
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from .config import get_settings
from .embeddings import get_embedder
from .ingest import ingest_pdf
from .llm import GroqProvider, LLMProvider, OllamaProvider
from .models import (
    DocumentList,
    DocumentSummary,
    IngestResponse,
    QueryRequest,
)
from .rag import RagEngine
from .store import VectorStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    provider: LLMProvider
    name = settings.llm_provider.lower()
    if name == "ollama":
        provider = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            request_timeout=settings.ollama_request_timeout,
        )
        model_label = settings.ollama_model
    elif name == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. "
                "Get a key at https://console.groq.com (free, no credit card)."
            )
        provider = GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
        model_label = settings.groq_model
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={settings.llm_provider!r}. Must be 'ollama' or 'groq'."
        )
    logger.info(
        f"Starting PaperPal backend (provider={name}, model={model_label})"
    )

    embedder = get_embedder(settings.embedding_model)
    store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        embedder=embedder,
    )

    rag = RagEngine(store=store, provider=provider, top_k=settings.top_k)

    app.state.store = store
    app.state.rag = rag
    app.state.settings = settings

    yield

    await rag.aclose()
    logger.info("Shutting down PaperPal backend")


app = FastAPI(title="PaperPal", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=IngestResponse)
async def upload(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    settings = get_settings()
    result = ingest_pdf(
        pdf_bytes,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not result.chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text in PDF (scanned or empty?)",
        )

    title = result.title or file.filename
    app.state.store.add_chunks(result.chunks, title=title)
    logger.info(
        f"Ingested {file.filename}: paper_id={result.paper_id} "
        f"pages={result.page_count} chunks={len(result.chunks)}"
    )
    return IngestResponse(
        paper_id=result.paper_id,
        title=title,
        pages=result.page_count,
        chunks=len(result.chunks),
    )


@app.get("/docs/list", response_model=DocumentList)
def list_docs() -> DocumentList:
    raw = app.state.store.list_papers()
    return DocumentList(
        documents=[
            DocumentSummary(
                paper_id=str(r["paper_id"]),
                title=str(r["title"]) if r.get("title") else None,
                pages=int(r["pages"]),
                chunks=int(r["chunks"]),
            )
            for r in raw
        ]
    )


@app.delete("/docs/{paper_id}", status_code=204)
def delete_doc(
    paper_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
) -> None:
    deleted = app.state.store.delete_paper(paper_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="paper_id not found")
    logger.info(f"Deleted paper_id={paper_id} ({deleted} chunks)")


@app.post("/query")
async def query(req: QueryRequest) -> StreamingResponse:
    rag: RagEngine = app.state.rag
    retrievals = rag.retrieve(
        req.question, paper_ids=req.paper_ids, top_k=req.top_k
    )

    async def sse() -> AsyncIterator[str]:
        retrieval_payload = [
            {
                "paper_id": r.paper_id,
                "page": r.page,
                "chunk_idx": r.chunk_idx,
                "text": r.text,
                "score": r.score,
            }
            for r in retrievals
        ]
        yield f"event: retrieved\ndata: {json.dumps(retrieval_payload)}\n\n"

        full_answer: list[str] = []
        try:
            async for token in rag.stream_answer(req.question, retrievals):
                full_answer.append(token)
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
        except Exception as exc:
            logger.exception("stream_answer failed")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
            return

        yield f"event: done\ndata: {json.dumps({'answer': ''.join(full_answer)})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
