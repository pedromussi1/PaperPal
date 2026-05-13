"""FastAPI entrypoint.

Endpoints:
    POST /upload       multipart PDF → ingest → return paper_id
    GET  /docs/list    list ingested papers
    POST /query        question → SSE stream of answer tokens (final event = full payload)
    POST /query/image  image attachment → OCR → SSE stream (same shape + leading ocr event)
    GET  /healthz      liveness check
"""

from __future__ import annotations

# IMPORTANT: SSL setup BEFORE anything else that might touch the network.
# httpx (used by huggingface_hub) uses its own SSL context and does not read
# SSL_CERT_FILE / REQUESTS_CA_BUNDLE. On Windows the bundled OpenSSL often
# can't find a valid CA path, which crashes the sentence-transformers /
# huggingface_hub startup with `CERTIFICATE_VERIFY_FAILED`. `truststore`
# patches Python's `ssl` module to use the OS certificate store — which is
# always valid on Windows/macOS/Linux. After this call, every later SSL
# request (httpx, requests, urllib) just works.
import truststore as _truststore

_truststore.inject_into_ssl()

# Populate os.environ from backend/.env so settings like VISION_MODEL are
# available to all modules. pydantic-settings will also load .env when
# Settings() is constructed, but it reads into the Settings object only —
# never back into os.environ. Anything that consults os.environ directly
# (HF env flags, third-party libs) needs this to run first.
from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

_load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from .config import get_settings
from .embeddings import get_embedder
from .hybrid import BM25Index
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
from .vlm import VlmError, extract_text_from_image

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

    bm25 = BM25Index(store) if settings.hybrid_retrieval else None

    rag = RagEngine(
        store=store,
        provider=provider,
        top_k=settings.top_k,
        retrieve_top_k=settings.retrieve_top_k,
        reranker_model=settings.reranker_model,
        bm25=bm25,
        rrf_k=settings.rrf_k,
    )

    app.state.store = store
    app.state.bm25 = bm25
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
        parent_window=settings.parent_window,
    )

    if not result.chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text in PDF (scanned or empty?)",
        )

    title = result.title or file.filename
    app.state.store.add_chunks(result.chunks, title=title)
    if app.state.bm25 is not None:
        app.state.bm25.invalidate()
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
    if app.state.bm25 is not None:
        app.state.bm25.invalidate()
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


@app.post("/query/image")
async def query_image(
    image: UploadFile = File(...),
    text: str = Form(""),
    top_k: int = Form(0),
    paper_ids: str = Form(""),
) -> StreamingResponse:
    """Ask a question whose content is an image of text.

    A local Ollama vision-language model transcribes the image while
    preserving its visual structure (boxes, columns, labeled options). The
    transcribed text becomes the query for the existing RAG pipeline. An
    optional ``text`` field is prepended so users can attach context.

    Multipart form fields (all but ``image`` optional):
        image      file (image/*) — required
        text       string — additional user-typed context, prepended to the transcription
        paper_ids  comma-separated paper IDs to scope retrieval to
        top_k      int > 0 — override the default retrieval depth

    Streams the same SSE events as /query, plus a leading ``ocr`` event
    carrying the transcribed text so the frontend can render the user's
    transcribed message. (The event name stayed ``ocr`` for client
    compatibility even though we no longer use Tesseract underneath.)
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image/* uploads are accepted.")

    image_bytes = await image.read()
    settings = get_settings()
    try:
        vlm = await extract_text_from_image(
            image_bytes,
            base_url=settings.ollama_base_url,
            model=settings.vision_model,
            timeout=settings.vision_request_timeout,
        )
    except VlmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not vlm.text:
        raise HTTPException(
            status_code=422,
            detail="The vision model returned no text. Try a clearer or higher-resolution image.",
        )

    user_text = (text or "").strip()
    if user_text:
        question = f"{user_text}\n\n[Text from attached image]\n{vlm.text}"
    else:
        question = vlm.text
    if len(question) > 5000:
        raise HTTPException(
            status_code=400,
            detail=f"Combined query too long ({len(question)} chars; cap is 5000).",
        )

    parsed_paper_ids = [p.strip() for p in paper_ids.split(",") if p.strip()] or None
    effective_top_k = top_k if top_k > 0 else None

    rag: RagEngine = app.state.rag
    retrievals = rag.retrieve(question, paper_ids=parsed_paper_ids, top_k=effective_top_k)
    logger.info(
        f"/query/image: vlm={len(vlm.text)} chars (model={vlm.model}), "
        f"retrieved={len(retrievals)} chunks, paper_ids={parsed_paper_ids}"
    )

    async def sse() -> AsyncIterator[str]:
        yield (
            "event: ocr\n"
            f"data: {json.dumps({'text': vlm.text, 'width': 0, 'height': 0})}\n\n"
        )

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
            async for token in rag.stream_answer(question, retrievals):
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
