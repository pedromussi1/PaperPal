---
title: PaperPal Backend
emoji: 📄
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: RAG over research papers — backend service (FastAPI + ChromaDB)
---

# PaperPal Backend

Backend service for [PaperPal](https://github.com/pedromussi1/PaperPal) — chat over research papers with grounded, page-anchored citations.

This Space hosts the **FastAPI backend only**. The frontend is a separate Next.js app deployed on Vercel.

## What this service does

- `POST /upload` — multipart PDF upload; parses, chunks page-aware, embeds, stores in ChromaDB
- `POST /query` — Server-Sent Events stream: retrieved chunks → answer tokens → final payload
- `GET  /docs/list` — list ingested papers
- `DELETE /docs/{paper_id}` — remove a paper from the index
- `GET  /healthz` — liveness check

## Stack

| Component | Tech |
|---|---|
| Web framework | FastAPI + uvicorn |
| PDF parsing | pymupdf (page-aware chunking) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent client) |
| LLM | Groq (`llama-3.1-8b-instant`) for the public demo, Ollama (Llama 3.1 8B) for local dev |

## Configuration

Required Space **secret**:

- `GROQ_API_KEY` — get one free at https://console.groq.com (no credit card)

Required Space **variable**:

- `LLM_PROVIDER=groq` — selects the cloud provider for this deployment

Optional:

- `GROQ_MODEL` (default `llama-3.1-8b-instant`) — also try `llama-3.3-70b-versatile`
- `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` — see `.env.example` in the source repo

## Caveats of the free Space tier

- The free CPU Space sleeps after ~48h of inactivity (cold-start ~30 sec on first wake).
- Ephemeral filesystem — uploaded PDFs and the ChromaDB index don't persist across restarts. Good enough for a demo; for real use, run locally with Ollama or upgrade the Space to a persistent tier.

## Local development

```sh
git clone https://github.com/pedromussi1/PaperPal
cd PaperPal/backend
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
cp .env.example .env  # set LLM_PROVIDER=ollama (default) for local
uvicorn app.main:app --reload
```

See the [main README](https://github.com/pedromussi1/PaperPal#readme) for the full setup including the frontend.

## License

MIT
