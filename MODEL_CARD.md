# PaperPal — Model Card

This is a **system card** rather than a model card: PaperPal isn't a single model, it's a retrieval-augmented-generation pipeline composed of an embedding model + a vector store + a chat LLM. This document describes the components, intended use, and known limitations.

## System overview

PaperPal answers natural-language questions about user-uploaded PDFs by:

1. Parsing the PDF page-by-page and chunking each page with a recursive character splitter.
2. Embedding chunks with a sentence-transformer and storing them in a persistent ChromaDB collection.
3. On query, embedding the question, retrieving the top-k chunks via cosine similarity, and passing them to a chat LLM with a system prompt that requires every claim be cited as `[paper_id:page]`.

## Components

| Role | Default | Configurable via |
|---|---|---|
| **PDF parser** | `pymupdf` (fitz) | n/a |
| **Chunker** | LangChain `RecursiveCharacterTextSplitter`, 800-token chunks, 100-token overlap | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim) | `EMBEDDING_MODEL` |
| **Vector store** | ChromaDB persistent client, cosine similarity | n/a |
| **LLM (local)** | Ollama + `llama3.1:8b` (4-bit Q4_K_M quantization) | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL` |
| **LLM (cloud)** | Groq + `llama-3.1-8b-instant` | `LLM_PROVIDER=groq`, `GROQ_MODEL` |

All components are open-source or have free-tier API access; no paid services are required.

## Intended use

- **Personal research assistant** — help a user navigate research papers they're reading.
- **Demo of grounded RAG** — illustrate page-anchored citations as a way to surface where each claim came from.
- **Educational reference** — example codebase for "how to build a RAG system end-to-end" with a swappable LLM, eval harness, and deployed demo.

## Out-of-scope / not designed for

- **Scanned PDFs** (no OCR pipeline in v1; born-digital arXiv-style PDFs only).
- **High-stakes decisions** — answers can be wrong or partially-cited. Do not use as a replacement for reading the source.
- **Multi-tenant production hosting** — the persistence layer has no auth; everyone with a session sees the same library.
- **Sensitive documents** in cloud mode — when `LLM_PROVIDER=groq`, retrieved chunk text is sent to Groq's servers. For sensitive material, run locally with Ollama, where nothing leaves the machine.

## Performance

Eval harness in [`backend/eval/`](backend/eval/) measures **citation accuracy** (set-based precision/recall/F1 of the model's `[paper_id:page]` markers vs. a hand-labelled gold set).

Result on a 12-question Transformer-paper set, local Ollama (Llama 3.1 8B):

| Metric | No-RAG baseline | RAG | Lift |
|---|---:|---:|---:|
| Citation precision | 0.000 | 0.500 | +0.500 |
| Citation recall    | 0.000 | 0.625 | +0.625 |
| Citation F1        | 0.000 | 0.542 | +0.542 |

See [`backend/eval/REPORT.md`](backend/eval/REPORT.md) for the full per-question breakdown.

## Known failure modes

Surfaced by the eval:

- **Drops citation format on math** — for the question *"What is the formula for Scaled Dot-Product Attention?"*, the model returned the formula but emitted `[1]` instead of the structured `[paper_id:page]` marker, so citation accuracy dropped to 0 even though the answer was correct.
- **Spurious extra citations** — on longer answers, the model sometimes appends a citation to a page that doesn't actually contain the claim. `att-01` cited page 8 for the statement *"Self-attention has been used in conjunction with recurrent networks"*, but that statement only appears on page 2.
- **Off-by-one page errors at section boundaries** — when an answer spans a page boundary (e.g., the "Optimizer" subsection straddles pages 7 and 8), the model picks one and may pick the wrong one. The retrieval typically returns both, but the model's citation choice can be brittle.
- **Hallucinated content under no-RAG mode** — without retrieval, the LLM can produce plausible-sounding answers from its training data even on details it has no real source for. This is the main reason the project ships with a non-trivial eval and a no-RAG baseline.

## Bias considerations

The defaults (English-language papers, MiniLM embeddings, Llama 3.1) inherit the biases of those models:
- MiniLM is trained primarily on English; non-English papers will retrieve poorly.
- Llama 3.1 8B has the typical chat-LLM disposition toward fluent-but-wrong answers when the retrieved context is insufficient.
- The page-aware chunking assumes Roman scripts and standard PDF layouts; right-to-left or vertical-text scripts may chunk poorly.

## Updates

| Date | Change |
|---|---|
| 2026-05-04 | Initial release: backend + frontend + dual-mode (Ollama / Groq) deploy |
| 2026-05-05 | Eval harness MVP with citation-accuracy metric and no-RAG baseline |

## License

MIT — see [LICENSE](LICENSE).
