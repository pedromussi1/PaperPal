# PaperPal — Project Writeup

A short narrative of what was built, why each major decision was made, and what the eval numbers actually mean. Companion document to [README.md](README.md) (the *what*) and [MODEL_CARD.md](MODEL_CARD.md) (the *reference*).

## Motivation

Most "chat with your PDFs" demos at the time of writing call out to a paid API (OpenAI, Anthropic, Cohere). That makes the demo brittle in three ways:

1. **Cost** — every query metered. A demo that costs nothing has lower friction and doesn't need a billing relationship.
2. **Privacy** — uploaded PDFs end up on a third-party's servers.
3. **Reproducibility** — the underlying API changes silently. An eval result run today against `gpt-4-1106-preview` is incomparable a year later when that model is retired.

PaperPal sets a hard $0 budget and keeps the LLM local by default. The whole stack is free, open-source software composable on a laptop. The cloud-fallback option (Groq) is added only as the *deployed* mode for a public demo, behind a swappable provider interface so the local mode is still authoritative.

## Architecture decisions and tradeoffs

### Why local Ollama as the default LLM

**Decision:** Ollama running Llama 3.1 8B (Q4_K_M quantization) on the user's machine.

**Tradeoff:** slower than a cloud API (~1–10× depending on hardware) and limited to whatever fits in the user's RAM. In return: no API key, no rate limits, no data leaves the machine, and the eval results are bit-reproducible.

**What I'd revisit:** for users without a GPU, the 8B model is borderline usable on CPU. A smaller default (Llama 3.2 3B or Phi-3.5) would broaden the audience but at noticeable quality cost. I chose to keep 8B as default and document smaller alternatives in the README rather than ship a weaker default.

### Why ChromaDB over Pinecone / Weaviate / pgvector

**Decision:** ChromaDB persistent client.

**Reasoning:** local-first ethos again — no separate service required, persists to a local directory. The architectural shape (embeddings + metadata + cosine similarity) is interchangeable with any other vector store, and I documented the migration path. For a portfolio project, the priority is *zero-setup* over *production scale*.

**What it costs:** ChromaDB on disk doesn't scale to millions of chunks well. For a real research-library use case (50+ papers), I'd swap to pgvector or Qdrant.

### Why a swappable LLMProvider

**Decision:** abstract the LLM behind a `LLMProvider` Protocol with two concrete drivers. One env var (`LLM_PROVIDER`) flips the entire pipeline.

**Why it matters:** the public demo on HuggingFace Spaces can't run Ollama (insufficient RAM on the free tier). Without the abstraction, deploying would have meant either running the demo against a paid API (violating the $0 rule) or skipping the demo entirely. The abstraction made it possible to ship a public demo on Groq's free tier *without* compromising the local-first identity.

**Cost:** about 80 lines of code in `app/llm.py`, plus a config branch in `app/main.py`. The retrieval logic, citation rules, and SSE streaming code didn't have to change at all.

### Why Server-Sent Events for streaming, not WebSockets

**Decision:** the backend streams answer tokens over SSE; the Next.js frontend parses the stream in a Route Handler and re-emits to the browser.

**Why:** SSE is one-way (server → client), which is exactly what we need. WebSockets would add bidirectional capability we don't use, plus connection-management complexity. SSE also passes through the Vercel → HF Spaces proxy chain without any special config — just keep `Cache-Control: no-cache` and `X-Accel-Buffering: no`.

### Why Next.js Route Handlers as a backend proxy

**Decision:** the browser only ever talks to the Next.js dev/prod server, which then proxies every API call to the FastAPI backend.

**Two benefits:**
1. **No CORS** — the browser sees only same-origin requests.
2. **Backend URL hidden from the client bundle** — the FastAPI URL lives in `BACKEND_URL` (server-only env var) and isn't exposed in the JS shipped to users.

The frontend ends up speaking a clean typed API to itself, with the network boundary moved one hop deeper.

## Eval methodology

Citation accuracy is the headline metric — set-based precision/recall/F1 of `[paper_id:page]` markers vs. a hand-labelled gold list of pages where the answer actually lives. This deliberately *doesn't* measure prose quality; it measures *grounding*. A model that produces a beautiful, fluent, totally fabricated answer with no citations would score near zero, which is what we want.

The no-RAG baseline runs the same questions through the same LLM with no retrieval. Because Llama 3.1 has seen the Transformer paper in training, the baseline's *prose* answers are often roughly correct — but it can't produce real citations, so the F1 collapses to 0. That gap is the lift retrieval gives.

**What the numbers actually say** (12 questions, local Ollama):

- RAG citation F1 = 0.542. Not great in absolute terms — about half the citations are right. But across 12 hand-labelled gold-page sets, this measures something real.
- The failure-mode breakdown is more interesting than the headline number. The model **drops** citations on math, **adds spurious** citations on long answers, and **picks the wrong page** at section boundaries. Each is a different model-behavior bug suggesting a different fix (better few-shot examples for math; constrained decoding to limit citations to retrieved pages; finer-grained chunking near section boundaries).

**What the eval doesn't yet measure:**
- **Faithfulness** — does the answer text only state things present in the retrieved context? Adding RAGAS or a similar LLM-as-judge check is on the roadmap.
- **Multi-paper coverage** — currently 12 questions on one paper. Extending to 3–5 papers would make the numbers more credible.
- **Ablations** — chunk size, k, embedding model. The harness can already do this if you re-ingest with different settings between runs; I haven't automated the sweep yet.

## Future work

The roadmap section in [README.md](README.md#roadmap) lists the active items. The two I'd prioritize next:

1. **RAGAS faithfulness as a judge** — for each retrieved chunk + answer pair, ask a strong judge LLM (Groq's Llama 3.3 70B is free and a reasonable choice) to score whether the claim in the answer is supported by the chunk. This catches the spurious-citation failure mode the current metric tolerates as long as the cited page exists.
2. **Chunk-size ablation** — re-ingest the test paper with chunk sizes 500/800/1200, run the eval against each, and produce a small table. This is the kind of result reviewers like to see — turns the project from "I built a RAG" into "I measured a RAG."

## Acknowledgements

- The Transformer paper (Vaswani et al., 2017) — used as the test fixture and as inspiration; available on arXiv at 1706.03762.
- The Ollama, sentence-transformers, ChromaDB, FastAPI, Next.js, shadcn/ui, and Groq teams — for the underlying open-source / free-tier components that make a $0 stack possible.
