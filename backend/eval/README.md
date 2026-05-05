# PaperPal eval harness

End-to-end evaluation of the RAG pipeline: feeds a hand-curated dataset through the running backend, scores each answer against gold-standard cited pages, and compares against a no-retrieval baseline.

The latest results live in [`REPORT.md`](./REPORT.md).

## What's measured

**Citation accuracy** (the primary metric): set-based precision/recall/F1 of the `[paper_id:page]` markers the model produces vs. a hand-labelled list of pages where the answer actually lives.

This metric is intentionally strict — it doesn't care how good the prose answer reads, only whether the model is *grounding* its claims in real, correct pages. That separates "the model produced a plausible-looking answer using its training data" (which is what the baseline does) from "the model is actually using the retrieved context" (which is what we want).

## Layout

```
eval/
├── dataset.jsonl       12 hand-curated Q-A-citation triples on the Transformer paper
├── fixtures/
│   └── attention.pdf   the test paper (auto-uploaded to the index by the runner)
├── run_eval.py         RAG runner — calls /query, parses SSE, scores citations
├── baseline_no_rag.py  same questions, no retrieval — calls the LLM directly
├── report.py           generates REPORT.md from one or two run directories
├── runs/               per-run results.jsonl (gitignored)
└── REPORT.md           checked-in summary of the latest run
```

## Reproducing the report

```sh
# from backend/, with uvicorn already running (LLM_PROVIDER=ollama or groq)
python -m eval.run_eval --name rag-mvp
python -m eval.baseline_no_rag --name baseline-mvp
python -m eval.report --rag rag-mvp --baseline baseline-mvp
```

That writes `runs/rag-mvp/results.jsonl`, `runs/baseline-mvp/results.jsonl`, and overwrites `REPORT.md`.

`run_eval.py` will auto-upload `fixtures/attention.pdf` if the paper isn't already in the index. Both runners use whatever LLM provider is configured in `.env` — re-run with a different `LLM_PROVIDER` to get apples-to-apples numbers across providers.

## What's NOT measured (yet)

Roadmap items the harness currently doesn't cover:

- **RAGAS metrics** (faithfulness, answer-relevance, context-precision, context-recall) — useful when the prose quality matters as much as the citations. Would require an LLM-as-judge step.
- **Multi-paper evals** — the dataset is currently a single paper; broader coverage would make the numbers more credible.
- **Chunk-size / k / embedding-model ablations** — the harness can already do this if you re-ingest with different settings between runs; a `sweep.py` to automate it is a natural next step.
