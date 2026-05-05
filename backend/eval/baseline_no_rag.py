"""No-retrieval baseline.

Asks the LLM the same questions WITHOUT any retrieved context. Used as
a contrast to show how much retrieval lifts citation accuracy. Any
`[paper_id:page]` markers the model produces here are fabricated — that's
the point of the comparison.

Calls the LLM provider directly (bypassing the FastAPI server) so the
baseline doesn't depend on the server being up.

Usage (from backend/):

    # uses whatever LLM_PROVIDER is set in .env
    python -m eval.baseline_no_rag

    # named run
    python -m eval.baseline_no_rag --name baseline-ollama
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings
from app.llm import GroqProvider, LLMProvider, OllamaProvider

from .run_eval import DATASET, RUNS_DIR, citation_metrics, extract_cited_pages

NO_RAG_SYSTEM = """You are an AI assistant. Answer the user's question using only your general knowledge.

You do not have access to any specific documents, papers, or retrieved chunks. Be concise; plain-text answers without citations."""


def build_provider() -> LLMProvider:
    s = get_settings()
    name = s.llm_provider.lower()
    if name == "ollama":
        return OllamaProvider(
            base_url=s.ollama_base_url,
            model=s.ollama_model,
            request_timeout=s.ollama_request_timeout,
        )
    if name == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing")
        return GroqProvider(api_key=s.groq_api_key, model=s.groq_model)
    raise RuntimeError(f"Unknown LLM_PROVIDER={s.llm_provider!r}")


async def main(run_name: str) -> int:
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        return 2

    questions = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(questions)} questions (no-RAG baseline)")

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    provider = build_provider()
    settings = get_settings()
    print(f"Provider: {settings.llm_provider} — no retrieval")

    results: list[dict[str, Any]] = []
    try:
        with results_path.open("w", encoding="utf-8") as out:
            for i, q in enumerate(questions, 1):
                print(f"[{i:>2}/{len(questions)}] {q['id']}: {q['question'][:60]}...", flush=True)
                start = time.monotonic()
                try:
                    parts: list[str] = []
                    async for tok in provider.stream(NO_RAG_SYSTEM, q["question"]):
                        parts.append(tok)
                    answer = "".join(parts)
                except Exception as exc:
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    out.write(json.dumps({**q, "error": str(exc)}) + "\n")
                    out.flush()
                    continue

                cited = extract_cited_pages(answer)
                metrics = citation_metrics(cited, set(q["expected_pages"]))
                rec = {
                    **q,
                    "cited_pages": sorted(cited),
                    "citation_metrics": metrics,
                    "answer": answer,
                    "retrieved": [],
                    "retrieved_pages": [],
                    "latency_first_token_s": None,
                    "latency_total_s": time.monotonic() - start,
                    "ts": datetime.now(UTC).isoformat(),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                results.append(rec)
                print(
                    f"  cited={sorted(cited) or []} expected={q['expected_pages']} "
                    f"f1={metrics['f1']:.2f} ({rec['latency_total_s']:.1f}s)"
                )
    finally:
        await provider.aclose()

    if not results:
        print("No successful results.")
        return 1

    n = len(results)
    avg_p = sum(r["citation_metrics"]["precision"] for r in results) / n
    avg_r = sum(r["citation_metrics"]["recall"] for r in results) / n
    avg_f = sum(r["citation_metrics"]["f1"] for r in results) / n
    avg_lat = sum(r["latency_total_s"] for r in results) / n
    print()
    print(f"=== No-RAG baseline summary ({n}/{len(questions)} questions) ===")
    print(f"  citation precision : {avg_p:.3f}")
    print(f"  citation recall    : {avg_r:.3f}")
    print(f"  citation F1        : {avg_f:.3f}")
    print(f"  mean latency       : {avg_lat:.1f}s")
    print(f"  results            : {results_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("baseline-%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.name)))
