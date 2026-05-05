"""End-to-end eval runner.

Reads `dataset.jsonl`, calls the running backend's /query for each
question, parses the SSE stream, extracts cited pages from the answer
text, computes citation-accuracy metrics, and saves per-question results
plus an aggregate summary.

Usage (from backend/):

    # Make sure uvicorn is running (LLM_PROVIDER=ollama or groq), then:
    python -m eval.run_eval                              # default run
    python -m eval.run_eval --name baseline-ollama       # named run
    python -m eval.run_eval --base-url https://my.space  # against deployed
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "dataset.jsonl"
FIXTURE = EVAL_DIR / "fixtures" / "attention.pdf"
RUNS_DIR = EVAL_DIR / "runs"

CITATION_RE = re.compile(r"\[([0-9a-f]{6,32}):(\d+)\]")


def extract_cited_pages(answer: str) -> set[int]:
    return {int(m.group(2)) for m in CITATION_RE.finditer(answer)}


def citation_metrics(cited: set[int], expected: set[int]) -> dict[str, float]:
    """Set-based precision/recall/F1 over cited vs. expected pages."""
    if not cited and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    correct = cited & expected
    p = len(correct) / len(cited) if cited else 0.0
    r = len(correct) / len(expected) if expected else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


async def _parse_sse(stream: AsyncIterator[str]) -> AsyncIterator[tuple[str, Any]]:
    """Yield (event_type, parsed_data) tuples from an SSE byte/text stream."""
    buffer = ""
    async for chunk in stream:
        buffer += chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event_type = "message"
            data_lines: list[str] = []
            for line in frame.splitlines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if not data_lines:
                continue
            yield event_type, json.loads("\n".join(data_lines))


async def ensure_paper_uploaded(client: httpx.AsyncClient, base_url: str) -> str:
    """Ensure the eval fixture is in the index. Returns its paper_id."""
    pdf_bytes = FIXTURE.read_bytes()
    target_pid = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    r = await client.get(f"{base_url}/docs/list")
    r.raise_for_status()
    docs = r.json().get("documents", [])
    if any(d["paper_id"] == target_pid for d in docs):
        return target_pid
    print(f"Fixture not in index, uploading {FIXTURE.name}...")
    files = {"file": (FIXTURE.name, pdf_bytes, "application/pdf")}
    r = await client.post(f"{base_url}/upload", files=files)
    r.raise_for_status()
    return str(r.json()["paper_id"])


async def run_one_query(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    paper_ids: list[str] | None,
) -> dict[str, Any]:
    start = time.monotonic()
    first_token_at: float | None = None
    retrieved: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    final_answer: str | None = None

    payload = {"question": question, "paper_ids": paper_ids, "top_k": None}
    async with client.stream("POST", f"{base_url}/query", json=payload) as resp:
        resp.raise_for_status()
        async for event_type, data in _parse_sse(resp.aiter_text()):
            if event_type == "retrieved":
                retrieved = data
            elif event_type == "token":
                if first_token_at is None:
                    first_token_at = time.monotonic() - start
                answer_parts.append(data["text"])
            elif event_type == "done":
                final_answer = data["answer"]
            elif event_type == "error":
                raise RuntimeError(f"Stream error: {data.get('message')}")

    return {
        "retrieved": retrieved,
        "answer": final_answer or "".join(answer_parts),
        "latency_first_token_s": first_token_at,
        "latency_total_s": time.monotonic() - start,
    }


async def main(base_url: str, run_name: str, paper_ids: list[str] | None) -> int:
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        return 2

    questions = [
        json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"Loaded {len(questions)} questions from {DATASET.name}")

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    async with httpx.AsyncClient(timeout=180.0) as client:
        target_pid = await ensure_paper_uploaded(client, base_url)
        scope = paper_ids if paper_ids is not None else [target_pid]
        print(f"Scoping retrieval to paper_id={target_pid}")

        results: list[dict[str, Any]] = []
        with results_path.open("w", encoding="utf-8") as out:
            for i, q in enumerate(questions, 1):
                short = q["question"][:60]
                print(f"[{i:>2}/{len(questions)}] {q['id']}: {short}...", flush=True)
                try:
                    out_data = await run_one_query(client, base_url, q["question"], scope)
                except Exception as exc:
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    rec = {**q, "error": str(exc)}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    continue

                cited = extract_cited_pages(out_data["answer"])
                metrics = citation_metrics(cited, set(q["expected_pages"]))
                retrieved_pages = sorted({c["page"] for c in out_data["retrieved"]})
                rec = {
                    **q,
                    "cited_pages": sorted(cited),
                    "citation_metrics": metrics,
                    "answer": out_data["answer"],
                    "retrieved": out_data["retrieved"],
                    "retrieved_pages": retrieved_pages,
                    "latency_first_token_s": out_data["latency_first_token_s"],
                    "latency_total_s": out_data["latency_total_s"],
                    "ts": datetime.now(UTC).isoformat(),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                results.append(rec)
                print(
                    f"  cited={sorted(cited) or []} expected={q['expected_pages']} "
                    f"f1={metrics['f1']:.2f} ({out_data['latency_total_s']:.1f}s)"
                )

    if not results:
        print("No successful results.")
        return 1

    n = len(results)
    avg_p = sum(r["citation_metrics"]["precision"] for r in results) / n
    avg_r = sum(r["citation_metrics"]["recall"] for r in results) / n
    avg_f = sum(r["citation_metrics"]["f1"] for r in results) / n
    avg_lat = sum(r["latency_total_s"] for r in results) / n
    print()
    print(f"=== Summary ({n}/{len(questions)} questions) ===")
    print(f"  citation precision : {avg_p:.3f}")
    print(f"  citation recall    : {avg_r:.3f}")
    print(f"  citation F1        : {avg_f:.3f}")
    print(f"  mean latency       : {avg_lat:.1f}s")
    print(f"  results            : {results_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("rag-%Y%m%d-%H%M%S"),
        help="Run name (subdir under runs/). Defaults to a timestamp.",
    )
    parser.add_argument(
        "--paper-ids",
        nargs="*",
        default=None,
        help="Restrict retrieval to these paper IDs (defaults to the eval fixture).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.base_url, args.name, args.paper_ids)))
