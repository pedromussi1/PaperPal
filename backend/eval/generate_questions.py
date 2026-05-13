"""LLM-generated eval questions for additional eval-corpus papers.

Reads each PDF in ``backend/eval/fixtures/``, picks N chunks distributed
across the paper's pages, and asks the local Ollama model to write one
factoid-style question per chunk whose answer is contained in that chunk.
Appends new entries to ``dataset.jsonl`` in the schema the eval harness
expects.

This is intentionally less rigorous than human-curated eval — the goal is
to grow the eval beyond a single-paper micro-set so retrieval ablations
(hybrid retrieval especially) have room to show signal. The release notes
for v0.9.0 are honest about the LLM-generation caveat. A future release
can swap LLM-generated questions for hand-curated ones without changing
any of the eval-runner code.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

# Make `from app...` work when called from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest import ingest_pdf  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DATASET = Path(__file__).parent / "dataset.jsonl"

SYSTEM_PROMPT = """\
You are creating evaluation questions for a research-paper QA system.
Given a chunk of text from a paper, write ONE specific factual question
whose answer is contained in this chunk.

OUTPUT FORMAT (strict JSON, no other text):
{"question": "...", "answer": "..."}

RULES:
1. The question must be answerable using ONLY the chunk provided.
2. Prefer factoid questions (specific numbers, names, definitions, formulas)
   over vague ones.
3. Avoid yes/no questions. Avoid questions whose answer is "the paper" or
   "the authors".
4. The answer should be 1-2 sentences extracted from or paraphrased from
   the chunk. Be specific.
5. Output ONLY the JSON object. No commentary, no markdown fence.
"""

USER_TEMPLATE = """Chunk from page {page}:

\"\"\"
{text}
\"\"\"

Generate one factual question whose answer is in this chunk."""


def _parse_json_response(content: str) -> dict | None:
    """Best-effort JSON extraction from an LLM response."""
    content = content.strip()
    # Strip markdown fences if the model added them.
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try to find the first JSON object in the response.
    match = re.search(r"\{.*?\}", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def generate_for_chunk(
    chunk_text: str,
    page: int,
    *,
    model: str,
    base_url: str,
    client: httpx.AsyncClient,
) -> dict | None:
    resp = await client.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(page=page, text=chunk_text)},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        },
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    parsed = _parse_json_response(content)
    if not isinstance(parsed, dict):
        return None
    if not parsed.get("question") or not parsed.get("answer"):
        return None
    return {
        "question": str(parsed["question"]).strip(),
        "answer": str(parsed["answer"]).strip(),
    }


def select_distributed(chunks, k: int):
    """Pick K chunks spread across the paper's pages (longest per page)."""
    by_page: dict[int, list] = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c)
    pages = sorted(by_page.keys())
    if len(pages) <= k:
        return [max(by_page[p], key=lambda c: len(c.text)) for p in pages]
    # Pick K pages evenly across the range; take the longest chunk per page.
    indices = [round(i * (len(pages) - 1) / (k - 1)) for i in range(k)]
    return [max(by_page[pages[i]], key=lambda c: len(c.text)) for i in indices]


async def generate_for_paper(
    pdf_path: Path,
    prefix: str,
    n_questions: int,
    *,
    model: str,
    base_url: str,
) -> list[dict]:
    print(f"\n=== {pdf_path.name} ===")
    result = ingest_pdf(pdf_path.read_bytes())
    print(f"  ingested: {result.page_count} pages, {len(result.chunks)} chunks, paper_id={result.paper_id}")

    selected = select_distributed(result.chunks, n_questions)
    print(f"  generating {len(selected)} question(s)")

    questions: list[dict] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, chunk in enumerate(selected, 1):
            try:
                q = await generate_for_chunk(
                    chunk.text, chunk.page, model=model, base_url=base_url, client=client
                )
            except Exception as exc:
                print(f"  [{i:>2}/{len(selected)}] ERROR: {exc}")
                continue
            if not q:
                print(f"  [{i:>2}/{len(selected)}] page {chunk.page}: <failed to parse>")
                continue
            entry = {
                "id": f"{prefix}-{i:02d}",
                "paper_id": result.paper_id,
                "category": "factoid",
                "question": q["question"],
                "expected_pages": [chunk.page],
                "gold_answer": q["answer"],
            }
            questions.append(entry)
            print(f"  [{i:>2}/{len(selected)}] page {chunk.page}: {q['question'][:80]}")
    return questions


PAPERS: list[tuple[str, str, int]] = [
    # (filename in fixtures/, id-prefix, # questions to generate)
    ("bert.pdf", "bert", 6),
    ("lora.pdf", "lora", 6),
    ("clip.pdf", "clip", 6),
]


async def main(model: str, base_url: str, dry_run: bool) -> int:
    all_questions: list[dict] = []
    for filename, prefix, n in PAPERS:
        pdf = FIXTURES / filename
        if not pdf.exists():
            print(f"SKIP missing: {pdf}")
            continue
        questions = await generate_for_paper(pdf, prefix, n, model=model, base_url=base_url)
        all_questions.extend(questions)

    if not all_questions:
        print("No questions generated.")
        return 1

    if dry_run:
        print(f"\n[dry-run] would append {len(all_questions)} questions to {DATASET}")
        for q in all_questions[:3]:
            print(json.dumps(q, indent=2))
        return 0

    with DATASET.open("a", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")
    print(f"\nAppended {len(all_questions)} questions to {DATASET}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--dry-run", action="store_true", help="Print but don't write.")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.model, args.base_url, args.dry_run)))
