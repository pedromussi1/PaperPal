"""Generate `REPORT.md` from one or two eval runs.

Usage (from backend/):

    # both — produces aggregate + lift table + side-by-side examples
    python -m eval.report --rag rag-20260505-120000 --baseline baseline-20260505-120500

    # rag only
    python -m eval.report --rag rag-20260505-120000

The output is a single markdown file at backend/eval/REPORT.md (or
wherever --output points) and is intended to be committed to the repo
as the canonical "this is how the system performs" artifact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).parent
RUNS_DIR = EVAL_DIR / "runs"
DEFAULT_OUTPUT = EVAL_DIR / "REPORT.md"


def load_run(name: str) -> list[dict[str, Any]]:
    p = RUNS_DIR / name / "results.jsonl"
    if not p.exists():
        raise SystemExit(f"Run not found: {p}")
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and "error" not in json.loads(line)
    ]


def _avg(records: list[dict[str, Any]], path: list[str]) -> float:
    vals: list[float] = []
    for r in records:
        cur: Any = r
        try:
            for k in path:
                cur = cur[k]
            if cur is not None:
                vals.append(float(cur))
        except (KeyError, TypeError):
            continue
    return sum(vals) / len(vals) if vals else 0.0


def render_aggregate(rag: list[dict[str, Any]], baseline: list[dict[str, Any]] | None) -> str:
    rag_p = _avg(rag, ["citation_metrics", "precision"])
    rag_r = _avg(rag, ["citation_metrics", "recall"])
    rag_f = _avg(rag, ["citation_metrics", "f1"])
    rag_lat = _avg(rag, ["latency_total_s"])

    if baseline:
        b_p = _avg(baseline, ["citation_metrics", "precision"])
        b_r = _avg(baseline, ["citation_metrics", "recall"])
        b_f = _avg(baseline, ["citation_metrics", "f1"])
        b_lat = _avg(baseline, ["latency_total_s"])
        return "\n".join(
            [
                "| Metric | No-RAG baseline | RAG | Lift |",
                "|---|---:|---:|---:|",
                f"| Citation precision | {b_p:.3f} | {rag_p:.3f} | +{rag_p - b_p:.3f} |",
                f"| Citation recall    | {b_r:.3f} | {rag_r:.3f} | +{rag_r - b_r:.3f} |",
                f"| Citation F1        | {b_f:.3f} | {rag_f:.3f} | +{rag_f - b_f:.3f} |",
                f"| Mean latency (s)   | {b_lat:.2f} | {rag_lat:.2f} | {rag_lat - b_lat:+.2f} |",
            ]
        )
    return "\n".join(
        [
            "| Metric | RAG |",
            "|---|---:|",
            f"| Citation precision | {rag_p:.3f} |",
            f"| Citation recall    | {rag_r:.3f} |",
            f"| Citation F1        | {rag_f:.3f} |",
            f"| Mean latency (s)   | {rag_lat:.2f} |",
        ]
    )


def render_per_question(records: list[dict[str, Any]]) -> str:
    rows = [
        "| ID | Category | Cited | Expected | Precision | Recall | F1 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in records:
        cm = r.get("citation_metrics", {})
        rows.append(
            f"| `{r['id']}` | {r.get('category', '')} | "
            f"{r.get('cited_pages', [])} | {r['expected_pages']} | "
            f"{cm.get('precision', 0):.2f} | {cm.get('recall', 0):.2f} | {cm.get('f1', 0):.2f} |"
        )
    return "\n".join(rows)


def render_examples(
    rag: list[dict[str, Any]],
    baseline: list[dict[str, Any]] | None,
    n: int = 3,
) -> str:
    base_by_id = {r["id"]: r for r in (baseline or [])}
    out: list[str] = []
    for r in rag[:n]:
        out.append(f"### {r['id']} — {r.get('category', '')}\n")
        out.append(f"**Q:** {r['question']}\n")
        out.append(f"**Expected pages:** {r['expected_pages']}\n")
        out.append(f"**Gold answer:** {r['gold_answer']}\n")
        out.append("\n**RAG answer:**\n")
        for line in (r.get("answer") or "").strip().splitlines() or ["(empty)"]:
            out.append(f"> {line}")
        out.append("")
        if r["id"] in base_by_id:
            out.append("\n**No-RAG baseline answer:**\n")
            for line in (base_by_id[r["id"]].get("answer") or "").strip().splitlines() or ["(empty)"]:
                out.append(f"> {line}")
            out.append("")
    return "\n".join(out)


def main(rag_name: str, baseline_name: str | None, output: Path) -> int:
    rag = load_run(rag_name)
    baseline = load_run(baseline_name) if baseline_name else None

    sections = [
        "# PaperPal eval report",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"- Dataset: `dataset.jsonl` ({len(rag)} questions on the Transformer paper)",
        f"- RAG run: `{rag_name}`",
    ]
    if baseline_name:
        sections.append(f"- Baseline run: `{baseline_name}` (no retrieval)")
    sections.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            render_aggregate(rag, baseline),
            "",
            "**How to read this:** the citation-accuracy metrics measure whether the model's `[paper_id:page]` citations point at pages where the answer actually lives. The no-RAG baseline cannot produce real citations (it has no retrieval), so its scores are near zero — the gap is the *lift* retrieval gives.",
            "",
            "## Per-question breakdown (RAG)",
            "",
            render_per_question(rag),
            "",
            "## Side-by-side examples",
            "",
            render_examples(rag, baseline),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", required=True, help="RAG run name (subdir under runs/)")
    parser.add_argument("--baseline", default=None, help="No-RAG baseline run name")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    raise SystemExit(main(args.rag, args.baseline, Path(args.output)))
