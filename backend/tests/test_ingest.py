"""Smoke tests for the PDF ingestion pipeline.

Covers chunking, page-aware metadata, and ligature normalization. PDF parsing
is exercised end-to-end with a tiny in-memory PDF generated via PyMuPDF, so
these tests don't depend on fixture files.
"""

from __future__ import annotations

import pymupdf

from app.ingest import _normalize, ingest_pdf


def _make_pdf(pages: list[str]) -> bytes:
    """Build a minimal PDF in-memory with one text block per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_normalize_fixes_ligatures_and_hyphens():
    raw = "ﬁnal\nstate ma-\nchine\nrunning   well"
    # Hyphen + newline rejoins ("ma-\nchine" → "machine"); other newlines
    # are preserved as semantic line breaks; runs of spaces collapse.
    assert _normalize(raw) == "final\nstate machine\nrunning well"


def test_normalize_collapses_excessive_blank_lines():
    raw = "para one\n\n\n\npara two"
    assert _normalize(raw) == "para one\n\npara two"


def test_ingest_assigns_correct_pages():
    pdf = _make_pdf(["Page one body text.", "Page two body text.", "Page three."])
    result = ingest_pdf(pdf, chunk_size=200, chunk_overlap=20)

    assert result.page_count == 3
    assert {c.page for c in result.chunks} == {1, 2, 3}
    assert all(c.paper_id == result.paper_id for c in result.chunks)


def test_ingest_skips_blank_pages():
    pdf = _make_pdf(["First page.", "", "Third page."])
    result = ingest_pdf(pdf, chunk_size=200, chunk_overlap=20)

    pages_with_chunks = {c.page for c in result.chunks}
    assert 1 in pages_with_chunks
    assert 3 in pages_with_chunks
    assert 2 not in pages_with_chunks


def test_ingest_is_deterministic_paper_id():
    pdf = _make_pdf(["Same content."])
    a = ingest_pdf(pdf)
    b = ingest_pdf(pdf)
    assert a.paper_id == b.paper_id


def test_chunk_id_format():
    pdf = _make_pdf(["Some body text on a single page."])
    result = ingest_pdf(pdf, chunk_size=100, chunk_overlap=10)
    chunk = result.chunks[0]
    assert chunk.chunk_id == f"{chunk.paper_id}:{chunk.page}:{chunk.chunk_idx}"
