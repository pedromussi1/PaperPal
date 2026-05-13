"""PDF → page-aware text chunks (parent-child).

Page-aware chunking is mandatory: every chunk must carry the page it came
from so citations can be rendered. We extract text per page with PyMuPDF,
then split each page into narrow chunks with a recursive character splitter.

Parent-child / "small-to-big" retrieval: each chunk stores two pieces of
text. ``text`` is the narrow form used for embedding (precise retrieval
matching). ``context`` is the wider neighborhood — this chunk plus the
chunks immediately before and after on the same page — and is what's sent
to the LLM. Narrow-match-with-wide-context is a well-established
retrieval pattern: it preserves precision in the index while giving the
generator enough surrounding text to actually answer the question.

Context never crosses page boundaries, since citations are page-anchored.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    page: int
    chunk_idx: int
    text: str          # narrow form — what gets embedded for retrieval
    context: str       # wide form — what gets passed to the LLM

    @property
    def chunk_id(self) -> str:
        return f"{self.paper_id}:{self.page}:{self.chunk_idx}"


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    title: str | None
    page_count: int
    chunks: list[Chunk]


_LIGATURE_FIXES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


def _normalize(text: str) -> str:
    for lig, repl in _LIGATURE_FIXES.items():
        text = text.replace(lig, repl)
    text = re.sub(r"-\n(?=\w)", "", text)  # rejoin words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paper_id_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def ingest_pdf(
    pdf_bytes: bytes,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    parent_window: int = 1,
    paper_id: str | None = None,
) -> IngestResult:
    """Parse a PDF and return page-aware parent-child chunks.

    ``chunk_size`` controls the narrow ``text`` used for retrieval embedding;
    ``parent_window`` controls how many chunks before and after are joined
    into ``context`` for the LLM. With ``chunk_size=400`` and
    ``parent_window=1``, each chunk's context spans roughly three chunks of
    text on its page (about 1.2 KB).

    Page numbers are 1-indexed (matching what users see in PDF viewers);
    ``context`` never crosses page boundaries since citations are page-anchored.
    """
    paper_id = paper_id or _paper_id_from_bytes(pdf_bytes)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    title: str | None = None
    page_count = 0

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        meta_title = (doc.metadata or {}).get("title", "").strip()
        if meta_title:
            title = meta_title

        for page_index, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            normalized = _normalize(raw)
            if not normalized:
                continue
            pieces = [p for p in splitter.split_text(normalized) if p.strip()]
            for i, piece in enumerate(pieces):
                lo = max(0, i - parent_window)
                hi = min(len(pieces), i + parent_window + 1)
                context = "\n".join(pieces[lo:hi])
                chunks.append(
                    Chunk(
                        paper_id=paper_id,
                        page=page_index,
                        chunk_idx=i,
                        text=piece,
                        context=context,
                    )
                )

    return IngestResult(
        paper_id=paper_id,
        title=title,
        page_count=page_count,
        chunks=chunks,
    )


def ingest_pdf_path(
    path: str | Path,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    parent_window: int = 1,
) -> IngestResult:
    data = Path(path).read_bytes()
    return ingest_pdf(
        data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_window=parent_window,
    )
