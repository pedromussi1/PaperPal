"""PDF → page-aware text chunks.

Page-aware chunking is mandatory: every chunk must carry the page it came
from so citations can be rendered. We extract text per page with PyMuPDF,
then split each page into chunks with a recursive character splitter.
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
    text: str

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
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    paper_id: str | None = None,
) -> IngestResult:
    """Parse a PDF and return page-aware chunks.

    Page numbers are 1-indexed (matching what users see in PDF viewers).
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
            for i, piece in enumerate(splitter.split_text(normalized)):
                if piece.strip():
                    chunks.append(
                        Chunk(
                            paper_id=paper_id,
                            page=page_index,
                            chunk_idx=i,
                            text=piece,
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
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> IngestResult:
    data = Path(path).read_bytes()
    return ingest_pdf(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
