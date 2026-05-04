from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    paper_id: str
    title: str | None = None
    pages: int
    chunks: int


class DocumentSummary(BaseModel):
    paper_id: str
    title: str | None
    pages: int
    chunks: int


class DocumentList(BaseModel):
    documents: list[DocumentSummary]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    paper_ids: list[str] | None = None
    top_k: int | None = Field(None, gt=0, le=32)


class RetrievedChunk(BaseModel):
    paper_id: str
    page: int
    chunk_idx: int
    text: str
    score: float


class Citation(BaseModel):
    paper_id: str
    page: int
    snippet: str


class QueryResponse(BaseModel):
    """Non-streaming response shape — kept for testability and the eval harness.

    The /query endpoint streams Server-Sent Events whose final event payload
    matches this schema.
    """

    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
