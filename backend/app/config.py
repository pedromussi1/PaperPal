from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider: "ollama" (local, free, default) or "groq" (cloud, free tier)
    llm_provider: str = "ollama"

    # Ollama-specific (used when llm_provider == "ollama")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_request_timeout: float = 120.0

    # Local vision-language model for /query/image. Must be a vision-capable
    # Ollama model that you've already pulled (e.g. `ollama pull llama3.2-vision:11b`).
    vision_model: str = "llama3.2-vision:11b"
    vision_request_timeout: float = 180.0

    # Groq-specific (used when llm_provider == "groq")
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Two-stage retrieval: pull `retrieve_top_k` candidates from the bi-encoder
    # store, then rerank with a cross-encoder down to the final `top_k` the LLM
    # sees. Setting `reranker_model=""` disables reranking (single-stage),
    # which is the canonical A/B baseline.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    retrieve_top_k: int = Field(24, gt=0)

    # Hybrid retrieval: run dense (bi-encoder) and BM25 in parallel and fuse
    # the two ranked lists with Reciprocal Rank Fusion before the cross-encoder
    # rerank step. Helps on queries that hinge on rare/exact terms (named
    # entities, formulas, numeric IDs) where pure dense often misses — but
    # only on multi-document corpora. Off by default until the scaled eval
    # (v0.9.0) characterizes the gain on a realistic dataset.
    hybrid_retrieval: bool = False
    rrf_k: int = Field(60, gt=0)

    # Parent-child chunking: narrow `chunk_size` chunks get embedded for
    # precise retrieval matching; the LLM sees a wider window of the
    # surrounding text (`parent_window` chunks on each side) for grounding.
    # See app/ingest.py.
    chunk_size: int = Field(400, gt=0)
    chunk_overlap: int = Field(80, ge=0)
    parent_window: int = Field(1, ge=0)

    top_k: int = Field(8, gt=0)

    chroma_persist_dir: Path = Path("./data/chroma")

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
