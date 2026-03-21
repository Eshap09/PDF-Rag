from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(1024, ge=64, le=8192)

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = Field(500, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=500)

    # Retrieval
    retrieval_top_k: int = Field(5, ge=1, le=20)
    retrieval_candidate_k: int = Field(20, ge=5, le=100)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "rag_documents"


    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Ingestion
    max_upload_mb: int = Field(50, ge=1, le=500)
    allowed_extensions: list[str] = [".txt", ".md", ".pdf"]


@lru_cache
def get_settings() -> Settings:
    return Settings()