from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import get_settings


@lru_cache
def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns a cached embedder instance.
    Model is downloaded once on first call, cached locally after that.
    lru_cache ensures we never load the model twice in the same process.
    """
    settings = get_settings()

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )