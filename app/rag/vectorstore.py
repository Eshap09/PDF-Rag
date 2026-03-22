import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embedder import get_embedder


def get_vectorstore() -> Chroma:
    settings = get_settings()

    # if CHROMA_PERSIST_DIR is set → embedded mode (brew install / no Docker)
    # otherwise → Docker HTTP mode (local dev)
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR")

    if persist_dir:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
    else:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embedder(),
    )


def add_documents(docs: list[Document]) -> list[str]:
    return get_vectorstore().add_documents(docs)


def delete_by_source(source: str) -> int:
    vs      = get_vectorstore()
    results = vs.get(where={"source": source}, include=[])
    ids     = results.get("ids", [])
    if ids:
        vs.delete(ids=ids)
    return len(ids)


def similarity_search(query: str, k: int = 20) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=k)


def clear() -> None:
    get_vectorstore().delete_collection()