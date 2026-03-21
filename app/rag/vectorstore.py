import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embedder import get_embedder


def get_vectorstore() -> Chroma:
    settings = get_settings()

    chroma_client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    return Chroma(
        client=chroma_client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embedder(),
    )


def add_documents(docs: list[Document]) -> list[str]:
    vectorstore = get_vectorstore()
    ids = vectorstore.add_documents(docs)
    return ids


def delete_by_source(source: str) -> int:
    """
    Delete all chunks belonging to a specific source file.
    Called before re-ingesting so we never have duplicates.
    Returns number of chunks deleted.
    """
    vectorstore = get_vectorstore()

    # get all chunk IDs where metadata.source matches
    results = vectorstore.get(
        where={"source": source},
        include=[],             # we only need the IDs, not content
    )

    ids = results.get("ids", [])

    if ids:
        vectorstore.delete(ids=ids)
        print(f"[vectorstore] deleted {len(ids)} chunks for source='{source}'")

    return len(ids)


def similarity_search(query: str, k: int = 20) -> list[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def clear() -> None:
    settings = get_settings()
    get_vectorstore().delete_collection()
    print(f"[vectorstore] cleared collection: {settings.chroma_collection}")