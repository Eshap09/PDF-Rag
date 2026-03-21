from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.rag.vectorstore import similarity_search


# module-level cache — loaded once, reused across requests
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        settings = get_settings()
        print(f"[retriever] loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def retrieve(query: str) -> list[Document]:
    """
    Full retrieval pipeline:
      1. Vector search  — top 20 candidates by embedding similarity
      2. Rerank         — cross-encoder rescores each (query, chunk) pair
      3. Return top_k   — only the most relevant chunks go to the LLM
    """
    settings = get_settings()

    # Step 1 — vector search, fetch more than we need
    candidates = similarity_search(query, k=settings.retrieval_candidate_k)

    if not candidates:
        return []

    # Step 2 — rerank if enabled
    if settings.use_reranker and len(candidates) > 1:
        candidates = _rerank(query, candidates)

    # Step 3 — return top_k
    return candidates[:settings.retrieval_top_k]


def _rerank(query: str, docs: list[Document]) -> list[Document]:
    """
    Cross-encoder reranking.
    Scores each (query, chunk) pair together — much more accurate than
    vector similarity alone because it sees both texts at the same time.
    """
    reranker = get_reranker()

    # build pairs
    pairs = [(query, doc.page_content) for doc in docs]

    # score — returns a float per pair
    scores = reranker.predict(pairs)

    # attach scores and sort descending
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # return just the docs in reranked order
    return [doc for _, doc in scored]