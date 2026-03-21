from fastapi import APIRouter
from fastapi import status

from app.exceptions import EmptyIndexError, LLMError
from app.api.schemas import QueryRequest, QueryResponse, SourceDocument
from app.rag.pipeline import query
from app.rag.vectorstore import get_vectorstore


router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
async def query_documents(request: QueryRequest):
    # fail fast if nothing is indexed
    if get_vectorstore()._collection.count() == 0:
        raise EmptyIndexError()

    try:
        result = query(request.question)
    except Exception as e:
        raise LLMError(str(e))

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
    )