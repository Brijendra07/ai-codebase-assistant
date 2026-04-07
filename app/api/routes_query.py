from fastapi import APIRouter

from app.db.models import (
    AskRequest,
    AskResponse,
    RetrievalComparisonResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.retrieval.retriever import (
    answer_repository_question,
    compare_retrieval_strategies,
    semantic_search,
)


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/search", response_model=SemanticSearchResponse)
async def search_repository(payload: SemanticSearchRequest) -> SemanticSearchResponse:
    return semantic_search(
        repo_path=payload.repo_path,
        query=payload.query,
        top_k=payload.top_k,
        language=payload.language,
        chunk_types=payload.chunk_types,
        file_path_contains=payload.file_path_contains,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_repository(payload: AskRequest) -> AskResponse:
    return answer_repository_question(
        repo_path=payload.repo_path,
        question=payload.question,
        top_k=payload.top_k,
        language=payload.language,
        chunk_types=payload.chunk_types,
        file_path_contains=payload.file_path_contains,
    )


@router.post("/compare-retrieval", response_model=RetrievalComparisonResponse)
async def compare_retrieval(payload: SemanticSearchRequest) -> RetrievalComparisonResponse:
    return compare_retrieval_strategies(
        repo_path=payload.repo_path,
        query=payload.query,
        top_k=payload.top_k,
        language=payload.language,
        chunk_types=payload.chunk_types,
        file_path_contains=payload.file_path_contains,
    )
