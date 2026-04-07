from fastapi import APIRouter

from app.db.models import AskRequest, AskResponse, SemanticSearchRequest, SemanticSearchResponse
from app.retrieval.retriever import answer_repository_question, semantic_search


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/search", response_model=SemanticSearchResponse)
async def search_repository(payload: SemanticSearchRequest) -> SemanticSearchResponse:
    return semantic_search(
        repo_path=payload.repo_path,
        query=payload.query,
        top_k=payload.top_k,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_repository(payload: AskRequest) -> AskResponse:
    return answer_repository_question(
        repo_path=payload.repo_path,
        question=payload.question,
        top_k=payload.top_k,
    )
