from fastapi import APIRouter

from app.agents.orchestrator import (
    run_cleanup_candidates,
    run_compare_files,
    run_explain_flow,
    run_trace_symbol,
)
from app.db.models import (
    AskRequest,
    AskResponse,
    CleanupCandidatesRequest,
    CleanupCandidatesResponse,
    CompareFilesRequest,
    CompareFilesResponse,
    ExplainFlowRequest,
    ExplainFlowResponse,
    RetrievalSettings,
    RetrievalComparisonResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    TraceSymbolRequest,
    TraceSymbolResponse,
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


@router.post("/explain-flow", response_model=ExplainFlowResponse)
async def explain_flow(payload: ExplainFlowRequest) -> ExplainFlowResponse:
    return run_explain_flow(
        repo_path=payload.repo_path,
        question=payload.question,
        retrieval_settings=RetrievalSettings(
            top_k=payload.top_k,
            language=payload.language,
            chunk_types=payload.chunk_types,
            file_path_contains=payload.file_path_contains,
        ),
    )


@router.post("/compare-files", response_model=CompareFilesResponse)
async def compare_files(payload: CompareFilesRequest) -> CompareFilesResponse:
    return run_compare_files(
        repo_path=payload.repo_path,
        file_path_a=payload.file_path_a,
        file_path_b=payload.file_path_b,
    )


@router.post("/trace-symbol", response_model=TraceSymbolResponse)
async def trace_symbol(payload: TraceSymbolRequest) -> TraceSymbolResponse:
    return run_trace_symbol(
        repo_path=payload.repo_path,
        symbol=payload.symbol,
        top_k=payload.top_k,
    )


@router.post("/cleanup-candidates", response_model=CleanupCandidatesResponse)
async def cleanup_candidates(payload: CleanupCandidatesRequest) -> CleanupCandidatesResponse:
    return run_cleanup_candidates(
        repo_path=payload.repo_path,
        top_k=payload.top_k,
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
