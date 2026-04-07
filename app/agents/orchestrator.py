"""Coordinate tool-based repository analysis workflows."""

from pathlib import Path

from fastapi import HTTPException, status

from app.agents.workflows import (
    cleanup_candidates_workflow,
    compare_files_workflow,
    explain_flow_workflow,
    trace_symbol_workflow,
)
from app.db.models import (
    CleanupCandidatesResponse,
    CompareFilesResponse,
    ExplainFlowResponse,
    RetrievalSettings,
    TraceSymbolResponse,
)
from app.embeddings.vector_store import get_vector_index


def run_explain_flow(
    repo_path: str,
    question: str,
    retrieval_settings: RetrievalSettings,
) -> ExplainFlowResponse:
    root = Path(repo_path).expanduser().resolve()
    stored_index = get_vector_index(str(root))

    if stored_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository embeddings not found. Run POST /repos/embed first.",
        )

    return explain_flow_workflow(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        question=question,
        retrieval_settings=retrieval_settings,
    )


def run_compare_files(
    repo_path: str,
    file_path_a: str,
    file_path_b: str,
) -> CompareFilesResponse:
    root = Path(repo_path).expanduser().resolve()
    stored_index = get_vector_index(str(root))

    if stored_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository embeddings not found. Run POST /repos/embed first.",
        )

    return compare_files_workflow(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        file_path_a=file_path_a,
        file_path_b=file_path_b,
    )


def run_trace_symbol(
    repo_path: str,
    symbol: str,
    top_k: int,
) -> TraceSymbolResponse:
    root = Path(repo_path).expanduser().resolve()
    stored_index = get_vector_index(str(root))

    if stored_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository embeddings not found. Run POST /repos/embed first.",
        )

    return trace_symbol_workflow(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        symbol=symbol,
        top_k=top_k,
    )


def run_cleanup_candidates(
    repo_path: str,
    top_k: int,
) -> CleanupCandidatesResponse:
    root = Path(repo_path).expanduser().resolve()
    stored_index = get_vector_index(str(root))

    if stored_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository embeddings not found. Run POST /repos/embed first.",
        )

    return cleanup_candidates_workflow(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        top_k=top_k,
    )
