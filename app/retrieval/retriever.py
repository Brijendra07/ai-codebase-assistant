"""Semantic retrieval over indexed code chunks."""

import time
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logger import get_logger
from app.chunking.code_chunker import chunk_file
from app.db.models import (
    AskResponse,
    RepositoryEmbeddingResponse,
    RetrievalSettings,
    RetrievalComparisonResponse,
    RetrievalComparisonStage,
    SearchResult,
    SemanticSearchResponse,
)
from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import (
    build_vector_index,
    get_vector_index,
    search_vector_index,
)
from app.ingestion.parser import parse_file
from app.ingestion.repo_loader import list_repository_files
from app.llm.answer_generator import generate_grounded_answer
from app.retrieval.hybrid_search import apply_metadata_filters, rerank_results
from app.retrieval.llamaindex_store import get_or_build_llamaindex_index, search_llamaindex_index

logger = get_logger(__name__)


def index_repository_embeddings(repo_path: str) -> RepositoryEmbeddingResponse:
    root = Path(repo_path).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid repository path: {root}",
        )

    file_records, _ = list_repository_files(root)

    chunks = []
    for file_record in file_records:
        parsed_file = parse_file(root, root / file_record.path)
        chunks.extend(chunk_file(parsed_file))

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks were generated for this repository.",
        )

    vectors = embed_texts([chunk.content for chunk in chunks])
    stored_index = build_vector_index(
        repo_path=str(root),
        repo_name=root.name,
        chunks=chunks,
        vectors=vectors,
    )

    return RepositoryEmbeddingResponse(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        embedding_model=settings.embedding_model_name,
        total_chunks_indexed=len(stored_index.chunks),
        vector_dimension=stored_index.dimension,
        backend=stored_index.backend,
    )


def semantic_search(
    repo_path: str,
    query: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
) -> SemanticSearchResponse:
    started_at = time.perf_counter()
    stored_index, matches = _retrieve_matches(
        repo_path,
        query,
        top_k,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )

    return SemanticSearchResponse(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        query=query,
        top_k=top_k,
        retrieval_backend=stored_index.backend,
        retrieval_settings=_build_retrieval_settings(top_k, language, chunk_types, file_path_contains),
        latency_ms=_elapsed_ms(started_at),
        total_results=len(matches),
        results=[
            SearchResult(score=score, chunk=chunk)
            for chunk, score in matches
        ],
    )


def answer_repository_question(
    repo_path: str,
    question: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
) -> AskResponse:
    retrieval_started_at = time.perf_counter()
    stored_index, matches = _retrieve_matches(
        repo_path,
        question,
        top_k,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )
    retrieval_latency_ms = _elapsed_ms(retrieval_started_at)
    search_results = [SearchResult(score=score, chunk=chunk) for chunk, score in matches]
    return generate_grounded_answer(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        question=question,
        results=search_results,
        retrieval_backend=stored_index.backend,
        retrieval_settings=_build_retrieval_settings(top_k, language, chunk_types, file_path_contains),
        retrieval_latency_ms=retrieval_latency_ms,
    )


def semantic_search_llamaindex(
    repo_path: str,
    query: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
) -> SemanticSearchResponse:
    started_at = time.perf_counter()
    stored_index, matches = _retrieve_matches_llamaindex(
        repo_path,
        query,
        top_k,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )

    return SemanticSearchResponse(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        query=query,
        top_k=top_k,
        retrieval_backend=stored_index.backend,
        retrieval_settings=_build_retrieval_settings(top_k, language, chunk_types, file_path_contains),
        latency_ms=_elapsed_ms(started_at),
        total_results=len(matches),
        results=[SearchResult(score=score, chunk=chunk) for chunk, score in matches],
    )


def answer_repository_question_llamaindex(
    repo_path: str,
    question: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
) -> AskResponse:
    retrieval_started_at = time.perf_counter()
    stored_index, matches = _retrieve_matches_llamaindex(
        repo_path,
        question,
        top_k,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )
    retrieval_latency_ms = _elapsed_ms(retrieval_started_at)
    search_results = [SearchResult(score=score, chunk=chunk) for chunk, score in matches]
    return generate_grounded_answer(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        question=question,
        results=search_results,
        retrieval_backend=stored_index.backend,
        retrieval_settings=_build_retrieval_settings(top_k, language, chunk_types, file_path_contains),
        retrieval_latency_ms=retrieval_latency_ms,
    )


def compare_retrieval_strategies(
    repo_path: str,
    query: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
) -> RetrievalComparisonResponse:
    started_at = time.perf_counter()
    stored_index = _get_stored_index(repo_path)
    query_vector = embed_texts([query])
    candidate_count = min(max(top_k * 8, top_k), len(stored_index.chunks))

    raw_matches = search_vector_index(stored_index, query_vector, candidate_count)
    filtered_matches = apply_metadata_filters(
        raw_matches,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )
    reranked_matches = rerank_results(
        query,
        filtered_matches if filtered_matches else raw_matches,
        top_k,
    )
    llama_index, llama_matches = _retrieve_matches_llamaindex(
        repo_path,
        query,
        top_k,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )

    return RetrievalComparisonResponse(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        query=query,
        top_k=top_k,
        retrieval_settings=_build_retrieval_settings(top_k, language, chunk_types, file_path_contains),
        latency_ms=_elapsed_ms(started_at),
        stages=[
            _build_stage("semantic_raw", raw_matches[:top_k]),
            _build_stage("metadata_filtered", filtered_matches[:top_k]),
            _build_stage("reranked_final", reranked_matches),
            _build_stage(llama_index.backend, llama_matches),
        ],
    )


def _retrieve_matches(
    repo_path: str,
    query: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
):
    stored_index = _get_stored_index(repo_path)
    query_vector = embed_texts([query])
    candidate_count = min(max(top_k * 8, top_k), len(stored_index.chunks))
    matches = search_vector_index(stored_index, query_vector, candidate_count)
    matches = apply_metadata_filters(
        matches,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )
    if not matches:
        matches = search_vector_index(stored_index, query_vector, candidate_count)
    matches = rerank_results(query, matches, top_k)
    logger.info(
        "retrieval_run query=%r top_k=%s language=%s chunk_types=%s file_path_contains=%s results=%s",
        query,
        top_k,
        language,
        chunk_types,
        file_path_contains,
        len(matches),
    )
    return stored_index, matches


def _retrieve_matches_llamaindex(
    repo_path: str,
    query: str,
    top_k: int,
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
):
    base_index = _get_stored_index(repo_path)
    stored_index = get_or_build_llamaindex_index(
        repo_path=base_index.repo_path,
        repo_name=base_index.repo_name,
        chunks=base_index.chunks,
    )
    candidate_count = min(max(top_k * 8, top_k), len(stored_index.chunks))
    matches = search_llamaindex_index(stored_index, query, candidate_count)
    matches = apply_metadata_filters(
        matches,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )
    if not matches:
        matches = search_llamaindex_index(stored_index, query, candidate_count)
    matches = rerank_results(query, matches, top_k)
    logger.info(
        "retrieval_run_llamaindex query=%r top_k=%s language=%s chunk_types=%s file_path_contains=%s results=%s",
        query,
        top_k,
        language,
        chunk_types,
        file_path_contains,
        len(matches),
    )
    return stored_index, matches


def _get_stored_index(repo_path: str):
    root = Path(repo_path).expanduser().resolve()
    stored_index = get_vector_index(str(root))

    if stored_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository embeddings not found. Run POST /repos/embed first.",
        )

    return stored_index


def _build_stage(stage: str, matches: list[tuple]) -> RetrievalComparisonStage:
    results = [SearchResult(score=score, chunk=chunk) for chunk, score in matches]
    return RetrievalComparisonStage(
        stage=stage,
        total_results=len(results),
        results=results,
    )


def _build_retrieval_settings(
    top_k: int,
    language: str | None,
    chunk_types: list[str] | None,
    file_path_contains: str | None,
) -> RetrievalSettings:
    return RetrievalSettings(
        top_k=top_k,
        language=language,
        chunk_types=chunk_types,
        file_path_contains=file_path_contains,
    )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
