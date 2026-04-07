"""Semantic retrieval over indexed code chunks."""

from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import settings
from app.chunking.code_chunker import chunk_file
from app.db.models import (
    AskResponse,
    RepositoryEmbeddingResponse,
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
from app.retrieval.hybrid_search import rerank_results


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


def semantic_search(repo_path: str, query: str, top_k: int) -> SemanticSearchResponse:
    stored_index, matches = _retrieve_matches(repo_path, query, top_k)

    return SemanticSearchResponse(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        query=query,
        top_k=top_k,
        total_results=len(matches),
        results=[
            SearchResult(score=score, chunk=chunk)
            for chunk, score in matches
        ],
    )


def answer_repository_question(repo_path: str, question: str, top_k: int) -> AskResponse:
    stored_index, matches = _retrieve_matches(repo_path, question, top_k)
    search_results = [SearchResult(score=score, chunk=chunk) for chunk, score in matches]
    return generate_grounded_answer(
        repo_name=stored_index.repo_name,
        repo_path=stored_index.repo_path,
        question=question,
        results=search_results,
    )


def _retrieve_matches(repo_path: str, query: str, top_k: int):
    root = Path(repo_path).expanduser().resolve()
    stored_index = get_vector_index(str(root))

    if stored_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository embeddings not found. Run POST /repos/embed first.",
        )

    query_vector = embed_texts([query])
    candidate_count = min(max(top_k * 8, top_k), len(stored_index.chunks))
    matches = search_vector_index(stored_index, query_vector, candidate_count)
    matches = rerank_results(query, matches, top_k)
    return stored_index, matches
