"""Persist and query vector embeddings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.db.models import ChunkRecord

try:
    import faiss
except ImportError:  # pragma: no cover - depends on local environment
    faiss = None


@dataclass
class StoredIndex:
    repo_name: str
    repo_path: str
    chunks: list[ChunkRecord]
    vectors: np.ndarray
    dimension: int
    backend: str
    index: object | None = None


_INDEX_REGISTRY: dict[str, StoredIndex] = {}


def build_vector_index(
    repo_path: str,
    repo_name: str,
    chunks: list[ChunkRecord],
    vectors: np.ndarray,
) -> StoredIndex:
    resolved_path = str(Path(repo_path).expanduser().resolve())

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Vectors must align one-to-one with chunks.")

    dimension = int(vectors.shape[1]) if vectors.size else 0

    if faiss is not None and dimension > 0:
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        stored_index = StoredIndex(
            repo_name=repo_name,
            repo_path=resolved_path,
            chunks=chunks,
            vectors=vectors,
            dimension=dimension,
            backend="faiss",
            index=index,
        )
    else:
        stored_index = StoredIndex(
            repo_name=repo_name,
            repo_path=resolved_path,
            chunks=chunks,
            vectors=vectors,
            dimension=dimension,
            backend="numpy",
        )

    _INDEX_REGISTRY[resolved_path] = stored_index
    return stored_index


def get_vector_index(repo_path: str) -> StoredIndex | None:
    resolved_path = str(Path(repo_path).expanduser().resolve())
    return _INDEX_REGISTRY.get(resolved_path)


def search_vector_index(
    stored_index: StoredIndex,
    query_vector: np.ndarray,
    top_k: int,
) -> list[tuple[ChunkRecord, float]]:
    if stored_index.dimension == 0 or not stored_index.chunks:
        return []

    query_vector = query_vector.astype("float32")
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    if stored_index.backend == "faiss" and stored_index.index is not None:
        scores, indices = stored_index.index.search(query_vector, top_k)
        return _format_results(stored_index.chunks, indices[0], scores[0])

    scores = stored_index.vectors @ query_vector[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (stored_index.chunks[index], float(scores[index]))
        for index in top_indices
        if index < len(stored_index.chunks)
    ]


def _format_results(
    chunks: list[ChunkRecord],
    indices: np.ndarray,
    scores: np.ndarray,
) -> list[tuple[ChunkRecord, float]]:
    results: list[tuple[ChunkRecord, float]] = []

    for index, score in zip(indices.tolist(), scores.tolist(), strict=False):
        if index < 0 or index >= len(chunks):
            continue
        results.append((chunks[index], float(score)))

    return results
