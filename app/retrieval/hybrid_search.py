"""Hybrid keyword and vector retrieval."""

import re

from app.db.models import ChunkRecord


TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")


def rerank_results(
    query: str,
    matches: list[tuple[ChunkRecord, float]],
    top_k: int,
) -> list[tuple[ChunkRecord, float]]:
    query_terms = _tokenize(query)
    rescored: list[tuple[ChunkRecord, float]] = []

    for chunk, semantic_score in matches:
        keyword_score = _keyword_score(query_terms, chunk)
        implementation_bonus = _implementation_bonus(chunk)
        final_score = semantic_score + keyword_score + implementation_bonus
        rescored.append((chunk, float(final_score)))

    rescored.sort(key=lambda item: item[1], reverse=True)
    return rescored[:top_k]


def _keyword_score(query_terms: set[str], chunk: ChunkRecord) -> float:
    if not query_terms:
        return 0.0

    path_terms = _tokenize(chunk.file_path.replace("/", " ").replace(".", " "))
    symbol_terms = _tokenize(chunk.symbol_name or "")
    content_terms = _tokenize(chunk.content)

    score = 0.0
    score += 0.35 * len(query_terms & path_terms)
    score += 0.45 * len(query_terms & symbol_terms)
    score += 0.12 * len(query_terms & content_terms)
    return score


def _implementation_bonus(chunk: ChunkRecord) -> float:
    if chunk.chunk_type in {"function", "class"}:
        return 0.2
    return 0.0


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}
