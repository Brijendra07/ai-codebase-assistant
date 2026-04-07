"""Hybrid keyword and vector retrieval."""

import re

from app.db.models import ChunkRecord


TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")
IMPLEMENTATION_HINTS = {
    "where",
    "implemented",
    "implementation",
    "function",
    "class",
    "api",
    "endpoint",
    "logic",
    "handler",
    "middleware",
    "service",
    "route",
}
FLOW_HINTS = {
    "flow",
    "trace",
    "path",
    "request",
    "how",
}


def rerank_results(
    query: str,
    matches: list[tuple[ChunkRecord, float]],
    top_k: int,
) -> list[tuple[ChunkRecord, float]]:
    query_terms = _tokenize(query)
    rescored: list[tuple[ChunkRecord, float]] = []
    prefer_implementation = bool(query_terms & IMPLEMENTATION_HINTS)
    prefer_flow = bool(query_terms & FLOW_HINTS)

    for chunk, semantic_score in matches:
        keyword_score = _keyword_score(query_terms, chunk)
        implementation_bonus = _implementation_bonus(chunk, prefer_implementation)
        documentation_penalty = _documentation_penalty(chunk, prefer_implementation, prefer_flow)
        final_score = semantic_score + keyword_score + implementation_bonus - documentation_penalty
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


def _implementation_bonus(chunk: ChunkRecord, prefer_implementation: bool) -> float:
    if chunk.chunk_type in {"function", "class"}:
        return 0.3 if prefer_implementation else 0.2

    if chunk.file_path.endswith(".py") and prefer_implementation:
        return 0.1

    return 0.0


def _documentation_penalty(
    chunk: ChunkRecord,
    prefer_implementation: bool,
    prefer_flow: bool,
) -> float:
    if not prefer_implementation and not prefer_flow:
        return 0.0

    doc_extensions = (".md", ".txt")
    if chunk.file_path.endswith(doc_extensions):
        return 0.35 if prefer_implementation else 0.15

    if chunk.chunk_type == "block" and chunk.file_path.endswith("__init__.py"):
        return 0.2

    return 0.0


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def apply_metadata_filters(
    matches: list[tuple[ChunkRecord, float]],
    language: str | None = None,
    chunk_types: list[str] | None = None,
    file_path_contains: str | None = None,
) -> list[tuple[ChunkRecord, float]]:
    filtered = matches

    if language:
        filtered = [
            (chunk, score)
            for chunk, score in filtered
            if chunk.language.lower() == language.lower()
        ]

    if chunk_types:
        allowed = {chunk_type.lower() for chunk_type in chunk_types}
        filtered = [
            (chunk, score)
            for chunk, score in filtered
            if chunk.chunk_type.lower() in allowed
        ]

    if file_path_contains:
        needle = file_path_contains.lower()
        filtered = [
            (chunk, score)
            for chunk, score in filtered
            if needle in chunk.file_path.lower()
        ]

    return filtered
