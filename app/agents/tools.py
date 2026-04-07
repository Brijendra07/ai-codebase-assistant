"""Agent tools for repository search and inspection."""

import re
from pathlib import Path

from fastapi import HTTPException, status

from app.db.models import CitationRecord, ParsedFile, RetrievalSettings, SearchResult
from app.ingestion.parser import parse_file
from app.ingestion.repo_loader import list_repository_files
from app.retrieval.retriever import semantic_search


COMMENT_PREFIXES_BY_LANGUAGE = {
    "python": ("#",),
    "javascript": ("//", "/*", "*"),
    "typescript": ("//", "/*", "*"),
    "java": ("//", "/*", "*"),
    "go": ("//", "/*", "*"),
    "rust": ("//", "/*", "*"),
    "cpp": ("//", "/*", "*"),
    "c": ("//", "/*", "*"),
    "c-header": ("//", "/*", "*"),
    "cpp-header": ("//", "/*", "*"),
    "csharp": ("//", "/*", "*"),
    "ruby": ("#",),
    "php": ("//", "#", "/*", "*"),
    "swift": ("//", "/*", "*"),
    "kotlin": ("//", "/*", "*"),
    "scala": ("//", "/*", "*"),
    "sql": ("--", "/*", "*"),
    "markdown": ("<!--",),
    "text": ("#", "//", "--"),
}

MARKER_PATTERN = re.compile(r"\b(TODO|FIXME)\b", re.IGNORECASE)


def search_code_chunks(
    repo_path: str,
    query: str,
    retrieval_settings: RetrievalSettings,
) -> tuple[list[SearchResult], float]:
    response = semantic_search(
        repo_path=repo_path,
        query=query,
        top_k=retrieval_settings.top_k,
        language=retrieval_settings.language,
        chunk_types=retrieval_settings.chunk_types,
        file_path_contains=retrieval_settings.file_path_contains,
    )
    return response.results, response.latency_ms


def summarize_search_results(results: list[SearchResult], limit: int = 3) -> str:
    if not results:
        return "No relevant code chunks were found."

    lines: list[str] = []
    for result in results[:limit]:
        chunk = result.chunk
        symbol = f" ({chunk.symbol_name})" if chunk.symbol_name else ""
        lines.append(
            f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"[{chunk.chunk_type}{symbol}]"
        )

    return "; ".join(lines)


def read_repository_file(repo_path: str, relative_file_path: str) -> ParsedFile:
    root = Path(repo_path).expanduser().resolve()
    file_path = (root / relative_file_path).resolve()

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository file not found: {relative_file_path}",
        )

    try:
        file_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File path is outside repository: {relative_file_path}",
        ) from exc

    return parse_file(root, file_path)


def summarize_file(parsed_file: ParsedFile, max_chars: int = 240) -> str:
    compact = " ".join(parsed_file.content.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def find_symbol_references(
    repo_path: str,
    symbol: str,
    top_k: int = 10,
) -> list[CitationRecord]:
    root = Path(repo_path).expanduser().resolve()
    file_records, _ = list_repository_files(root)
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    matches: list[CitationRecord] = []

    for file_record in file_records:
        parsed_file = parse_file(root, root / file_record.path)
        for line_number, line in enumerate(parsed_file.content.splitlines(), start=1):
            if not pattern.search(line):
                continue
            matches.append(
                CitationRecord(
                    file_path=parsed_file.path,
                    start_line=line_number,
                    end_line=line_number,
                    chunk_type="reference",
                    symbol_name=symbol,
                )
            )
            if len(matches) >= top_k:
                return matches

    return matches


def find_cleanup_candidates(
    repo_path: str,
    top_k: int = 10,
) -> list[tuple[CitationRecord, str]]:
    root = Path(repo_path).expanduser().resolve()
    file_records, _ = list_repository_files(root)
    candidates: list[tuple[CitationRecord, str]] = []

    for file_record in file_records:
        parsed_file = parse_file(root, root / file_record.path)
        lines = parsed_file.content.splitlines()

        for line_number, line in enumerate(lines, start=1):
            if _is_cleanup_comment(parsed_file.language, line):
                candidates.append(
                    (
                        CitationRecord(
                            file_path=parsed_file.path,
                            start_line=line_number,
                            end_line=line_number,
                            chunk_type="cleanup-marker",
                            symbol_name=None,
                        ),
                        f"Contains {line.strip()}",
                    )
                )

        if parsed_file.line_count > 120:
            candidates.append(
                (
                    CitationRecord(
                        file_path=parsed_file.path,
                        start_line=1,
                        end_line=parsed_file.line_count,
                        chunk_type="large-file",
                        symbol_name=None,
                    ),
                    f"Large file with {parsed_file.line_count} lines",
                )
            )

        if len(candidates) >= top_k:
            break

    return candidates[:top_k]


def _is_cleanup_comment(language: str, line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    prefixes = COMMENT_PREFIXES_BY_LANGUAGE.get(language, ("#", "//", "--", "/*", "*"))
    has_comment_prefix = any(stripped.startswith(prefix) for prefix in prefixes)

    if not has_comment_prefix:
        return False

    return bool(MARKER_PATTERN.search(stripped))
