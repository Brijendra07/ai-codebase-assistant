"""Structured response schemas for answer generation."""

from app.db.models import CitationRecord


def build_citations_from_results(citations: list[CitationRecord]) -> list[dict[str, str | int | None]]:
    return [
        {
            "file_path": citation.file_path,
            "start_line": citation.start_line,
            "end_line": citation.end_line,
            "chunk_type": citation.chunk_type,
            "symbol_name": citation.symbol_name,
        }
        for citation in citations
    ]
