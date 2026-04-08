"""Higher-level agent workflows for multi-step analysis."""

import time

from app.agents.tools import (
    find_cleanup_candidates,
    find_symbol_references,
    read_repository_file,
    search_code_chunks,
    summarize_file,
    summarize_search_results,
)
from app.db.models import (
    CitationRecord,
    CleanupCandidatesResponse,
    CompareFilesResponse,
    ExplainFlowResponse,
    RetrievalSettings,
    ToolStep,
    TraceSymbolResponse,
)
from app.llm.answer_generator import generate_grounded_answer


def explain_flow_workflow(
    repo_name: str,
    repo_path: str,
    question: str,
    retrieval_settings: RetrievalSettings,
) -> ExplainFlowResponse:
    results, retrieval_latency_ms = search_code_chunks(
        repo_path=repo_path,
        query=question,
        retrieval_settings=retrieval_settings,
    )

    answer_response = generate_grounded_answer(
        repo_name=repo_name,
        repo_path=repo_path,
        question=question,
        results=results,
        retrieval_backend="faiss",
        retrieval_settings=retrieval_settings,
        retrieval_latency_ms=retrieval_latency_ms,
    )

    tool_steps = [
        ToolStep(
            tool_name="search_code_chunks",
            description="Retrieve flow-relevant implementation chunks using metadata-aware search.",
            output_summary=summarize_search_results(results),
        ),
        ToolStep(
            tool_name="generate_grounded_answer",
            description="Synthesize a grounded flow explanation from retrieved chunks.",
            output_summary=answer_response.answer,
        ),
    ]

    return ExplainFlowResponse(
        repo_name=repo_name,
        repo_path=repo_path,
        question=question,
        flow_summary=answer_response.answer,
        answer_mode=answer_response.answer_mode,
        retrieval_settings=retrieval_settings,
        retrieval_latency_ms=answer_response.retrieval_latency_ms,
        generation_latency_ms=answer_response.generation_latency_ms,
        latency_ms=answer_response.latency_ms,
        tool_steps=tool_steps,
        citations=[
            CitationRecord(
                file_path=citation.file_path,
                start_line=citation.start_line,
                end_line=citation.end_line,
                chunk_type=citation.chunk_type,
                symbol_name=citation.symbol_name,
            )
            for citation in answer_response.citations
        ],
    )


def compare_files_workflow(
    repo_name: str,
    repo_path: str,
    file_path_a: str,
    file_path_b: str,
) -> CompareFilesResponse:
    started_at = time.perf_counter()
    parsed_a = read_repository_file(repo_path, file_path_a)
    parsed_b = read_repository_file(repo_path, file_path_b)

    summary_a = summarize_file(parsed_a)
    summary_b = summarize_file(parsed_b)
    comparison_summary = _build_file_comparison(parsed_a, parsed_b, summary_a, summary_b)
    generation_latency_ms = _elapsed_ms(started_at)

    tool_steps = [
        ToolStep(
            tool_name="read_repository_file",
            description="Load the first file for grounded inspection.",
            output_summary=f"{parsed_a.path} ({parsed_a.language}, {parsed_a.line_count} lines)",
        ),
        ToolStep(
            tool_name="read_repository_file",
            description="Load the second file for grounded inspection.",
            output_summary=f"{parsed_b.path} ({parsed_b.language}, {parsed_b.line_count} lines)",
        ),
        ToolStep(
            tool_name="compare_files",
            description="Compare file purpose, structure, and content summaries.",
            output_summary=comparison_summary,
        ),
    ]

    return CompareFilesResponse(
        repo_name=repo_name,
        repo_path=repo_path,
        file_path_a=parsed_a.path,
        file_path_b=parsed_b.path,
        comparison_summary=comparison_summary,
        answer_mode="tool",
        generation_latency_ms=generation_latency_ms,
        latency_ms=generation_latency_ms,
        tool_steps=tool_steps,
        citations=[
            CitationRecord(
                file_path=parsed_a.path,
                start_line=1,
                end_line=max(parsed_a.line_count, 1),
                chunk_type="file",
                symbol_name=None,
            ),
            CitationRecord(
                file_path=parsed_b.path,
                start_line=1,
                end_line=max(parsed_b.line_count, 1),
                chunk_type="file",
                symbol_name=None,
            ),
        ],
    )


def trace_symbol_workflow(
    repo_name: str,
    repo_path: str,
    symbol: str,
    top_k: int,
) -> TraceSymbolResponse:
    started_at = time.perf_counter()
    citations = find_symbol_references(repo_path, symbol, top_k=top_k)
    latency_ms = _elapsed_ms(started_at)

    if citations:
        summary = (
            f"Found {len(citations)} reference(s) for `{symbol}` across the repository. "
            f"The earliest matches are in "
            + ", ".join(f"{citation.file_path}:{citation.start_line}" for citation in citations[:3])
            + "."
        )
    else:
        summary = f"No references were found for `{symbol}`."

    tool_steps = [
        ToolStep(
            tool_name="find_symbol_references",
            description="Scan repository files for exact symbol references.",
            output_summary=summary,
        )
    ]

    return TraceSymbolResponse(
        repo_name=repo_name,
        repo_path=repo_path,
        symbol=symbol,
        summary=summary,
        answer_mode="tool",
        latency_ms=latency_ms,
        tool_steps=tool_steps,
        citations=citations,
    )


def cleanup_candidates_workflow(
    repo_name: str,
    repo_path: str,
    top_k: int,
) -> CleanupCandidatesResponse:
    started_at = time.perf_counter()
    candidates = find_cleanup_candidates(repo_path, top_k=top_k)
    latency_ms = _elapsed_ms(started_at)

    if candidates:
        summary = "Cleanup candidates found: " + "; ".join(
            f"{citation.file_path}:{citation.start_line}-{citation.end_line} ({reason})"
            for citation, reason in candidates
        )
    else:
        summary = "No cleanup candidates were detected with the current heuristics."

    tool_steps = [
        ToolStep(
            tool_name="find_cleanup_candidates",
            description="Look for TODO/FIXME markers and simple cleanup heuristics such as oversized files.",
            output_summary=summary,
        )
    ]

    return CleanupCandidatesResponse(
        repo_name=repo_name,
        repo_path=repo_path,
        summary=summary,
        answer_mode="tool",
        latency_ms=latency_ms,
        tool_steps=tool_steps,
        citations=[citation for citation, _ in candidates],
    )


def _build_file_comparison(parsed_a, parsed_b, summary_a: str, summary_b: str) -> str:
    differences: list[str] = []

    if parsed_a.language != parsed_b.language:
        differences.append(
            f"{parsed_a.path} is {parsed_a.language}, while {parsed_b.path} is {parsed_b.language}."
        )
    else:
        differences.append(
            f"Both files are {parsed_a.language} files, but they serve different roles."
        )

    differences.append(
        f"{parsed_a.path} has {parsed_a.line_count} lines, while {parsed_b.path} has {parsed_b.line_count} lines."
    )
    differences.append(f"{parsed_a.path} summary: {summary_a}")
    differences.append(f"{parsed_b.path} summary: {summary_b}")

    return " ".join(differences)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
