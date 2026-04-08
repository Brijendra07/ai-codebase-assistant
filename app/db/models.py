"""Persistence and API-facing models for repositories, chunks, and results."""

from datetime import datetime

from pydantic import BaseModel, Field


class RepositoryIndexRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")


class FileRecord(BaseModel):
    path: str
    extension: str
    size_bytes: int


class ParsedFile(BaseModel):
    path: str
    language: str
    content: str
    line_count: int


class ChunkRecord(BaseModel):
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None = None
    start_line: int
    end_line: int
    content: str


class RepositoryIndexResponse(BaseModel):
    repo_id: str
    repo_name: str
    repo_path: str
    indexed_at: datetime
    total_files_scanned: int
    total_files_indexed: int
    skipped_files: int
    files: list[FileRecord]


class RepositoryChunkResponse(BaseModel):
    repo_name: str
    repo_path: str
    total_files_parsed: int
    total_chunks: int
    chunks: list[ChunkRecord]


class RepositoryEmbeddingResponse(BaseModel):
    repo_name: str
    repo_path: str
    embedding_model: str
    total_chunks_indexed: int
    vector_dimension: int
    backend: str


class SemanticSearchRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of results to return")
    language: str | None = Field(None, description="Optional language filter like python or javascript")
    chunk_types: list[str] | None = Field(None, description="Optional chunk type filters like function, class, or block")
    file_path_contains: str | None = Field(None, description="Optional file path substring filter")


class SearchResult(BaseModel):
    score: float
    chunk: ChunkRecord


class RetrievalSettings(BaseModel):
    top_k: int
    language: str | None = None
    chunk_types: list[str] | None = None
    file_path_contains: str | None = None


class SemanticSearchResponse(BaseModel):
    repo_name: str
    repo_path: str
    query: str
    top_k: int
    retrieval_backend: str
    retrieval_settings: RetrievalSettings
    latency_ms: float
    total_results: int
    results: list[SearchResult]


class RetrievalComparisonStage(BaseModel):
    stage: str
    total_results: int
    results: list[SearchResult]


class RetrievalComparisonResponse(BaseModel):
    repo_name: str
    repo_path: str
    query: str
    top_k: int
    retrieval_settings: RetrievalSettings
    latency_ms: float
    stages: list[RetrievalComparisonStage]


class EvalCase(BaseModel):
    name: str
    query: str
    expected_file_paths: list[str]
    language: str | None = None
    chunk_types: list[str] | None = None
    file_path_contains: str | None = None


class EvalRunRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    top_k: int = Field(5, ge=1, le=20, description="Top-k cutoff for retrieval evaluation")
    cases: list[EvalCase] | None = Field(
        None,
        description="Optional custom eval cases. If omitted, built-in sample cases are used.",
    )


class EvalCaseResult(BaseModel):
    name: str
    query: str
    expected_file_paths: list[str]
    retrieved_file_paths: list[str]
    hit: bool
    hit_at_k: float
    retrieval_latency_ms: float
    latency_ms: float


class EvalRunResponse(BaseModel):
    run_id: str | None = None
    created_at: datetime | None = None
    repo_name: str
    repo_path: str
    top_k: int
    latency_ms: float
    total_cases: int
    hits: int
    hit_rate: float
    results: list[EvalCaseResult]


class EvalResultsResponse(BaseModel):
    total_runs: int
    results: list[EvalRunResponse]


class AnswerFeedbackRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    question: str = Field(..., min_length=1, description="Question that was asked")
    answer_mode: str = Field(..., description="Answer mode such as llm, fallback, or tool")
    rating: int = Field(..., ge=1, le=5, description="Feedback rating from 1 to 5")
    comments: str | None = Field(None, description="Optional user comments")


class AnswerFeedbackResponse(BaseModel):
    feedback_id: str
    created_at: datetime
    status: str


class CitationRecord(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str | None = None


class AskRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    question: str = Field(..., min_length=1, description="Natural language repository question")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of retrieved chunks to use")
    language: str | None = Field(None, description="Optional language filter like python or javascript")
    chunk_types: list[str] | None = Field(None, description="Optional chunk type filters like function, class, or block")
    file_path_contains: str | None = Field(None, description="Optional file path substring filter")


class AskResponse(BaseModel):
    repo_name: str
    repo_path: str
    question: str
    answer: str
    grounded: bool
    answer_mode: str
    retrieval_backend: str
    retrieval_settings: RetrievalSettings
    retrieval_latency_ms: float
    generation_latency_ms: float
    latency_ms: float
    citations: list[CitationRecord]


class ExplainFlowRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    question: str = Field(..., min_length=1, description="Flow-oriented question such as request flow or feature flow")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of retrieved chunks to use")
    language: str | None = Field(None, description="Optional language filter like python or javascript")
    chunk_types: list[str] | None = Field(None, description="Optional chunk type filters like function, class, or block")
    file_path_contains: str | None = Field(None, description="Optional file path substring filter")


class ToolStep(BaseModel):
    tool_name: str
    description: str
    output_summary: str


class ExplainFlowResponse(BaseModel):
    repo_name: str
    repo_path: str
    question: str
    flow_summary: str
    answer_mode: str
    retrieval_settings: RetrievalSettings
    retrieval_latency_ms: float
    generation_latency_ms: float
    latency_ms: float
    tool_steps: list[ToolStep]
    citations: list[CitationRecord]


class CompareFilesRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    file_path_a: str = Field(..., description="First repository-relative file path")
    file_path_b: str = Field(..., description="Second repository-relative file path")


class CompareFilesResponse(BaseModel):
    repo_name: str
    repo_path: str
    file_path_a: str
    file_path_b: str
    comparison_summary: str
    answer_mode: str
    generation_latency_ms: float
    latency_ms: float
    tool_steps: list[ToolStep]
    citations: list[CitationRecord]


class TraceSymbolRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    symbol: str = Field(..., min_length=1, description="Symbol name to trace across the repository")
    top_k: int = Field(10, ge=1, le=50, description="Maximum number of symbol matches to return")


class TraceSymbolResponse(BaseModel):
    repo_name: str
    repo_path: str
    symbol: str
    summary: str
    answer_mode: str
    latency_ms: float
    tool_steps: list[ToolStep]
    citations: list[CitationRecord]


class CleanupCandidatesRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local repository")
    top_k: int = Field(10, ge=1, le=50, description="Maximum number of cleanup candidates to return")


class CleanupCandidatesResponse(BaseModel):
    repo_name: str
    repo_path: str
    summary: str
    answer_mode: str
    latency_ms: float
    tool_steps: list[ToolStep]
    citations: list[CitationRecord]
