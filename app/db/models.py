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


class SearchResult(BaseModel):
    score: float
    chunk: ChunkRecord


class SemanticSearchResponse(BaseModel):
    repo_name: str
    repo_path: str
    query: str
    top_k: int
    total_results: int
    results: list[SearchResult]


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


class AskResponse(BaseModel):
    repo_name: str
    repo_path: str
    question: str
    answer: str
    grounded: bool
    answer_mode: str
    citations: list[CitationRecord]
