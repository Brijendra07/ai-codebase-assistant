"""Optional LlamaIndex-backed retrieval for side-by-side RAG comparison."""

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import settings
from app.db.models import ChunkRecord

try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:  # pragma: no cover - depends on local environment
    Document = None
    VectorStoreIndex = None
    HuggingFaceEmbedding = None


@dataclass
class StoredLlamaIndex:
    repo_name: str
    repo_path: str
    chunks: list[ChunkRecord]
    index: object
    backend: str = "llamaindex"


_LLAMA_INDEX_REGISTRY: dict[str, StoredLlamaIndex] = {}


def get_or_build_llamaindex_index(
    repo_path: str,
    repo_name: str,
    chunks: list[ChunkRecord],
) -> StoredLlamaIndex:
    resolved_path = str(Path(repo_path).expanduser().resolve())
    stored_index = _LLAMA_INDEX_REGISTRY.get(resolved_path)
    if stored_index is not None:
        return stored_index

    _ensure_llamaindex_available()
    documents = [
        Document(
            text=chunk.content,
            metadata={
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "symbol_name": chunk.symbol_name or "",
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            },
        )
        for chunk in chunks
    ]
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=False,
    )
    stored_index = StoredLlamaIndex(
        repo_name=repo_name,
        repo_path=resolved_path,
        chunks=chunks,
        index=index,
    )
    _LLAMA_INDEX_REGISTRY[resolved_path] = stored_index
    return stored_index


def search_llamaindex_index(
    stored_index: StoredLlamaIndex,
    query: str,
    top_k: int,
) -> list[tuple[ChunkRecord, float]]:
    retriever = stored_index.index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results: list[tuple[ChunkRecord, float]] = []

    for node_with_score in nodes:
        node = node_with_score.node
        metadata = node.metadata or {}
        results.append(
            (
                ChunkRecord(
                    file_path=str(metadata.get("file_path", "")),
                    language=str(metadata.get("language", "text")),
                    chunk_type=str(metadata.get("chunk_type", "block")),
                    symbol_name=_normalize_symbol_name(metadata.get("symbol_name")),
                    start_line=int(metadata.get("start_line", 1)),
                    end_line=int(metadata.get("end_line", 1)),
                    content=getattr(node, "text", "") or "",
                ),
                float(node_with_score.score or 0.0),
            )
        )

    return results


def _normalize_symbol_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ensure_llamaindex_available() -> None:
    if Document is None or VectorStoreIndex is None or HuggingFaceEmbedding is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LlamaIndex dependencies are not installed. "
                "Install llama-index-core and llama-index-embeddings-huggingface."
            ),
        )
