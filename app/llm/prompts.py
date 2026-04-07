"""Prompt templates for grounded repository Q&A."""

from app.db.models import SearchResult


def build_grounded_answer_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks: list[str] = []

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"File: {chunk.file_path}",
                    f"Lines: {chunk.start_line}-{chunk.end_line}",
                    f"Chunk Type: {chunk.chunk_type}",
                    f"Symbol: {chunk.symbol_name or 'N/A'}",
                    "Content:",
                    chunk.content,
                ]
            )
        )

    context = "\n\n".join(context_blocks)

    return (
        "You are a repository assistant. Answer only from the provided context.\n"
        "If the context is insufficient, say so clearly.\n"
        "Prefer concise technical explanations.\n"
        "Cite source numbers like [Source 1] when making claims.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{context}"
    )
