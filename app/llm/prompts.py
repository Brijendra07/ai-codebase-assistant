"""Prompt templates for grounded repository Q&A."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.db.models import SearchResult

_GROUNDED_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a repository assistant. Answer only from the provided context. "
            "If the context is insufficient, say so clearly. Prefer concise technical "
            "explanations. Cite source numbers like [Source 1] when making claims.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRetrieved Context:\n{context}",
        ),
    ]
)


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

    prompt_value = _GROUNDED_ANSWER_TEMPLATE.invoke(
        {
            "question": question,
            "context": "\n\n".join(context_blocks),
        }
    )
    return prompt_value.to_string()


def parse_answer_text(text: str) -> str:
    return StrOutputParser().invoke(text).strip()
