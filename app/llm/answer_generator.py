"""Generate grounded answers from retrieved context."""

import logging
import time
from urllib.parse import urlencode

import httpx
from app.core.config import settings
from app.db.models import AskResponse, CitationRecord, RetrievalSettings, SearchResult
from app.llm.prompts import build_grounded_answer_prompt

try:
    from google import genai
except ImportError:  # pragma: no cover - depends on local environment
    genai = None

logger = logging.getLogger(__name__)


def generate_grounded_answer(
    repo_name: str,
    repo_path: str,
    question: str,
    results: list[SearchResult],
    retrieval_settings: RetrievalSettings,
    retrieval_latency_ms: float,
) -> AskResponse:
    started_at = time.perf_counter()
    citations = [
        CitationRecord(
            file_path=result.chunk.file_path,
            start_line=result.chunk.start_line,
            end_line=result.chunk.end_line,
            chunk_type=result.chunk.chunk_type,
            symbol_name=result.chunk.symbol_name,
        )
        for result in results
    ]

    if not results:
        return AskResponse(
            repo_name=repo_name,
            repo_path=repo_path,
            question=question,
            answer="I could not find relevant repository context for this question.",
            grounded=False,
            answer_mode="no-context",
            retrieval_settings=retrieval_settings,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=_elapsed_ms(started_at),
            latency_ms=round(retrieval_latency_ms + _elapsed_ms(started_at), 2),
            citations=[],
        )

    if _should_use_vertex():
        try:
            answer = _generate_with_vertex(question, results)
            return AskResponse(
                repo_name=repo_name,
                repo_path=repo_path,
                question=question,
                answer=answer,
                grounded=True,
                answer_mode="llm",
                retrieval_settings=retrieval_settings,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=_elapsed_ms(started_at),
                latency_ms=round(retrieval_latency_ms + _elapsed_ms(started_at), 2),
                citations=citations,
            )
        except Exception as exc:  # pragma: no cover - external API behavior
            logger.warning("Vertex generation failed, falling back to local summary: %s", exc)

    if _should_use_gemini():
        try:
            answer = _generate_with_gemini(question, results)
            return AskResponse(
                repo_name=repo_name,
                repo_path=repo_path,
                question=question,
                answer=answer,
                grounded=True,
                answer_mode="llm",
                retrieval_settings=retrieval_settings,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=_elapsed_ms(started_at),
                latency_ms=round(retrieval_latency_ms + _elapsed_ms(started_at), 2),
                citations=citations,
            )
        except Exception as exc:  # pragma: no cover - external API behavior
            logger.warning("Gemini generation failed, falling back to local summary: %s", exc)

    return AskResponse(
        repo_name=repo_name,
        repo_path=repo_path,
        question=question,
        answer=_generate_fallback_answer(question, results),
        grounded=True,
        answer_mode="fallback",
        retrieval_settings=retrieval_settings,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=_elapsed_ms(started_at),
        latency_ms=round(retrieval_latency_ms + _elapsed_ms(started_at), 2),
        citations=citations,
    )


def _should_use_vertex() -> bool:
    return (
        settings.llm_provider.lower() == "vertex"
        and bool(settings.vertex_api_key)
    )


def _should_use_gemini() -> bool:
    return (
        settings.llm_provider.lower() == "gemini"
        and bool(settings.gemini_api_key)
        and genai is not None
    )


def _generate_with_gemini(question: str, results: list[SearchResult]) -> str:
    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = build_grounded_answer_prompt(question, results)
    response = client.models.generate_content(
        model=settings.llm_model_name,
        contents=prompt,
    )
    response_text = (getattr(response, "text", None) or "").strip()
    return response_text or _generate_fallback_answer(question, results)


def _generate_with_vertex(question: str, results: list[SearchResult]) -> str:
    prompt = build_grounded_answer_prompt(question, results)
    endpoint = (
        f"{settings.vertex_base_url}/publishers/google/models/"
        f"{settings.llm_model_name}:generateContent"
    )
    params = urlencode({"key": settings.vertex_api_key})
    url = f"{endpoint}?{params}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return _generate_fallback_answer(question, results)

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    return text or _generate_fallback_answer(question, results)


def _generate_fallback_answer(question: str, results: list[SearchResult]) -> str:
    top_results = results[:3]
    lines = [f"Question: {question}", "", "Based on retrieved context:"]

    for index, result in enumerate(top_results, start=1):
        chunk = result.chunk
        symbol_text = f" ({chunk.symbol_name})" if chunk.symbol_name else ""
        snippet = _summarize_chunk(chunk.content)
        lines.append(
            f"{index}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"[{chunk.chunk_type}{symbol_text}] -> {snippet}"
        )

    lines.append("")
    lines.append(
        "This is a fallback grounded answer summarizing retrieved chunks. "
        "Configure an LLM provider to generate a more natural synthesized response."
    )
    return "\n".join(lines)


def _summarize_chunk(content: str, max_length: int = 220) -> str:
    compact = " ".join(content.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
