"""Run evaluation experiments against the assistant."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.logger import get_logger
from app.db.models import (
    AnswerFeedbackRequest,
    AnswerFeedbackResponse,
    EvalCase,
    EvalCaseResult,
    EvalResultsResponse,
    EvalRunResponse,
)
from app.eval.dataset import get_default_eval_cases
from app.eval.metrics import compute_hit_at_k, summarize_hit_rate
from app.retrieval.retriever import semantic_search

logger = get_logger(__name__)
EVAL_RESULTS_PATH = Path("data/eval/eval_runs.jsonl")
FEEDBACK_RESULTS_PATH = Path("data/eval/answer_feedback.jsonl")


def run_retrieval_eval(
    repo_path: str,
    top_k: int,
    cases: list[EvalCase] | None = None,
) -> EvalRunResponse:
    started_at = time.perf_counter()
    resolved_repo_path = str(Path(repo_path).expanduser().resolve())
    eval_cases = cases or get_default_eval_cases()
    results: list[EvalCaseResult] = []

    for case in eval_cases:
        case_started_at = time.perf_counter()
        search_response = semantic_search(
            repo_path=resolved_repo_path,
            query=case.query,
            top_k=top_k,
            language=case.language,
            chunk_types=case.chunk_types,
            file_path_contains=case.file_path_contains,
        )
        retrieved_file_paths = [result.chunk.file_path for result in search_response.results]
        hit_at_k = compute_hit_at_k(case.expected_file_paths, retrieved_file_paths)
        results.append(
            EvalCaseResult(
                name=case.name,
                query=case.query,
                expected_file_paths=case.expected_file_paths,
                retrieved_file_paths=retrieved_file_paths,
                hit=hit_at_k > 0,
                hit_at_k=hit_at_k,
                retrieval_latency_ms=search_response.latency_ms,
                latency_ms=_elapsed_ms(case_started_at),
            )
        )

    hits, hit_rate = summarize_hit_rate(results)
    repo_name = Path(resolved_repo_path).name
    total_latency_ms = _elapsed_ms(started_at)
    logger.info(
        "eval_run repo=%s top_k=%s total_cases=%s hits=%s hit_rate=%.3f latency_ms=%.2f",
        resolved_repo_path,
        top_k,
        len(results),
        hits,
        hit_rate,
        total_latency_ms,
    )

    response = EvalRunResponse(
        run_id=f"eval-{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        repo_name=repo_name,
        repo_path=resolved_repo_path,
        top_k=top_k,
        latency_ms=total_latency_ms,
        total_cases=len(results),
        hits=hits,
        hit_rate=hit_rate,
        results=results,
    )
    save_eval_run(response)
    return response


def save_eval_run(result: EvalRunResponse) -> None:
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json())
        handle.write("\n")


def load_eval_runs(limit: int = 20) -> EvalResultsResponse:
    if not EVAL_RESULTS_PATH.exists():
        return EvalResultsResponse(total_runs=0, results=[])

    lines = EVAL_RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    parsed = [EvalRunResponse.model_validate_json(line) for line in lines if line.strip()]
    parsed.sort(
        key=lambda item: item.created_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return EvalResultsResponse(
        total_runs=len(parsed),
        results=parsed[:limit],
    )


def save_answer_feedback(payload: AnswerFeedbackRequest) -> AnswerFeedbackResponse:
    FEEDBACK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = AnswerFeedbackResponse(
        feedback_id=f"fb-{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        status="saved",
    )
    record = {
        "feedback_id": response.feedback_id,
        "created_at": response.created_at.isoformat(),
        "repo_path": payload.repo_path,
        "question": payload.question,
        "answer_mode": payload.answer_mode,
        "rating": payload.rating,
        "comments": payload.comments,
    }
    with FEEDBACK_RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")
    logger.info(
        "answer_feedback repo=%s answer_mode=%s rating=%s",
        payload.repo_path,
        payload.answer_mode,
        payload.rating,
    )
    return response


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
