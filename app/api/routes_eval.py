from fastapi import APIRouter, Query

from app.db.models import (
    AnswerFeedbackRequest,
    AnswerFeedbackResponse,
    EvalResultsResponse,
    EvalRunRequest,
    EvalRunResponse,
)
from app.eval.runner import load_eval_runs, run_retrieval_eval, save_answer_feedback


router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(payload: EvalRunRequest) -> EvalRunResponse:
    return run_retrieval_eval(
        repo_path=payload.repo_path,
        top_k=payload.top_k,
        cases=payload.cases,
    )


@router.get("/results", response_model=EvalResultsResponse)
async def get_eval_results(limit: int = Query(20, ge=1, le=100)) -> EvalResultsResponse:
    return load_eval_runs(limit=limit)


@router.post("/feedback", response_model=AnswerFeedbackResponse)
async def save_feedback(payload: AnswerFeedbackRequest) -> AnswerFeedbackResponse:
    return save_answer_feedback(payload)
