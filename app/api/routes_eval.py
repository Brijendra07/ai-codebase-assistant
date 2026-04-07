from fastapi import APIRouter

from app.db.models import EvalRunRequest, EvalRunResponse
from app.eval.runner import run_retrieval_eval


router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(payload: EvalRunRequest) -> EvalRunResponse:
    return run_retrieval_eval(
        repo_path=payload.repo_path,
        top_k=payload.top_k,
        cases=payload.cases,
    )
