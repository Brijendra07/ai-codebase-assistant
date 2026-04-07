from fastapi import APIRouter

from app.db.models import RepositoryIndexRequest, RepositoryIndexResponse
from app.ingestion.repo_loader import load_local_repository


router = APIRouter(prefix="/repos", tags=["repos"])


@router.post("/index", response_model=RepositoryIndexResponse)
async def index_repository(payload: RepositoryIndexRequest) -> RepositoryIndexResponse:
    return load_local_repository(payload.repo_path)
