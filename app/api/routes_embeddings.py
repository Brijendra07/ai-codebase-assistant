from fastapi import APIRouter

from app.db.models import RepositoryEmbeddingResponse, RepositoryIndexRequest
from app.retrieval.retriever import index_repository_embeddings


router = APIRouter(prefix="/repos", tags=["repos"])


@router.post("/embed", response_model=RepositoryEmbeddingResponse)
async def embed_repository(payload: RepositoryIndexRequest) -> RepositoryEmbeddingResponse:
    return index_repository_embeddings(payload.repo_path)
