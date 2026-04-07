from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.chunking.code_chunker import chunk_file
from app.db.models import RepositoryChunkResponse, RepositoryIndexRequest
from app.ingestion.parser import parse_file
from app.ingestion.repo_loader import list_repository_files


router = APIRouter(prefix="/repos", tags=["repos"])


@router.post("/chunks", response_model=RepositoryChunkResponse)
async def chunk_repository(payload: RepositoryIndexRequest) -> RepositoryChunkResponse:
    root = Path(payload.repo_path).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid repository path: {root}",
        )

    file_records, _ = list_repository_files(root)

    chunks = []
    for file_record in file_records:
        file_path = root / file_record.path
        parsed_file = parse_file(root, file_path)
        chunks.extend(chunk_file(parsed_file))

    return RepositoryChunkResponse(
        repo_name=root.name,
        repo_path=str(root),
        total_files_parsed=len(file_records),
        total_chunks=len(chunks),
        chunks=chunks,
    )
