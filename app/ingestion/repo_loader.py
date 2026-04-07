"""Load repositories from local paths, archives, or remote sources."""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status

from app.db.models import FileRecord, RepositoryIndexResponse
from app.ingestion.file_filter import should_index_file, should_skip_directory


def list_repository_files(root: Path) -> tuple[list[FileRecord], int]:
    files: list[FileRecord] = []
    total_files_scanned = 0

    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not should_skip_directory(Path(current_root) / dirname)
        ]

        for filename in filenames:
            total_files_scanned += 1
            file_path = Path(current_root) / filename

            if not should_index_file(file_path):
                continue

            relative_path = file_path.relative_to(root).as_posix()
            files.append(
                FileRecord(
                    path=relative_path,
                    extension=file_path.suffix.lower(),
                    size_bytes=file_path.stat().st_size,
                )
            )

    return sorted(files, key=lambda item: item.path), total_files_scanned


def load_local_repository(repo_path: str) -> RepositoryIndexResponse:
    root = Path(repo_path).expanduser().resolve()

    if not root.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository path not found: {root}",
        )

    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Repository path is not a directory: {root}",
        )

    files, total_files_scanned = list_repository_files(root)

    return RepositoryIndexResponse(
        repo_id=f"repo-{uuid4().hex[:12]}",
        repo_name=root.name,
        repo_path=str(root),
        indexed_at=datetime.now(timezone.utc),
        total_files_scanned=total_files_scanned,
        total_files_indexed=len(files),
        skipped_files=total_files_scanned - len(files),
        files=files,
    )
