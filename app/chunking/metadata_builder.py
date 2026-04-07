"""Build metadata records for chunks and files."""

from app.db.models import ChunkRecord, ParsedFile


def build_chunk_record(
    parsed_file: ParsedFile,
    chunk_type: str,
    start_line: int,
    end_line: int,
    content: str,
    symbol_name: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        file_path=parsed_file.path,
        language=parsed_file.language,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        content=content,
    )
