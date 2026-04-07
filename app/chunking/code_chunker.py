"""Create code-aware chunks from parsed repository content."""

import ast

from app.chunking.metadata_builder import build_chunk_record
from app.db.models import ChunkRecord, ParsedFile


def chunk_file(parsed_file: ParsedFile) -> list[ChunkRecord]:
    if parsed_file.language == "python":
        python_chunks = _chunk_python_file(parsed_file)
        if python_chunks:
            return python_chunks

    return _chunk_by_lines(parsed_file)


def _chunk_python_file(parsed_file: ParsedFile) -> list[ChunkRecord]:
    try:
        tree = ast.parse(parsed_file.content)
    except SyntaxError:
        return _chunk_by_lines(parsed_file)

    lines = parsed_file.content.splitlines()
    chunks: list[ChunkRecord] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", node.lineno)
            content = "\n".join(lines[start_line - 1 : end_line]).strip()

            if not content:
                continue

            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(
                build_chunk_record(
                    parsed_file=parsed_file,
                    chunk_type=chunk_type,
                    symbol_name=node.name,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                )
            )

    return [chunk for chunk in chunks if _is_useful_chunk(chunk)]


def _chunk_by_lines(parsed_file: ParsedFile, max_lines: int = 40) -> list[ChunkRecord]:
    lines = parsed_file.content.splitlines()
    if not lines:
        return []

    chunks: list[ChunkRecord] = []

    for start_index in range(0, len(lines), max_lines):
        end_index = min(start_index + max_lines, len(lines))
        content = "\n".join(lines[start_index:end_index]).strip()

        if not content:
            continue

        chunks.append(
            build_chunk_record(
                parsed_file=parsed_file,
                chunk_type="block",
                start_line=start_index + 1,
                end_line=end_index,
                content=content,
            )
        )

    return [chunk for chunk in chunks if _is_useful_chunk(chunk)]


def _is_useful_chunk(chunk: ChunkRecord) -> bool:
    content = chunk.content.strip()
    if not content:
        return False

    compact = " ".join(content.split())
    if len(compact) < 30:
        return False

    if chunk.chunk_type == "block" and chunk.end_line - chunk.start_line < 1 and compact.startswith('"""'):
        return False

    return True
