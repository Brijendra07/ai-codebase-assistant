"""Evaluation dataset definitions."""

from app.db.models import EvalCase


def get_default_eval_cases() -> list[EvalCase]:
    return [
        EvalCase(
            name="repository_ingestion",
            query="Where is repository ingestion implemented?",
            expected_file_paths=["app/ingestion/repo_loader.py"],
            language="python",
            chunk_types=["function", "class"],
            file_path_contains="ingestion",
        ),
        EvalCase(
            name="file_filtering",
            query="Which code filters unsupported files during indexing?",
            expected_file_paths=["app/ingestion/file_filter.py"],
            language="python",
            chunk_types=["function"],
            file_path_contains="ingestion",
        ),
        EvalCase(
            name="chunking_logic",
            query="Where is code chunking implemented?",
            expected_file_paths=["app/chunking/code_chunker.py"],
            language="python",
            chunk_types=["function"],
            file_path_contains="chunking",
        ),
        EvalCase(
            name="query_routes",
            query="Where is the ask API route defined?",
            expected_file_paths=["app/api/routes_query.py"],
            language="python",
            chunk_types=["function"],
            file_path_contains="routes_query",
        ),
        EvalCase(
            name="embedding_indexing",
            query="Where are repository embeddings created?",
            expected_file_paths=["app/retrieval/retriever.py"],
            language="python",
            chunk_types=["function"],
        ),
    ]
