"""Metrics for retrieval and answer quality."""

from app.db.models import EvalCaseResult


def compute_hit_at_k(expected_file_paths: list[str], retrieved_file_paths: list[str]) -> float:
    expected = {path.lower() for path in expected_file_paths}
    retrieved = [path.lower() for path in retrieved_file_paths]

    return 1.0 if any(path in expected for path in retrieved) else 0.0


def summarize_hit_rate(results: list[EvalCaseResult]) -> tuple[int, float]:
    hits = sum(1 for result in results if result.hit)
    total = len(results)
    hit_rate = hits / total if total else 0.0
    return hits, hit_rate
