"""Generate embeddings for code chunks."""

from functools import lru_cache

import numpy as np

from app.core.config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - depends on local environment
    SentenceTransformer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
        ) from IMPORT_ERROR

    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")
