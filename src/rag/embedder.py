from __future__ import annotations

import os

# Use locally cached model; avoids SSL handshake failures on filtered networks (NetFree).
# Run scripts/download_model.py once to populate the cache, then this stays offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, normalize_embeddings=True).tolist()
