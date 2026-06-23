from __future__ import annotations

from src.config import VECTOR_BACKEND
from src.rag.backends.base import VectorBackend

_backend: VectorBackend | None = None


def get_backend() -> VectorBackend:
    global _backend
    if _backend is None:
        if VECTOR_BACKEND == "qdrant":
            from src.rag.backends.qdrant import QdrantBackend
            _backend = QdrantBackend()
        else:
            from src.rag.backends.chroma import ChromaBackend
            _backend = ChromaBackend()
    return _backend
