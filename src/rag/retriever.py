from src.config import TOP_K
from src.rag.indexer import get_backend


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    return get_backend().retrieve(query, k)
