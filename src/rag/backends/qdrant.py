from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM
from src.rag.chunker import load_chunks
from src.rag.embedder import embed


class QdrantBackend:
    def __init__(self) -> None:
        self._client = QdrantClient(url=QDRANT_URL)
        self._ready = False

    def build_index(self) -> None:
        chunks = load_chunks()
        texts = [c["text"] for c in chunks]
        embeddings = embed(texts)

        existing = [c.name for c in self._client.get_collections().collections]
        if QDRANT_COLLECTION in existing:
            self._client.delete_collection(QDRANT_COLLECTION)

        self._client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        self._client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=i + 1,
                    vector=embeddings[i],
                    payload={
                        "chunk_id": chunks[i]["chunk_id"],
                        "text": texts[i],
                        "source": chunks[i]["source"],
                    },
                )
                for i in range(len(chunks))
            ],
        )
        self._ready = True

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.build_index()

    def retrieve(self, query: str, k: int) -> list[dict]:
        self._ensure_ready()
        results = self._client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=embed([query])[0],
            limit=k,
            with_payload=True,
        )
        return [
            {
                "chunk_id": hit.payload["chunk_id"],
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "score": round(hit.score, 4),
            }
            for hit in results
        ]
