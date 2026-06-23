from __future__ import annotations

import chromadb
from src.config import CHROMA_COLLECTION
from src.rag.chunker import load_chunks
from src.rag.embedder import embed


class ChromaBackend:
    def __init__(self) -> None:
        self._collection: chromadb.Collection | None = None

    def build_index(self) -> None:
        chunks = load_chunks()
        texts = [c["text"] for c in chunks]
        embeddings = embed(texts)

        client = chromadb.EphemeralClient()
        self._collection = client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c["source"]} for c in chunks],
        )

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self.build_index()
        return self._collection

    def retrieve(self, query: str, k: int) -> list[dict]:
        collection = self._get_collection()
        results = collection.query(
            query_embeddings=embed([query]),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": round(1.0 - results["distances"][0][i], 4),
            }
            for i in range(len(results["ids"][0]))
        ]
