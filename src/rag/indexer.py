from __future__ import annotations

import chromadb
from src.config import CHROMA_COLLECTION
from src.rag.chunker import load_chunks
from src.rag.embedder import embed

_collection: chromadb.Collection | None = None


def build_index() -> chromadb.Collection:
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = build_index()
    return _collection
