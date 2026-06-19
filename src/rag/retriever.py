from src.config import TOP_K
from src.rag.embedder import embed
from src.rag.indexer import get_collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    query_embedding = embed([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": round(1.0 - results["distances"][0][i], 4),
            }
        )
    return chunks
