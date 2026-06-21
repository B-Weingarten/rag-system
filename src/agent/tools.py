from src.rag.retriever import retrieve


def search_knowledge_base(query: str) -> list[dict]:
    """Retrieve top-k relevant chunks from the ML/math knowledge base."""
    return retrieve(query)
