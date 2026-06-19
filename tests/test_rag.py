import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import retrieve

QUERIES = [
    "What is cosine similarity and how is it calculated?",
    "How does backpropagation compute gradients?",
    "What are the steps of a RAG indexing and query pipeline?",
]


def print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    for rank, chunk in enumerate(results, start=1):
        lines = chunk["text"].split("\n")
        heading = lines[0]
        excerpt = " ".join(line for line in lines[1:] if line.strip())[:140]
        print(
            f"  [{rank}] chunk_id={chunk['chunk_id']} | score={chunk['score']:.4f}"
            f" | source={chunk['source']}"
        )
        print(f"      {heading}")
        print(f"      {excerpt}...")
    print()


if __name__ == "__main__":
    print("Building index (loading embedding model + indexing 18 chunks)...")
    for query in QUERIES:
        results = retrieve(query)
        print_results(query, results)
    print("Done.")
