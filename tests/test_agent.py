import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import run_agent

CASES = [
    {
        "label": "RAG path",
        "query": "What is cosine similarity?",
        "expect_rag": True,
    },
    {
        "label": "Direct path",
        "query": "What is 2+2?",
        "expect_rag": False,
    },
]


def print_trace(label: str, query: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"Query        : {query}")
    print(f"used_rag     : {result['used_rag']}")
    print(f"Answer       : {result['answer'][:300]}")
    if result["retrieved_chunks"]:
        print(f"Chunks ({len(result['retrieved_chunks'])}):")
        for i, c in enumerate(result["retrieved_chunks"], 1):
            heading = c["text"].split("\n")[0]
            print(f"  [{i}] {c['chunk_id']} | score={c['score']:.4f} | {heading}")
    else:
        print("Chunks       : (none)")
    print()


if __name__ == "__main__":
    print("Initializing index and agent (first run loads embedding model)...")
    all_passed = True
    for case in CASES:
        result = run_agent(case["query"])
        print_trace(case["label"], case["query"], result)
        if result["used_rag"] != case["expect_rag"]:
            print(
                f"  !! ROUTING MISMATCH: expected used_rag={case['expect_rag']}, "
                f"got {result['used_rag']}"
            )
            all_passed = False
        else:
            print(f"  OK: routing matched expected used_rag={case['expect_rag']}")

    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))
