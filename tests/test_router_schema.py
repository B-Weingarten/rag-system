"""
Tests that the router LLM adheres to the RouteDecision JSON schema.

Calls the LLM directly (same as router_node) and prints the raw JSON
the model returns, so you can see schema adherence in the output.

Run:
    .\.venv\Scripts\python.exe tests\test_router_schema.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.agent.graph import _ROUTER_PROMPT
from src.api.schemas import RouteDecision
from src.config import OLLAMA_HOST, OLLAMA_MODEL

CASES = [
    # Clearly in the knowledge base → rag
    {"query": "What is cosine similarity?",        "expected": "rag"},
    {"query": "How does RAG retrieval work?",      "expected": "rag"},
    # Clearly general knowledge → direct
    {"query": "What is 2 + 2?",                    "expected": "direct"},
    {"query": "What is the capital of France?",    "expected": "direct"},
    {"query": "Tell me a joke.",                   "expected": "direct"},
]


def run_router(query: str) -> tuple[str, str, bool]:
    """
    Returns (raw_json, parsed_route, schema_valid).
    Mirrors exactly what router_node does.
    """
    prompt = _ROUTER_PROMPT.format(query=query)
    llm = ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL, format="json")
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        decision = RouteDecision(**json.loads(raw))
        return raw, decision.route, True
    except Exception:
        return raw, "direct", False


def main():
    print(f"Model : {OLLAMA_MODEL}  |  Schema : RouteDecision\n")
    print("=" * 62)

    passed = 0
    for case in CASES:
        query, expected = case["query"], case["expected"]
        raw_json, route, schema_valid = run_router(query)

        status = "PASS" if route == expected else "FAIL"
        if status == "PASS":
            passed += 1

        print(f'Query           : "{query}"')
        print(f"Raw JSON        : {raw_json}")
        print(f"Schema valid    : {'YES' if schema_valid else 'NO  <-- parse error'}")
        print(f"Parsed route    : {route}")
        print(f"Expected        : {expected}  -->  {status}")
        print("-" * 62)

    total = len(CASES)
    print(f"\nResult: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
