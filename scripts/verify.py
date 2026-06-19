"""
Part 1 verification: confirm Ollama is running and gemma3:1b responds.
Run: python scripts/verify.py
"""

import sys
import requests
import ollama

OLLAMA_HOST = "http://localhost:11434"
MODEL = "gemma3:1b"


def check_server() -> bool:
    try:
        r = requests.get(OLLAMA_HOST, timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def main():
    print("=" * 60)
    print("Part 1 — Ollama Verification")
    print("=" * 60)

    # 1. Health check
    print(f"\n[1] Checking Ollama server at {OLLAMA_HOST} ...", end=" ")
    if not check_server():
        print("FAIL")
        print(
            "\nERROR: Ollama server is not running.\n"
            "Run: .\\scripts\\deploy.ps1   (or: ollama serve)"
        )
        sys.exit(1)
    print("OK")

    # 2. List available models
    print(f"[2] Checking model '{MODEL}' is available ...", end=" ")
    models = ollama.list()
    available = [m.model for m in models.models]
    if not any(MODEL in name for name in available):
        print("FAIL")
        print(
            f"\nERROR: Model '{MODEL}' not found.\n"
            f"Run: ollama pull {MODEL}"
        )
        sys.exit(1)
    print("OK")

    # 3. Hello World round-trip
    print(f"[3] Sending 'Hello World' prompt to {MODEL} ...")
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Please respond with exactly two words: Hello World",
            }
        ],
    )
    reply = response.message.content.strip()

    print(f"\n    Model response: {reply!r}")
    print(f"    Prompt tokens : {response.prompt_eval_count}")
    print(f"    Output tokens : {response.eval_count}")

    print("\n" + "=" * 60)
    print("PASS — Ollama is running and gemma3:1b is responding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
