"""
Manual integration test for the FastAPI SSE /chat endpoint.
Run with the server already started:
    .\.venv\Scripts\python.exe scripts\test_api.py
"""
import json
import sys

import requests

BASE_URL = "http://localhost:8000"


def check_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"[health] {r.json()}\n")


def stream_chat(message: str):
    print(f"[chat] Q: {message}")
    print("[chat] Streaming answer: ", end="", flush=True)

    with requests.post(
        f"{BASE_URL}/chat",
        json={"message": message},
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        done_event = None

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = json.loads(raw_line[len("data:"):].strip())

            if payload["type"] == "token":
                print(payload["content"], end="", flush=True)
            elif payload["type"] == "done":
                done_event = payload
            elif payload["type"] == "error":
                print(f"\n[ERROR] {payload['content']}", file=sys.stderr)
                return

    print()  # newline after streamed tokens
    if done_event:
        print(f"[chat] used_rag: {done_event['used_rag']}")
        chunks = done_event["retrieved_chunks"]
        if chunks:
            print(f"[chat] retrieved {len(chunks)} chunk(s):")
            for c in chunks:
                print(f"       - {c['chunk_id']} (score={c['score']:.4f}): {c['text'][:80]}...")
    print()


def main():
    check_health()
    stream_chat("What is cosine similarity?")   # expects used_rag=True
    stream_chat("What is 2 + 2?")              # expects used_rag=False


if __name__ == "__main__":
    main()
