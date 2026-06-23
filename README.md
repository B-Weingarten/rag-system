# The Agentic Edge Stack

ML Systems Engineer technical assessment. A LangGraph RAG agent with structured routing and SSE streaming, running entirely on CPU via Ollama (`gemma3:1b`).

## Architecture

```
User query
    │
    ▼
POST /chat ── FastAPI + SSE
    │
    ▼
router_node  (gemma3:1b + format="json" → RouteDecision)
    │
    ├─ "rag" ──▶  VectorBackend (ChromaDB | Qdrant)
    │             all-MiniLM-L6-v2 · 384-dim · 18 chunks
    │
    └─ "direct" ─┐
                 └──▶ gemma3:1b (Ollama) ──▶ SSE token stream
```

## Quick Start

```powershell
# 1. Create venv and install dependencies
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2A. Start Ollama (native) — unblock first if downloaded as zip:
#     Unblock-File .\scripts\deploy.ps1
.\scripts\deploy.ps1

# 2B. Start Ollama (Docker Compose) — alternative to 2A, custom CA certs auto-trusted:
# docker compose up -d
# docker exec ollama ollama pull gemma3:1b

# 3. Verify Ollama is live
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/verify.py

# 4. Run the core test scripts (Ollama must be running)
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe tests/test_rag.py
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe tests/test_agent.py

# 5. Start the API server
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

## Parts & Bonuses

| | Description | Deliverables | Run |
|--|-------------|--------------|-----|
| **Part 1** | Ollama install + `gemma3:1b` pull — native via `scripts/deploy.ps1` or Docker Compose | `scripts/deploy.ps1`<br>`scripts/verify.py`<br>`docker-compose.yml` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/verify.py` |
| **Part 2** | RAG — `all-MiniLM-L6-v2` embeddings, ChromaDB in-memory, 18 chunks | dataset: `data/model_math_rag_dataset.md`<br>code: `src/rag/`<br>test: `tests/test_rag.py` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe tests/test_rag.py` |
| **Part 3** | LangGraph agent — `router_node` → `rag_node` / `direct_node`; sync + async API | code: `src/agent/`<br>test: `tests/test_agent.py` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe tests/test_agent.py` |
| **Part 4** | FastAPI `POST /chat` SSE streaming + `GET /health` | `src/api/main.py`<br>`src/api/schemas.py` | start server first, then: `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe tests/test_api.py` |
| **Bonus 1** | Structured output embedded inside the agent flow: the router LLM returns a minimal validated JSON (`RouteDecision`) that directly controls the LangGraph routing decision. Schema kept intentionally small for reliability with a lightweight local model. | `src/api/schemas.py` (`RouteDecision`)<br>`tests/test_router_schema.py` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe tests/test_router_schema.py` |
| **Bonus 2** | Quantization benchmark — Q4 vs Q8, TPS + RAM → `reports/quantization_report.md` | `reports/quantization_report.md` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/benchmark_quantization.py` |
| **Bonus 3** | `VectorBackend` Protocol + Qdrant backend + `k8s/qdrant.yaml` | `src/rag/backends/base.py`<br>`src/rag/backends/chroma.py`<br>`src/rag/backends/qdrant.py`<br>`k8s/qdrant.yaml` | `kubectl apply -f k8s/qdrant.yaml`<br>`$env:VECTOR_BACKEND="qdrant"; $env:QDRANT_URL="http://localhost:6333"`<br>`$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload` |

## API Reference

**`GET /health`**
```
curl http://localhost:8000/health
```
Response: `{"status": "ok", "model": "gemma3:1b"}`

**`POST /chat`** — Server-Sent Events stream
```
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is cosine similarity?\"}"
```

SSE event payloads:
```json
{"type": "token",  "content": "Cosine similarity..."}
{"type": "done",   "used_rag": true, "route": "rag", "retrieved_chunks": [...]}
{"type": "error",  "content": "..."}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_BACKEND` | `chroma` | `chroma` (in-memory) or `qdrant` (persistent, K8s) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant service URL — used when `VECTOR_BACKEND=qdrant` |

Switch to the Qdrant backend:
```powershell
kubectl apply -f k8s/qdrant.yaml
$env:VECTOR_BACKEND="qdrant"; $env:QDRANT_URL="http://localhost:6333"
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```
