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

# 2. Deploy Ollama and pull gemma3:1b
.\scripts\deploy.ps1

# 3. Verify Ollama is live
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/verify.py

# 4. Run the full test suite
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m pytest tests/ -v

# 5. Start the API server
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

## Parts & Bonuses

| | Description | Run |
|--|-------------|-----|
| **Part 1** | Ollama install + `gemma3:1b` pull via `scripts/deploy.ps1` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/verify.py` |
| **Part 2** | RAG — `all-MiniLM-L6-v2` embeddings, ChromaDB in-memory, 18 chunks | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m pytest tests/test_rag.py -v` |
| **Part 3** | LangGraph agent — `router_node` → `rag_node` / `direct_node`; sync + async API | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m pytest tests/test_agent.py -v` |
| **Part 4** | FastAPI `POST /chat` SSE streaming + `GET /health` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m pytest tests/test_api.py -v` |
| **Bonus 1** | Pydantic `RouteDecision` schema + `format="json"` LLM enforcement | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m pytest tests/test_router_schema.py -v` |
| **Bonus 2** | Quantization benchmark — Q4/Q8/Q2, TPS + RAM → `reports/quantization_report.md` | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/benchmark_quantization.py` |
| **Bonus 3** | `VectorBackend` Protocol + Qdrant backend + `k8s/qdrant.yaml` | `kubectl apply -f k8s/qdrant.yaml` |

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
