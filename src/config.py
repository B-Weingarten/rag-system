from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "data" / "model_math_rag_dataset.md"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output size

# Switch between "chroma" (in-memory) and "qdrant" (K8s)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")

CHROMA_COLLECTION = "rag_chunks"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "rag_chunks"

TOP_K = 3

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma3:1b"
