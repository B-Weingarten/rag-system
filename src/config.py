from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "data" / "model_math_rag_dataset.md"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "rag_chunks"
TOP_K = 3

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma3:1b"
