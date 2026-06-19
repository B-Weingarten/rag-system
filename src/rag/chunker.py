import re
from pathlib import Path
from src.config import DATASET_PATH


def load_chunks(path: Path = DATASET_PATH) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    # Split on every line that starts a new ## section
    raw_parts = re.split(r"\n(?=## )", text)

    chunks = []
    chunk_num = 0
    for part in raw_parts:
        stripped = part.strip()
        if not stripped.startswith("## "):
            continue
        chunk_num += 1
        # Remove trailing horizontal-rule separator added between sections
        cleaned = re.sub(r"\s*\n---\s*$", "", stripped).strip()
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_num:02d}",
                "text": cleaned,
                "source": path.name,
            }
        )
    return chunks
