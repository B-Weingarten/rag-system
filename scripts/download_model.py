"""
One-time model download script for environments with SSL certificate issues (e.g. NetFree).
Run once before first use:  python scripts/download_model.py

After this completes, the model is cached locally and the app runs fully offline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch httpx.Client to skip SSL verification before any HF imports.
# Required on networks (e.g. NetFree) whose intermediate CA certificate lacks
# the key usage extension that Python's ssl module now enforces.
import httpx

_orig_client = httpx.Client


class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs["verify"] = False
        super().__init__(*args, **kwargs)


httpx.Client = _NoVerifyClient

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Keep offline flag unset so the download actually hits the network.
os.environ.pop("HF_HUB_OFFLINE", None)

from huggingface_hub import snapshot_download
from src.config import EMBEDDING_MODEL

print(f"Downloading {EMBEDDING_MODEL} ...")
path = snapshot_download(EMBEDDING_MODEL)
print(f"Cached at: {path}")
print("Done. The embedding model is now available for offline use.")
