"""
Quick standalone health check for the embedding endpoint used by
build_portal_docs_index.py and build_dataset_vector_index.py.

Sends a single trivial embed_query() call — no batching, no index writes — so it
fails/succeeds in seconds. Useful for confirming the embedding provider is back
online before re-running a full index rebuild.

Usage:
    python scripts/check_embedding_health.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.assistant.llm import get_embeddings_model


def main() -> int:
    print("Probing embedding endpoint...")
    try:
        vec = get_embeddings_model().embed_query("health check")
        print(f"OK — embedding succeeded (dim={len(vec)}). Safe to retry the index rebuild.")
        return 0
    except Exception as e:
        print(f"FAILED — {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
