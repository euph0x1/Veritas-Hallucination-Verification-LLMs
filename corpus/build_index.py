"""
corpus/build_index.py  —  Offline indexing pipeline.

Run once before starting the API:
    python -m corpus.build_index

What it does:
    1. Loads Wikipedia articles from HuggingFace datasets
    2. Splits each article into overlapping text chunks
    3. Embeds every chunk with the configured embedding model
    4. Builds a FAISS index and saves it to data/index/
    5. Saves chunk metadata (text, title, url) alongside the index

Design decisions:
    - Chunking is token-aware (splits on word boundaries, not character count)
    - Overlap ensures claims near chunk boundaries still have supporting evidence
    - Metadata is stored separately from FAISS so we can retrieve text by index ID
    - The module is stateless: re-running it overwrites the existing index
"""

import json
import logging
import sys
from pathlib import Path
from typing import Iterator

import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.providers import SentenceTransformerProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = settings.corpus_chunk_size,
    overlap: int = settings.corpus_chunk_overlap,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Word-level splitting is fast and keeps chunks readable.
    Token-level splitting (using a tokenizer) is more precise but
    adds a dependency — deferred to V2.

    Args:
        text: Full article text.
        chunk_size: Target chunk size in words.
        overlap: Number of words to repeat between consecutive chunks.

    Returns:
        List of chunk strings. Empty list if text is too short.
    """
    words = text.split()
    if len(words) < 20:          # skip stubs and redirects
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 50:      # skip degenerate short tails
            chunks.append(chunk)
    return chunks


# ── Wikipedia loader ──────────────────────────────────────────────────────────

def load_wikipedia_articles(max_articles: int) -> Iterator[dict]:
    """
    Stream Wikipedia articles from HuggingFace.

    Uses the 20220301.en snapshot (stable, well-known).
    Streams to avoid loading the full dataset into memory.

    Yields dicts with keys: title, text, url
    """
    logger.info("Loading Wikipedia dataset (streaming, max=%d articles)…", max_articles)
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for article in dataset:
        if count >= max_articles:
            break
        text = article.get("text", "").strip()
        if not text or len(text) < 200:   # skip stubs
            continue
        yield {
            "title": article.get("title", ""),
            "text": text,
            "url": article.get("url", ""),
        }
        count += 1

    logger.info("Loaded %d articles.", count)


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(
    max_articles: int = settings.corpus_max_articles,
    index_path: Path = settings.faiss_index_path,
    metadata_path: Path = settings.faiss_metadata_path,
) -> None:
    """
    Full offline pipeline: load → chunk → embed → index → save.

    Args:
        max_articles: Number of Wikipedia articles to process.
        index_path: Where to save the FAISS index file.
        metadata_path: Where to save chunk metadata JSON.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect all chunks ────────────────────────────────────────────
    logger.info("Step 1/4 — Chunking articles…")
    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for article in load_wikipedia_articles(max_articles):
        chunks = chunk_text(article["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "title": article["title"],
                "url": article["url"],
                "text": chunk,
                "chunk_id": len(all_chunks) - 1,
            })

    logger.info("Total chunks: %d", len(all_chunks))
    if not all_chunks:
        logger.error("No chunks produced. Check corpus settings.")
        sys.exit(1)

    # ── Step 2: Embed in batches ──────────────────────────────────────────────
    logger.info("Step 2/4 — Embedding chunks (model: %s)…", settings.embedding_model)
    provider = SentenceTransformerProvider()
    # Load model before batch loop
    provider._load()

    batch_size = 256
    all_vectors: list[np.ndarray] = []

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch = all_chunks[i: i + batch_size]
        vecs = provider._encode(batch)    # sync encode; we're in a script, not async
        all_vectors.append(vecs)

    embeddings = np.vstack(all_vectors).astype(np.float32)
    logger.info("Embedding matrix: %s  dtype=%s", embeddings.shape, embeddings.dtype)

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    logger.info("Step 3/4 — Building FAISS IndexFlatIP…")
    # IndexFlatIP = exact inner-product search.
    # Correct because embeddings are L2-normalised → IP == cosine similarity.
    # V2: swap for IndexHNSWFlat for approximate search at large scale.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index contains %d vectors.", index.ntotal)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    logger.info("Step 4/4 — Saving index and metadata…")
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved → %s", index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    logger.info("Metadata saved → %s  (%d chunks)", metadata_path, len(all_metadata))

    logger.info("✓ Index build complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Veritas FAISS index from Wikipedia")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=settings.corpus_max_articles,
        help=f"Number of Wikipedia articles to index (default: {settings.corpus_max_articles})",
    )
    args = parser.parse_args()

    build_index(max_articles=args.max_articles)
