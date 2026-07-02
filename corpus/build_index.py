"""
corpus/build_index.py  —  Offline indexing pipeline.

Run once before starting the API:
    python -m corpus.build_index

Key improvement over original:
    - First paragraph of each article is extracted as a dedicated chunk
      and indexed TWICE (duplicate entry) to boost its retrieval weight.
      Wikipedia intros contain the most important facts: birth dates,
      birthplaces, key inventions — exactly what NLI needs to verify claims.
    - Chunk size 100 words so each chunk is focused on one topic.
    - Overlap of 20 words maintains context across chunk boundaries.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.providers import SentenceTransformerProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_intro(text: str, max_words: int = 150) -> str | None:
    """
    Extract the introductory paragraph from a Wikipedia article.

    The intro is everything before the first section heading (== Header ==)
    or the first 150 words, whichever comes first. This is where Wikipedia
    stores the most important facts: birth date, birthplace, key inventions.
    """
    lines = text.splitlines()
    intro_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("=="):
            break
        if stripped:
            intro_lines.append(stripped)
    intro = " ".join(intro_lines).strip()
    if not intro:
        return None
    words = intro.split()
    return " ".join(words[:max_words]) if len(words) > max_words else intro


def chunk_text(
    text: str,
    chunk_size: int = settings.corpus_chunk_size,
    overlap: int = settings.corpus_chunk_overlap,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) < 20:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


def load_wikipedia_articles(max_articles: int) -> Iterator[dict]:
    """Stream Wikipedia articles from HuggingFace (20220301.en snapshot)."""
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
        if not text or len(text) < 200:
            continue
        yield {
            "title": article.get("title", ""),
            "text": text,
            "url": article.get("url", ""),
        }
        count += 1
    logger.info("Loaded %d articles.", count)


def build_index(
    max_articles: int = settings.corpus_max_articles,
    index_path: Path = settings.faiss_index_path,
    metadata_path: Path = settings.faiss_metadata_path,
) -> None:
    """
    Full offline pipeline: load → chunk → embed → index → save.
    Intro paragraphs are added twice to boost their retrieval weight.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/4 — Chunking articles…")
    all_chunks: list[str] = []
    all_metadata: list[dict] = []
    intro_boost_count = 0

    for article in load_wikipedia_articles(max_articles):
        title = article["title"]
        url = article["url"]

        # Extract and boost the intro paragraph
        intro = extract_intro(article["text"])
        if intro and len(intro.split()) >= 20:
            for _ in range(2):
                all_chunks.append(intro)
                all_metadata.append({
                    "title": title,
                    "url": url,
                    "text": intro,
                    "chunk_id": len(all_chunks) - 1,
                    "is_intro": True,
                })
            intro_boost_count += 1

        # Regular chunks for the full article body
        chunks = chunk_text(article["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "title": title,
                "url": url,
                "text": chunk,
                "chunk_id": len(all_chunks) - 1,
                "is_intro": False,
            })

    logger.info(
        "Total chunks: %d  (intro boosts: %d articles)",
        len(all_chunks), intro_boost_count,
    )
    if not all_chunks:
        logger.error("No chunks produced. Check corpus settings.")
        sys.exit(1)

    logger.info("Step 2/4 — Embedding chunks (model: %s)…", settings.embedding_model)
    provider = SentenceTransformerProvider()
    provider._load()

    batch_size = 256
    all_vectors: list[np.ndarray] = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch = all_chunks[i: i + batch_size]
        vecs = provider._encode(batch)
        all_vectors.append(vecs)

    embeddings = np.vstack(all_vectors).astype(np.float32)
    logger.info("Embedding matrix: %s  dtype=%s", embeddings.shape, embeddings.dtype)

    logger.info("Step 3/4 — Building FAISS IndexFlatIP…")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index contains %d vectors.", index.ntotal)

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