"""
corpus/loader.py  —  Online index loader.

Loads the pre-built FAISS index and metadata into memory.
Called once at API startup; the loaded index is shared across requests.

Usage:
    from corpus.loader import CorpusIndex
    corpus = CorpusIndex()          # loads from settings paths
    results = corpus.search(vector, top_k=5)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieved evidence chunk."""
    text: str
    title: str
    url: str
    chunk_id: int
    similarity: float       # cosine similarity [−1, 1], higher = more relevant


class CorpusIndex:
    """
    Wraps FAISS index + metadata for retrieval.

    Thread-safe for concurrent reads (FAISS search is read-only).
    Re-instantiating this class is safe; it just reloads from disk.
    """

    def __init__(
        self,
        index_path: Path = settings.faiss_index_path,
        metadata_path: Path = settings.faiss_metadata_path,
    ):
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []
        self._loaded = False

    def load(self) -> None:
        """
        Load FAISS index and metadata from disk.
        Call once at application startup.

        Raises:
            FileNotFoundError: If the index hasn't been built yet.
            RuntimeError: If metadata count doesn't match index size.
        """
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._index_path}. "
                "Run: python -m corpus.build_index"
            )
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {self._metadata_path}. "
                "Run: python -m corpus.build_index"
            )

        logger.info("Loading FAISS index from %s…", self._index_path)
        self._index = faiss.read_index(str(self._index_path))

        logger.info("Loading metadata from %s…", self._metadata_path)
        with open(self._metadata_path, encoding="utf-8") as f:
            self._metadata = json.load(f)

        if self._index.ntotal != len(self._metadata):
            raise RuntimeError(
                f"Index/metadata mismatch: FAISS has {self._index.ntotal} vectors "
                f"but metadata has {len(self._metadata)} entries. "
                "Re-run: python -m corpus.build_index"
            )

        logger.info(
            "Corpus index loaded. Vectors: %d, Dimension: %d",
            self._index.ntotal,
            self._index.d,
        )
        self._loaded = True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = settings.retrieval_top_k,
        min_similarity: float = settings.retrieval_min_similarity,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar chunks to a query vector.

        Args:
            query_vector: 1-D float32 embedding (must be L2-normalised).
            top_k: Maximum number of results to return.
            min_similarity: Chunks below this similarity are filtered out.

        Returns:
            List of SearchResult, sorted by descending similarity.
            May be shorter than top_k if few chunks meet min_similarity.
        """
        if not self._loaded:
            raise RuntimeError("CorpusIndex.load() has not been called.")

        # FAISS expects shape (1, dim)
        vec = query_vector.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        scores, indices = self._index.search(vec, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS pads with -1 when fewer results exist
                continue
            if float(score) < min_similarity:
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                text=meta["text"],
                title=meta["title"],
                url=meta["url"],
                chunk_id=int(idx),
                similarity=float(score),
            ))

        return results

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return self._index.ntotal if self._loaded else 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded
