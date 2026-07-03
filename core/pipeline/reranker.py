"""
core/pipeline/reranker.py  —  Cross-encoder reranking of retrieved evidence.

Sits between the FAISS retriever and the NLI verifier.

Pipeline position:
    FAISS retrieval (top-10) → Reranker (top-3) → NLI verifier

Why this helps:
    FAISS bi-encoder retrieval embeds the claim and each chunk independently
    and compares their vectors. This is fast but imprecise — it finds chunks
    that are *topically* similar to the claim, not necessarily chunks that
    *directly address* the specific fact being verified.

    A cross-encoder reads the (claim, chunk) pair together as a single input,
    allowing it to understand the relationship between the two texts. This
    produces much more accurate relevance scores at the cost of more compute.

    Example: For the claim "Bell was born in Edinburgh", the FAISS retriever
    finds several Bell-related chunks. The reranker correctly scores the
    birthplace chunk highest because it reads both texts together and
    understands that "born in Edinburgh" is directly addressed by
    "Bell was born in Edinburgh, Scotland on March 3, 1847."

Model:
    cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking (127M query-passage pairs)
    - Fast (6 transformer layers) and accurate
    - Outputs a relevance score (higher = more relevant)
    - Upgrade path: ms-marco-MiniLM-L-12-v2 for higher accuracy

Fallback:
    If the model fails to load (no internet, disk space, etc.),
    the reranker transparently returns the original FAISS-ordered
    chunks so the pipeline continues without interruption.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from corpus.loader import SearchResult
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single evidence chunk with its reranker score."""
    chunk: SearchResult
    rerank_score: float         # raw cross-encoder logit (higher = more relevant)
    original_rank: int          # position in FAISS results (0 = highest similarity)


@dataclass
class RerankerOutput:
    """Output of one reranking pass for a single claim."""
    claim: str
    reranked: list[SearchResult]        # top-k chunks ordered by reranker score
    rerank_scores: list[float]          # corresponding reranker scores
    latency_ms: float
    fallback_used: bool                 # True if reranker failed and FAISS order was kept
    reranker_model: str


class CrossEncoderReranker:
    """
    Reranks FAISS-retrieved evidence chunks using a cross-encoder model.

    Lazy-loaded: the model downloads and loads on first rerank() call,
    not at import time. This avoids blocking the event loop during startup
    if reranking is disabled.

    Thread safety: the model is loaded once and used read-only. Safe for
    concurrent async calls because inference runs in a thread pool executor.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        top_k: int = settings.rerank_top_k,
    ):
        self._model_name = model_name
        self._top_k = top_k
        self._model = None
        self._available: Optional[bool] = None  # None = not yet attempted
        self._load_error: Optional[str] = None

    def _load(self) -> bool:
        """
        Load the cross-encoder model.
        Returns True if loaded successfully, False if failed.
        Sets self._available so subsequent calls don't retry a failed load.
        """
        if self._available is not None:
            return self._available

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)
            self._available = True
            logger.info("Reranker model loaded: %s", self._model_name)
        except Exception as e:
            self._available = False
            self._load_error = str(e)
            logger.warning(
                "Reranker model failed to load — will use FAISS order as fallback. "
                "Error: %s", e
            )
        return self._available

    async def rerank(
        self,
        claim: str,
        chunks: list[SearchResult],
    ) -> RerankerOutput:
        """
        Rerank a list of evidence chunks for a given claim.

        Args:
            claim: The atomic factual claim being verified.
            chunks: Evidence chunks from FAISS, ordered by cosine similarity.

        Returns:
            RerankerOutput with chunks re-ordered by cross-encoder relevance score.
            If the reranker is unavailable, returns original FAISS order with
            fallback_used=True.
        """
        if not chunks:
            return RerankerOutput(
                claim=claim,
                reranked=[],
                rerank_scores=[],
                latency_ms=0.0,
                fallback_used=False,
                reranker_model=self._model_name,
            )

        t0 = time.monotonic()

        # Try to load model if not yet attempted
        model_ready = self._load()

        if not model_ready:
            # Graceful fallback: return original FAISS order, capped at top_k
            logger.debug(
                "Reranker unavailable — using FAISS order for: %.60s", claim
            )
            return RerankerOutput(
                claim=claim,
                reranked=chunks[:self._top_k],
                rerank_scores=[c.similarity for c in chunks[:self._top_k]],
                latency_ms=(time.monotonic() - t0) * 1000,
                fallback_used=True,
                reranker_model=self._model_name,
            )

        # Run cross-encoder scoring in thread pool (blocking inference)
        loop = asyncio.get_event_loop()
        try:
            reranked, scores = await loop.run_in_executor(
                None, self._score_and_sort, claim, chunks
            )
        except Exception as e:
            logger.warning(
                "Reranker inference failed for claim '%.60s' — "
                "falling back to FAISS order. Error: %s", claim, e
            )
            return RerankerOutput(
                claim=claim,
                reranked=chunks[:self._top_k],
                rerank_scores=[c.similarity for c in chunks[:self._top_k]],
                latency_ms=(time.monotonic() - t0) * 1000,
                fallback_used=True,
                reranker_model=self._model_name,
            )

        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "Reranked %d → %d chunks in %.0fms for: %.60s",
            len(chunks), len(reranked), latency_ms, claim,
        )

        return RerankerOutput(
            claim=claim,
            reranked=reranked,
            rerank_scores=scores,
            latency_ms=latency_ms,
            fallback_used=False,
            reranker_model=self._model_name,
        )

    def _score_and_sort(
        self,
        claim: str,
        chunks: list[SearchResult],
    ) -> tuple[list[SearchResult], list[float]]:
        """
        Synchronous cross-encoder scoring — runs in thread pool.

        Builds (claim, chunk_text) pairs, scores them all in one batch,
        sorts by descending score, and returns the top-k.
        """
        pairs = [[claim, chunk.text] for chunk in chunks]
        scores: list[float] = self._model.predict(pairs, show_progress_bar=False).tolist()

        # Sort chunks by reranker score descending
        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        top = ranked[:self._top_k]
        top_chunks = [chunk for _, chunk in top]
        top_scores = [score for score, _ in top]

        return top_chunks, top_scores

    @property
    def is_available(self) -> bool:
        """True if the reranker model loaded successfully."""
        if self._available is None:
            return self._load()
        return self._available

    @property
    def model_name(self) -> str:
        return self._model_name
