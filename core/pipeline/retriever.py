"""
core/pipeline/retriever.py  —  Step 3: Evidence retrieval + reranking.

Pipeline:
    1. Embed claim with BGE bi-encoder
    2. FAISS top-k search (default top-10)
    3. Cross-encoder reranking → top-3  [if ENABLE_RERANKING=true]
    4. Return ClaimEvidence with final ranked chunks for NLI

If reranking is disabled or unavailable, step 3 is skipped and FAISS
results are passed directly to the verifier (original V1 behaviour).
"""

import logging
import time
from dataclasses import dataclass, field

from core.providers.base import BaseEmbeddingProvider
from corpus.loader import CorpusIndex, SearchResult
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ClaimEvidence:
    """A single claim paired with its final evidence chunks."""
    claim: str
    evidence: list[SearchResult]        # ordered by final ranking score
    embedding_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    reranking_used: bool = False        # True if cross-encoder reranking was applied
    rerank_scores: list[float] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return len(self.evidence) > 0

    @property
    def top_similarity(self) -> float:
        """Similarity score of the best-matching evidence chunk."""
        return self.evidence[0].similarity if self.evidence else 0.0


@dataclass
class RetrieverResult:
    claim_evidences: list[ClaimEvidence]
    total_claims: int
    claims_with_evidence: int
    reranking_enabled: bool = False

    @property
    def coverage_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.claims_with_evidence / self.total_claims


class EvidenceRetriever:
    """
    Retrieves and optionally reranks evidence for each claim.

    With reranking enabled (default):
        FAISS retrieves top-10 → CrossEncoder reranks → top-3 passed to NLI

    With reranking disabled (ENABLE_RERANKING=false):
        FAISS retrieves top-k → passed directly to NLI (V1 behaviour)
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        corpus_index: CorpusIndex,
        top_k: int = settings.retrieval_top_k,
        min_similarity: float = settings.retrieval_min_similarity,
        enable_reranking: bool = settings.enable_reranking,
    ):
        self._embedder = embedding_provider
        self._index = corpus_index
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._enable_reranking = enable_reranking

        # Lazily instantiated — only created if reranking is enabled
        self._reranker = None
        if enable_reranking:
            from core.pipeline.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
                top_k=settings.rerank_top_k,
            )
            logger.info(
                "Reranker enabled: %s → top-%d after reranking",
                settings.reranker_model,
                settings.rerank_top_k,
            )
        else:
            logger.info("Reranking disabled — using FAISS order directly.")

    async def retrieve(self, claims: list[str]) -> RetrieverResult:
        """
        Retrieve (and optionally rerank) evidence for every claim.

        Args:
            claims: Atomic factual claims from the extractor.

        Returns:
            RetrieverResult with one ClaimEvidence per claim.
        """
        if not claims:
            return RetrieverResult(
                claim_evidences=[],
                total_claims=0,
                claims_with_evidence=0,
                reranking_enabled=self._enable_reranking,
            )

        claim_evidences: list[ClaimEvidence] = []

        for claim in claims:
            ce = await self._retrieve_one(claim)
            claim_evidences.append(ce)
            logger.debug(
                "Claim: %.60s → %d chunks  reranked=%s  top_sim=%.3f",
                claim,
                len(ce.evidence),
                ce.reranking_used,
                ce.top_similarity,
            )

        claims_with_evidence = sum(1 for ce in claim_evidences if ce.has_evidence)
        reranked_count = sum(1 for ce in claim_evidences if ce.reranking_used)

        logger.info(
            "Retrieval complete: %d/%d claims have evidence  reranked=%d/%d",
            claims_with_evidence, len(claims),
            reranked_count, len(claims),
        )

        return RetrieverResult(
            claim_evidences=claim_evidences,
            total_claims=len(claims),
            claims_with_evidence=claims_with_evidence,
            reranking_enabled=self._enable_reranking,
        )

    async def _retrieve_one(self, claim: str) -> ClaimEvidence:
        """
        Embed claim → FAISS search → optional reranking.
        """
        # ── Step 1: Embed claim ───────────────────────────────────────────────
        t0 = time.monotonic()
        vector = await self._embedder.embed_single(claim)
        embed_ms = (time.monotonic() - t0) * 1000

        # ── Step 2: FAISS search ──────────────────────────────────────────────
        t1 = time.monotonic()
        faiss_results = self._index.search(
            query_vector=vector,
            top_k=self._top_k,
            min_similarity=self._min_similarity,
        )
        retrieval_ms = (time.monotonic() - t1) * 1000

        if not faiss_results:
            return ClaimEvidence(
                claim=claim,
                evidence=[],
                embedding_latency_ms=embed_ms,
                retrieval_latency_ms=retrieval_ms,
                reranking_used=False,
            )

        # ── Step 3: Rerank (if enabled) ───────────────────────────────────────
        if self._reranker is not None:
            rerank_output = await self._reranker.rerank(claim, faiss_results)
            return ClaimEvidence(
                claim=claim,
                evidence=rerank_output.reranked,
                embedding_latency_ms=embed_ms,
                retrieval_latency_ms=retrieval_ms,
                rerank_latency_ms=rerank_output.latency_ms,
                reranking_used=not rerank_output.fallback_used,
                rerank_scores=rerank_output.rerank_scores,
            )

        # ── No reranking — return FAISS results directly ──────────────────────
        return ClaimEvidence(
            claim=claim,
            evidence=faiss_results,
            embedding_latency_ms=embed_ms,
            retrieval_latency_ms=retrieval_ms,
            reranking_used=False,
        )