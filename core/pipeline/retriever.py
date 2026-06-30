"""
core/pipeline/retriever.py  —  Step 3: Evidence retrieval.

For each claim, embed it and retrieve the top-k most semantically
similar chunks from the FAISS corpus index.

Returns (claim → evidence list) pairs for downstream NLI verification.
"""

import logging
from dataclasses import dataclass

from core.providers.base import BaseEmbeddingProvider
from corpus.loader import CorpusIndex, SearchResult
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ClaimEvidence:
    """A single claim paired with its retrieved evidence chunks."""
    claim: str
    evidence: list[SearchResult]        # ordered by descending similarity
    embedding_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None

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

    @property
    def coverage_rate(self) -> float:
        """Fraction of claims that found at least one evidence chunk."""
        if self.total_claims == 0:
            return 0.0
        return self.claims_with_evidence / self.total_claims


class EvidenceRetriever:
    """
    Retrieves evidence for each claim using FAISS similarity search.

    Claims are embedded one at a time (not batched) to keep the
    per-claim evidence structure clean. If batched embedding becomes
    a bottleneck at scale, add a batch_retrieve() method in V2.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        corpus_index: CorpusIndex,
        top_k: int = settings.retrieval_top_k,
        min_similarity: float = settings.retrieval_min_similarity,
    ):
        self._embedder = embedding_provider
        self._index = corpus_index
        self._top_k = top_k
        self._min_similarity = min_similarity

    async def retrieve(self, claims: list[str]) -> RetrieverResult:
        """
        Retrieve evidence for every claim in the list.

        Args:
            claims: Atomic factual claims from the extractor.

        Returns:
            RetrieverResult containing one ClaimEvidence per claim.
        """
        if not claims:
            return RetrieverResult(
                claim_evidences=[],
                total_claims=0,
                claims_with_evidence=0,
            )

        claim_evidences: list[ClaimEvidence] = []

        for claim in claims:
            ce = await self._retrieve_one(claim)
            claim_evidences.append(ce)
            logger.debug(
                "Claim: %.60s… → %d evidence chunks (top sim=%.3f)",
                claim,
                len(ce.evidence),
                ce.top_similarity,
            )

        claims_with_evidence = sum(1 for ce in claim_evidences if ce.has_evidence)
        logger.info(
            "Retrieval complete: %d/%d claims have evidence",
            claims_with_evidence,
            len(claims),
        )

        return RetrieverResult(
            claim_evidences=claim_evidences,
            total_claims=len(claims),
            claims_with_evidence=claims_with_evidence,
        )

    async def _retrieve_one(self, claim: str) -> ClaimEvidence:
        """Embed one claim and search the index."""
        import time

        t0 = time.monotonic()
        vector = await self._embedder.embed_single(claim)
        embed_ms = (time.monotonic() - t0) * 1000

        t1 = time.monotonic()
        evidence = self._index.search(
            query_vector=vector,
            top_k=self._top_k,
            min_similarity=self._min_similarity,
        )
        retrieval_ms = (time.monotonic() - t1) * 1000

        return ClaimEvidence(
            claim=claim,
            evidence=evidence,
            embedding_latency_ms=embed_ms,
            retrieval_latency_ms=retrieval_ms,
        )
