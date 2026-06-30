"""
core/pipeline/scorer.py  —  Step 5: Hallucination scoring.

Aggregates ClaimVerification results into the 6 metrics specified in
the PRD. Returns a structured HallucinationReport used throughout the
correction loop and experiment logger.
"""

import logging
from dataclasses import dataclass, field

from core.pipeline.verifier import ClaimVerification, NLILabel, VerifierResult

logger = logging.getLogger(__name__)


@dataclass
class HallucinationReport:
    """
    Full hallucination assessment for one response.

    All ratio fields are in [0.0, 1.0].
    A lower hallucination_score means a more trustworthy response.
    """
    # ── Claim counts ──────────────────────────────────────────────────────────
    total_claims: int
    supported_claims: int       # label = ENTAILMENT
    contradicted_claims: int    # label = CONTRADICTION
    neutral_claims: int         # label = NEUTRAL
    uncertain_claims: int       # label = UNCERTAIN (below confidence threshold)
    no_evidence_claims: int     # label = NO_EVIDENCE (no corpus hit)

    # ── Aggregate metrics (PRD §Metrics) ─────────────────────────────────────
    hallucination_score: float          # (contradicted + neutral + uncertain) / total
    avg_nli_confidence: float           # mean confidence across all verified claims
    avg_retrieval_similarity: float     # mean cosine sim of best evidence per claim

    # ── Per-claim detail (passed through for critique + correction) ───────────
    verifications: list[ClaimVerification] = field(repr=False)

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_clean(self) -> bool:
        """True if no claims are contradicted, neutral, or uncertain."""
        return self.hallucination_score == 0.0

    @property
    def unsupported_verifications(self) -> list[ClaimVerification]:
        """Claims that need correction: contradiction + neutral + uncertain."""
        return [
            v for v in self.verifications
            if v.label in (NLILabel.CONTRADICTION, NLILabel.NEUTRAL, NLILabel.UNCERTAIN)
        ]

    def to_dict(self) -> dict:
        """Serialise for experiment logging (excludes full verification objects)."""
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "neutral_claims": self.neutral_claims,
            "uncertain_claims": self.uncertain_claims,
            "no_evidence_claims": self.no_evidence_claims,
            "hallucination_score": round(self.hallucination_score, 4),
            "avg_nli_confidence": round(self.avg_nli_confidence, 4),
            "avg_retrieval_similarity": round(self.avg_retrieval_similarity, 4),
        }


class HallucinationScorer:
    """
    Computes the hallucination report from NLI verification results.

    Hallucination score formula (V1):
        score = (contradicted + neutral + uncertain) / total_claims

    Rationale: contradictions are active falsehoods, neutral means
    unverifiable, uncertain means low-confidence — all three warrant
    correction. NO_EVIDENCE claims are excluded from the denominator
    because absence of corpus coverage is a retrieval gap, not a
    hallucination signal.
    """

    def score(self, verifier_result: VerifierResult) -> HallucinationReport:
        """
        Compute the hallucination report from NLI outputs.

        Args:
            verifier_result: Output from NLIVerifier.verify().

        Returns:
            HallucinationReport with all 6 PRD metrics populated.
        """
        vvs = verifier_result.verifications

        if not vvs:
            return HallucinationReport(
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                neutral_claims=0,
                uncertain_claims=0,
                no_evidence_claims=0,
                hallucination_score=0.0,
                avg_nli_confidence=0.0,
                avg_retrieval_similarity=0.0,
                verifications=[],
            )

        # ── Label counts ──────────────────────────────────────────────────────
        supported     = sum(1 for v in vvs if v.label == NLILabel.ENTAILMENT)
        contradicted  = sum(1 for v in vvs if v.label == NLILabel.CONTRADICTION)
        neutral       = sum(1 for v in vvs if v.label == NLILabel.NEUTRAL)
        uncertain     = sum(1 for v in vvs if v.label == NLILabel.UNCERTAIN)
        no_evidence   = sum(1 for v in vvs if v.label == NLILabel.NO_EVIDENCE)

        # ── Hallucination score ───────────────────────────────────────────────
        # Denominator: claims that had evidence (i.e. could be verified)
        verifiable = len(vvs) - no_evidence
        if verifiable > 0:
            h_score = (contradicted + neutral + uncertain) / verifiable
        else:
            # All claims had no corpus evidence — cannot assess
            h_score = 0.0

        # ── Average NLI confidence ────────────────────────────────────────────
        # Include only claims that went through NLI (exclude NO_EVIDENCE)
        nli_confidences = [v.confidence for v in vvs if v.label != NLILabel.NO_EVIDENCE]
        avg_confidence = sum(nli_confidences) / len(nli_confidences) if nli_confidences else 0.0

        # ── Average retrieval similarity ──────────────────────────────────────
        similarities = [v.evidence_similarity for v in vvs if v.evidence_similarity > 0]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        report = HallucinationReport(
            total_claims=len(vvs),
            supported_claims=supported,
            contradicted_claims=contradicted,
            neutral_claims=neutral,
            uncertain_claims=uncertain,
            no_evidence_claims=no_evidence,
            hallucination_score=round(h_score, 4),
            avg_nli_confidence=round(avg_confidence, 4),
            avg_retrieval_similarity=round(avg_similarity, 4),
            verifications=vvs,
        )

        logger.info(
            "Hallucination report: score=%.3f  supported=%d  contradicted=%d  "
            "neutral=%d  uncertain=%d  no_evidence=%d",
            report.hallucination_score,
            supported, contradicted, neutral, uncertain, no_evidence,
        )

        return report
