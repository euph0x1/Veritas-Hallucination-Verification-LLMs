"""
core/orchestrator.py  —  Full pipeline coordinator.

The orchestrator owns the complete Generate → Extract → Retrieve →
Verify → Score → Critique → Correct → Re-Verify flow.

It is the only module that knows about the full pipeline sequence.
Individual pipeline modules are unaware of each other — they only
know their own inputs and outputs.

Re-verification (V1 decision):
    Only re-verify claims that were previously NEUTRAL or CONTRADICTION.
    Full pipeline re-extraction is deferred to V2.
"""

import logging
import time
from dataclasses import dataclass, field

from core.pipeline.generator import ResponseGenerator, GeneratorResult
from core.pipeline.extractor import ClaimExtractor, ExtractorResult
from core.pipeline.retriever import EvidenceRetriever, RetrieverResult
from core.pipeline.verifier import NLIVerifier, VerifierResult, NLILabel, ClaimVerification
from core.pipeline.scorer import HallucinationScorer, HallucinationReport
from core.pipeline.critic import CritiqueGenerator, CritiqueResult
from core.pipeline.corrector import SelfCorrectionEngine, CorrectorResult
from core.providers.base import BaseLLMProvider, BaseEmbeddingProvider
from corpus.loader import CorpusIndex

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output of one full Veritas pipeline run."""
    query: str

    # ── Per-stage outputs ─────────────────────────────────────────────────────
    initial_response: str
    claims: list[str]
    initial_report: HallucinationReport
    critique: CritiqueResult
    correction: CorrectorResult
    final_report: HallucinationReport

    # ── Top-level summary ──────────────────────────────────────────────────────
    final_response: str             # the best corrected (or original) response
    total_latency_ms: float

    # ── Parse metadata ─────────────────────────────────────────────────────────
    claim_parse_error: str | None = None

    def summary(self) -> dict:
        """Compact summary for API responses and logging."""
        return {
            "query": self.query,
            "final_response": self.final_response,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "claims_extracted": len(self.claims),
            "claim_parse_error": self.claim_parse_error,
            "initial_hallucination_score": self.initial_report.hallucination_score,
            "final_hallucination_score": self.final_report.hallucination_score,
            "total_improvement": self.correction.total_improvement,
            "correction_iterations": self.correction.iterations_run,
            "stop_reason": self.correction.stop_reason,
            "initial_metrics": self.initial_report.to_dict(),
            "final_metrics": self.final_report.to_dict(),
        }


class VeritasPipeline:
    """
    End-to-end hallucination verification pipeline.

    Instantiate once at application startup with pre-loaded providers
    and corpus index. All methods are async and safe for concurrent use.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        embedding_provider: BaseEmbeddingProvider,
        corpus_index: CorpusIndex,
    ):
        self._generator  = ResponseGenerator(llm_provider)
        self._extractor  = ClaimExtractor(llm_provider)
        self._retriever  = EvidenceRetriever(embedding_provider, corpus_index)
        self._verifier   = NLIVerifier()
        self._scorer     = HallucinationScorer()
        self._critic     = CritiqueGenerator(llm_provider)
        self._corrector  = SelfCorrectionEngine(llm_provider)

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, query: str) -> PipelineResult:
        """
        Execute the complete pipeline for a user query.

        Steps:
            1. Generate initial response
            2. Extract atomic claims
            3. Retrieve evidence per claim
            4. NLI verify claims
            5. Compute hallucination score
            6. Generate targeted critiques
            7. Self-correction loop (max 3 iterations)
            8. Re-verify corrected claims only
            9. Return best response + full report
        """
        t_start = time.monotonic()
        logger.info("═══ Pipeline start: %.80s ═══", query)

        # ── Step 1: Generate ──────────────────────────────────────────────────
        gen_result: GeneratorResult = await self._generator.generate(query)
        logger.info("Step 1 ✓ Generated response (%d chars)", len(gen_result.response_text))

        # ── Step 2: Extract claims ────────────────────────────────────────────
        extract_result: ExtractorResult = await self._extractor.extract(gen_result.response_text)
        claims = extract_result.claims
        logger.info("Step 2 ✓ Extracted %d claims (error=%s)", len(claims), extract_result.parse_error)

        if not claims:
            # No verifiable claims — return the original response as-is
            logger.warning("No claims extracted — returning original response without verification.")
            empty_report = self._scorer.score(VerifierResult(verifications=[]))
            empty_critique = CritiqueResult(critiques=[], unsupported_count=0, total_claims=0)
            empty_correction = CorrectorResult(
                best_response=gen_result.response_text,
                best_score=0.0,
                initial_score=0.0,
                iterations_run=0,
                total_improvement=0.0,
                stop_reason="no_claims",
            )
            return PipelineResult(
                query=query,
                initial_response=gen_result.response_text,
                claims=[],
                initial_report=empty_report,
                critique=empty_critique,
                correction=empty_correction,
                final_report=empty_report,
                final_response=gen_result.response_text,
                total_latency_ms=(time.monotonic() - t_start) * 1000,
                claim_parse_error=extract_result.parse_error,
            )

        # ── Step 3: Retrieve evidence ─────────────────────────────────────────
        retriever_result: RetrieverResult = await self._retriever.retrieve(claims)
        logger.info(
            "Step 3 ✓ Retrieval coverage: %d/%d claims",
            retriever_result.claims_with_evidence, retriever_result.total_claims,
        )

        # ── Step 4 + 5: Verify + Score ────────────────────────────────────────
        verifier_result: VerifierResult = await self._verifier.verify(retriever_result.claim_evidences)
        initial_report: HallucinationReport = self._scorer.score(verifier_result)
        logger.info("Step 4+5 ✓ Initial hallucination score: %.3f", initial_report.hallucination_score)

        # ── Step 6: Critique ──────────────────────────────────────────────────
        critique_result: CritiqueResult = await self._critic.critique(initial_report)
        logger.info("Step 6 ✓ Generated %d critiques", len(critique_result.critiques))

        # ── Step 7+8: Self-correction loop with re-verification ───────────────
        correction_result: CorrectorResult = await self._corrector.correct(
            original_response=gen_result.response_text,
            critique_result=critique_result,
            initial_report=initial_report,
            reverify_fn=self._reverify,
        )
        logger.info(
            "Step 7+8 ✓ Correction: stop=%s  iter=%d  score %.3f → %.3f",
            correction_result.stop_reason,
            correction_result.iterations_run,
            correction_result.initial_score,
            correction_result.best_score,
        )

        # ── Final re-verification on best response ────────────────────────────
        final_report = await self._reverify(correction_result.best_response)

        total_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "═══ Pipeline complete in %.0fms. Final score: %.3f ═══",
            total_ms, final_report.hallucination_score,
        )

        return PipelineResult(
            query=query,
            initial_response=gen_result.response_text,
            claims=claims,
            initial_report=initial_report,
            critique=critique_result,
            correction=correction_result,
            final_report=final_report,
            final_response=correction_result.best_response,
            total_latency_ms=total_ms,
            claim_parse_error=extract_result.parse_error,
        )

    async def verify_only(self, text: str) -> HallucinationReport:
        """
        Run verification on arbitrary text without generation or correction.
        Used by the /verify API endpoint.
        """
        extract = await self._extractor.extract(text)
        if not extract.claims:
            return self._scorer.score(VerifierResult(verifications=[]))
        retrieval = await self._retriever.retrieve(extract.claims)
        verification = await self._verifier.verify(retrieval.claim_evidences)
        return self._scorer.score(verification)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _reverify(self, text: str) -> HallucinationReport:
        """
        Re-verification strategy (V1):
            Re-extract and re-verify only the unsupported claims from the
            corrected text by running the full mini-pipeline on the new text.

        This is simpler and more correct than trying to match old claims
        to new text positions. The slight overhead is acceptable in V1.

        V2 option: cache retrieval results and only re-run NLI on changed claims.
        """
        return await self.verify_only(text)
