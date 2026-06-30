"""
tests/unit/test_pipeline.py

Unit tests for individual pipeline modules.
Uses mocks so no real Ollama or FAISS is required.
Run: pytest tests/unit/ -v
"""

import json
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from core.providers.base import LLMResponse, EmbeddingResponse
from core.pipeline.extractor import ClaimExtractor
from core.pipeline.scorer import HallucinationScorer
from core.pipeline.verifier import VerifierResult, ClaimVerification, NLILabel
from core.pipeline.critic import CritiqueGenerator
from core.pipeline.corrector import SelfCorrectionEngine
from core.pipeline.scorer import HallucinationReport


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_llm_provider(response_text: str = '["Claim one.", "Claim two."]'):
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=LLMResponse(
        text=response_text,
        model="mock/test",
        latency_ms=10.0,
    ))
    return provider


def make_verification(label: NLILabel, confidence: float = 0.9) -> ClaimVerification:
    return ClaimVerification(
        claim="Test claim",
        label=label,
        confidence=confidence,
        entailment_score=0.9 if label == NLILabel.ENTAILMENT else 0.05,
        neutral_score=0.9 if label == NLILabel.NEUTRAL else 0.05,
        contradiction_score=0.9 if label == NLILabel.CONTRADICTION else 0.05,
        evidence_used="Some evidence text",
        evidence_title="Wikipedia: Test",
        evidence_similarity=0.85,
    )


# ── ClaimExtractor tests ──────────────────────────────────────────────────────

class TestClaimExtractor:

    @pytest.mark.asyncio
    async def test_extracts_valid_json_array(self):
        provider = make_llm_provider('["Claim A.", "Claim B.", "Claim C."]')
        extractor = ClaimExtractor(provider)
        result = await extractor.extract("Marie Curie was born in Warsaw in 1867.")
        assert result.claims == ["Claim A.", "Claim B.", "Claim C."]
        assert result.parse_error is None

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        provider = make_llm_provider('```json\n["Claim A.", "Claim B."]\n```')
        extractor = ClaimExtractor(provider)
        result = await extractor.extract("Some text.")
        assert result.claims == ["Claim A.", "Claim B."]
        assert result.parse_error == "stripped_markdown_fences"

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_claims(self):
        provider = make_llm_provider("[]")
        extractor = ClaimExtractor(provider)
        result = await extractor.extract("")
        assert result.claims == []
        assert result.parse_error == "Empty input"

    @pytest.mark.asyncio
    async def test_fallback_to_line_heuristic(self):
        provider = make_llm_provider("- Claim one here.\n- Claim two here.")
        extractor = ClaimExtractor(provider)
        result = await extractor.extract("Some text")
        assert len(result.claims) == 2
        assert result.parse_error == "line_heuristic_fallback"

    def test_parse_extracts_json_substring(self):
        extractor = ClaimExtractor(make_llm_provider())
        raw = 'Here are the claims: ["Fact one.", "Fact two."] Hope that helps!'
        claims, error = extractor._parse(raw)
        assert claims == ["Fact one.", "Fact two."]
        assert error == "extracted_json_substring"


# ── HallucinationScorer tests ─────────────────────────────────────────────────

class TestHallucinationScorer:
    scorer = HallucinationScorer()

    def test_all_supported_score_is_zero(self):
        vv = VerifierResult(verifications=[
            make_verification(NLILabel.ENTAILMENT),
            make_verification(NLILabel.ENTAILMENT),
        ])
        report = self.scorer.score(vv)
        assert report.hallucination_score == 0.0
        assert report.supported_claims == 2
        assert report.is_clean is True

    def test_half_contradicted_score_is_half(self):
        vv = VerifierResult(verifications=[
            make_verification(NLILabel.ENTAILMENT),
            make_verification(NLILabel.CONTRADICTION),
        ])
        report = self.scorer.score(vv)
        assert report.hallucination_score == 0.5
        assert report.contradicted_claims == 1

    def test_no_evidence_excluded_from_denominator(self):
        vv = VerifierResult(verifications=[
            make_verification(NLILabel.ENTAILMENT),
            ClaimVerification(
                claim="Unverifiable claim",
                label=NLILabel.NO_EVIDENCE,
                confidence=0.0,
                entailment_score=0.0,
                neutral_score=0.0,
                contradiction_score=0.0,
                evidence_used=None,
                evidence_title=None,
                evidence_similarity=0.0,
            ),
        ])
        report = self.scorer.score(vv)
        # 1 verifiable (entailment), 0 unsupported → score = 0
        assert report.hallucination_score == 0.0
        assert report.no_evidence_claims == 1

    def test_uncertain_counts_as_unsupported(self):
        vv = VerifierResult(verifications=[
            make_verification(NLILabel.UNCERTAIN, confidence=0.4),
            make_verification(NLILabel.ENTAILMENT),
        ])
        report = self.scorer.score(vv)
        assert report.uncertain_claims == 1
        assert report.hallucination_score == 0.5

    def test_empty_verifications(self):
        report = self.scorer.score(VerifierResult(verifications=[]))
        assert report.total_claims == 0
        assert report.hallucination_score == 0.0

    def test_to_dict_keys(self):
        vv = VerifierResult(verifications=[make_verification(NLILabel.ENTAILMENT)])
        report = self.scorer.score(vv)
        d = report.to_dict()
        expected_keys = {
            "total_claims", "supported_claims", "contradicted_claims",
            "neutral_claims", "uncertain_claims", "no_evidence_claims",
            "hallucination_score", "avg_nli_confidence", "avg_retrieval_similarity",
        }
        assert expected_keys == set(d.keys())


# ── NLI confidence threshold tests ───────────────────────────────────────────

class TestNLIVerifier:

    def test_decide_label_high_confidence_entailment(self):
        from core.pipeline.verifier import NLIVerifier
        v = NLIVerifier(confidence_threshold=0.65)
        label, conf = v._decide_label({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
        assert label == NLILabel.ENTAILMENT
        assert conf == pytest.approx(0.9)

    def test_decide_label_below_threshold_returns_uncertain(self):
        from core.pipeline.verifier import NLIVerifier
        v = NLIVerifier(confidence_threshold=0.65)
        label, conf = v._decide_label({"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2})
        assert label == NLILabel.UNCERTAIN
        assert conf == pytest.approx(0.5)

    def test_decide_label_contradiction(self):
        from core.pipeline.verifier import NLIVerifier
        v = NLIVerifier(confidence_threshold=0.65)
        label, conf = v._decide_label({"entailment": 0.05, "neutral": 0.1, "contradiction": 0.85})
        assert label == NLILabel.CONTRADICTION

    def test_decide_label_empty_scores_returns_uncertain(self):
        from core.pipeline.verifier import NLIVerifier
        v = NLIVerifier(confidence_threshold=0.65)
        label, conf = v._decide_label({})
        assert label == NLILabel.UNCERTAIN
        assert conf == 0.0


# ── SelfCorrectionEngine tests ────────────────────────────────────────────────

class TestSelfCorrectionEngine:

    def _make_report(self, score: float, clean: bool = False) -> HallucinationReport:
        label = NLILabel.ENTAILMENT if clean else NLILabel.CONTRADICTION
        vv = VerifierResult(verifications=[make_verification(label)])
        scorer = HallucinationScorer()
        # Manually override score for testing
        report = scorer.score(vv)
        report.hallucination_score = score
        return report

    @pytest.mark.asyncio
    async def test_skips_correction_when_clean(self):
        provider = make_llm_provider("Corrected text")
        engine = SelfCorrectionEngine(provider, max_iterations=3)

        from core.pipeline.critic import CritiqueResult
        clean_report = self._make_report(0.0, clean=True)
        clean_critique = CritiqueResult(critiques=[], unsupported_count=0, total_claims=1)

        result = await engine.correct(
            original_response="Original",
            critique_result=clean_critique,
            initial_report=clean_report,
            reverify_fn=AsyncMock(return_value=clean_report),
        )
        assert result.stop_reason == "clean"
        assert result.iterations_run == 0
        assert result.best_response == "Original"

    @pytest.mark.asyncio
    async def test_early_exit_on_no_improvement(self):
        from core.pipeline.critic import CritiqueResult, ClaimCritique
        provider = make_llm_provider("Corrected text")
        engine = SelfCorrectionEngine(provider, max_iterations=3, improvement_threshold=0.1)

        initial_report = self._make_report(0.5)
        # Reverify always returns the same score (no improvement)
        same_report = self._make_report(0.49)  # improvement = 0.01, below 0.1 threshold

        stub_critique = CritiqueResult(
            critiques=[ClaimCritique(
                claim="Test", label=NLILabel.CONTRADICTION, confidence=0.8,
                instruction="Fix this", evidence_text=None, evidence_source=None
            )],
            unsupported_count=1,
            total_claims=1,
        )

        result = await engine.correct(
            original_response="Original",
            critique_result=stub_critique,
            initial_report=initial_report,
            reverify_fn=AsyncMock(return_value=same_report),
        )
        assert result.stop_reason == "early_exit"
        assert result.iterations_run == 1
