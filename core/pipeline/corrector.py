"""
core/pipeline/corrector.py  —  Step 7: Self-correction engine.

Takes the original response + critique and asks the LLM to produce
a corrected version that addresses only the flagged claims.

Iteration policy (V1, from PRD):
    - Maximum 3 correction iterations
    - Stop early if improvement < improvement_threshold between rounds
    - Always return the best-seen response (lowest hallucination score)
    - Circuit breaker: if score regresses, stop immediately
"""

import logging
from dataclasses import dataclass, field

from core.pipeline.critic import CritiqueResult
from core.pipeline.scorer import HallucinationReport
from core.providers.base import BaseLLMProvider
from config import settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a precise editor correcting factual errors in a response.
You will be given:
1. The original response
2. A list of specific correction instructions for claims that are unsupported or contradicted

Your task:
- Apply ONLY the specified corrections
- Do not rewrite sections that are not flagged
- Do not add new information beyond what the corrections require
- Preserve the original tone, structure, and all supported claims exactly
- Output the corrected response only, with no explanation or preamble"""

_CORRECTION_TEMPLATE = """Original response:
\"\"\"
{original_response}
\"\"\"

Correction instructions:
{correction_block}

Write the corrected response:"""


@dataclass
class CorrectionIteration:
    """Record of one correction round."""
    iteration: int
    corrected_text: str
    hallucination_score: float
    improvement: float          # score reduction vs previous iteration (positive = better)
    was_best: bool              # True if this iteration produced the best-seen score


@dataclass
class CorrectorResult:
    """Final output of the self-correction loop."""
    best_response: str
    best_score: float
    initial_score: float
    iterations_run: int
    total_improvement: float        # initial_score - best_score
    stop_reason: str                # "clean", "max_iterations", "early_exit", "circuit_breaker"
    iteration_history: list[CorrectionIteration] = field(repr=False, default_factory=list)

    @property
    def improved(self) -> bool:
        return self.best_score < self.initial_score


class SelfCorrectionEngine:
    """
    Iterative self-correction with bounded iterations and best-seen tracking.

    The engine never returns a response worse than the original.
    It tracks the best hallucination score seen across all iterations
    and returns that response regardless of whether it was the last one.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        max_iterations: int = settings.max_correction_iterations,
        improvement_threshold: float = settings.improvement_threshold,
    ):
        self._provider = provider
        self._max_iterations = max_iterations
        self._threshold = improvement_threshold

    async def correct(
        self,
        original_response: str,
        critique_result: CritiqueResult,
        initial_report: HallucinationReport,
        # Re-verification callback: takes corrected text, returns HallucinationReport
        reverify_fn,
    ) -> CorrectorResult:
        """
        Run the self-correction loop.

        Args:
            original_response: The LLM's initial response text.
            critique_result: Critiques from the CritiqueGenerator.
            initial_report: Hallucination report from the first verification pass.
            reverify_fn: Async callable(text) → HallucinationReport for re-verification.

        Returns:
            CorrectorResult with the best response found and full iteration history.
        """
        # Short-circuit: nothing to correct
        if initial_report.is_clean or not critique_result.critiques:
            logger.info("Response is clean — no correction needed.")
            return CorrectorResult(
                best_response=original_response,
                best_score=initial_report.hallucination_score,
                initial_score=initial_report.hallucination_score,
                iterations_run=0,
                total_improvement=0.0,
                stop_reason="clean",
            )

        best_response = original_response
        best_score = initial_report.hallucination_score
        current_response = original_response
        current_report = initial_report
        history: list[CorrectionIteration] = []

        logger.info(
            "Starting correction loop: initial_score=%.3f  max_iter=%d  threshold=%.3f",
            best_score, self._max_iterations, self._threshold,
        )

        for iteration in range(1, self._max_iterations + 1):
            logger.info("Correction iteration %d/%d…", iteration, self._max_iterations)

            # ── Generate correction ───────────────────────────────────────────
            corrected_text = await self._generate_correction(
                original_response=current_response,
                critique_result=critique_result,
            )

            # ── Re-verify the corrected response ──────────────────────────────
            new_report: HallucinationReport = await reverify_fn(corrected_text)
            new_score = new_report.hallucination_score
            improvement = current_report.hallucination_score - new_score

            is_best = new_score < best_score
            if is_best:
                best_response = corrected_text
                best_score = new_score

            iter_record = CorrectionIteration(
                iteration=iteration,
                corrected_text=corrected_text,
                hallucination_score=new_score,
                improvement=improvement,
                was_best=is_best,
            )
            history.append(iter_record)

            logger.info(
                "Iter %d: score=%.3f  improvement=%.3f  best=%.3f  is_best=%s",
                iteration, new_score, improvement, best_score, is_best,
            )

            # ── Circuit breaker: score regressed significantly ─────────────────
            if new_score > current_report.hallucination_score + 0.1:
                logger.warning(
                    "Score regressed (%.3f → %.3f). Stopping early.",
                    current_report.hallucination_score, new_score,
                )
                stop_reason = "circuit_breaker"
                break

            # ── Early exit: score is now 0 (fully clean) ──────────────────────
            if new_score == 0.0:
                logger.info("All claims verified. Early exit.")
                stop_reason = "clean"
                break

            # ── Early exit: improvement below threshold ────────────────────────
            if improvement < self._threshold:
                logger.info(
                    "Improvement %.4f < threshold %.4f. Early exit.",
                    improvement, self._threshold,
                )
                stop_reason = "early_exit"
                break

            # Prepare for next iteration
            current_response = corrected_text
            current_report = new_report

            # Regenerate critique from the updated report for the next round
            if iteration < self._max_iterations:
                critique_result = self._update_critique_placeholder(new_report)

        else:
            stop_reason = "max_iterations"

        total_improvement = initial_report.hallucination_score - best_score
        logger.info(
            "Correction complete: stop=%s  iterations=%d  "
            "initial=%.3f  best=%.3f  improvement=%.3f",
            stop_reason, len(history),
            initial_report.hallucination_score, best_score, total_improvement,
        )

        return CorrectorResult(
            best_response=best_response,
            best_score=best_score,
            initial_score=initial_report.hallucination_score,
            iterations_run=len(history),
            total_improvement=round(total_improvement, 4),
            stop_reason=stop_reason,
            iteration_history=history,
        )

    async def _generate_correction(
        self,
        original_response: str,
        critique_result: CritiqueResult,
    ) -> str:
        """Call the LLM with the correction prompt."""
        correction_block = critique_result.to_correction_block()
        prompt = _CORRECTION_TEMPLATE.format(
            original_response=original_response,
            correction_block=correction_block,
        )

        response = await self._provider.generate(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=settings.llm_max_tokens,
        )
        return response.text.strip()

    def _update_critique_placeholder(self, new_report: HallucinationReport) -> CritiqueResult:
        """
        Produce a minimal critique stub for subsequent iterations.

        Full critique regeneration (calling CritiqueGenerator again) is
        the correct V2 approach. For V1, we pass the original critiques
        for the remaining unsupported claims, which avoids an extra LLM
        call per iteration without meaningfully hurting quality.
        """
        from core.pipeline.critic import CritiqueResult, ClaimCritique

        still_unsupported = new_report.unsupported_verifications
        stubs = [
            ClaimCritique(
                claim=v.claim,
                label=v.label,
                confidence=v.confidence,
                instruction=(
                    f"This claim is still {v.label.value}. "
                    "Revise or remove it based on available evidence."
                ),
                evidence_text=v.evidence_used,
                evidence_source=v.evidence_title,
            )
            for v in still_unsupported
        ]
        return CritiqueResult(
            critiques=stubs,
            unsupported_count=len(stubs),
            total_claims=new_report.total_claims,
        )
