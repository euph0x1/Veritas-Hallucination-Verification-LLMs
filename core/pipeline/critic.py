"""
core/pipeline/critic.py  —  Step 6: Targeted critique generation.

Produces claim-level, actionable critique for each unsupported claim.
The critique is grounded in the retrieved evidence so the self-correction
engine has concrete information to work with, not just a "this is wrong."

Design:
    - One critique prompt per unsupported claim (not one giant prompt)
    - Each critique includes: what's wrong, what evidence says, what to fix
    - Critiques are assembled into a single structured block passed to the corrector
"""

import logging
from dataclasses import dataclass

from core.pipeline.scorer import HallucinationReport
from core.pipeline.verifier import ClaimVerification, NLILabel
from core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a precise fact-checker generating correction instructions.
Given an unsupported or contradicted factual claim and relevant evidence, produce a
single, specific instruction telling the writer exactly what to fix and why.

Your instruction must:
- Be one to three sentences maximum
- Reference the specific conflicting detail (date, name, number, etc.)
- State what the evidence actually says
- Tell the writer exactly what to change

Do not rewrite the text yourself. Do not explain NLI or verification. Output only the instruction."""

_CRITIQUE_TEMPLATE = """Claim: "{claim}"

Verification result: {label} (confidence: {confidence:.0%})

Best matching evidence from corpus:
Source: {source}
Text: "{evidence}"

Write a specific correction instruction for this claim."""

_NO_EVIDENCE_TEMPLATE = """Claim: "{claim}"

Verification result: NO_EVIDENCE — no supporting information was found in the knowledge base.

Write a correction instruction noting that this claim could not be verified and should
either be removed, qualified with uncertainty language, or supported with a citation."""


@dataclass
class ClaimCritique:
    """Critique for one unsupported claim."""
    claim: str
    label: NLILabel
    confidence: float
    instruction: str        # the actionable correction instruction
    evidence_text: str | None
    evidence_source: str | None


@dataclass
class CritiqueResult:
    """All critiques for one response, ready to pass to the corrector."""
    critiques: list[ClaimCritique]
    unsupported_count: int
    total_claims: int

    def to_correction_block(self) -> str:
        """
        Format critiques into a structured block for the correction prompt.

        Each critique is numbered and labelled so the LLM can
        address them systematically.
        """
        if not self.critiques:
            return "No corrections required."

        lines = [
            f"The following {len(self.critiques)} claim(s) require correction:\n"
        ]
        for i, c in enumerate(self.critiques, 1):
            lines.append(
                f"[{i}] Original claim: \"{c.claim}\"\n"
                f"    Issue: {c.label.value.upper()}\n"
                f"    Instruction: {c.instruction}\n"
            )
        return "\n".join(lines)


class CritiqueGenerator:
    """
    Generates targeted, evidence-grounded critiques for unsupported claims.

    Only processes claims labelled CONTRADICTION, NEUTRAL, or UNCERTAIN.
    ENTAILMENT and NO_EVIDENCE claims are handled differently:
    - ENTAILMENT: no critique needed
    - NO_EVIDENCE: uses a template-only critique (no LLM call needed for these)
    """

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def critique(self, report: HallucinationReport) -> CritiqueResult:
        """
        Generate critiques for all unsupported claims in the report.

        Args:
            report: HallucinationReport from the scorer.

        Returns:
            CritiqueResult with one ClaimCritique per unsupported claim.
        """
        unsupported = report.unsupported_verifications

        if not unsupported:
            logger.info("No unsupported claims — no critique needed.")
            return CritiqueResult(
                critiques=[],
                unsupported_count=0,
                total_claims=report.total_claims,
            )

        logger.info("Generating critiques for %d unsupported claims…", len(unsupported))

        critiques: list[ClaimCritique] = []
        for verification in unsupported:
            critique = await self._critique_one(verification)
            critiques.append(critique)

        return CritiqueResult(
            critiques=critiques,
            unsupported_count=len(unsupported),
            total_claims=report.total_claims,
        )

    async def _critique_one(self, v: ClaimVerification) -> ClaimCritique:
        """Generate a critique for a single unsupported claim."""

        if v.label == NLILabel.NO_EVIDENCE:
            # No LLM call needed — template is sufficient
            instruction = (
                f"The claim \"{v.claim}\" could not be verified against any "
                "available evidence. Consider removing it, qualifying it with "
                "uncertainty language (e.g., 'reportedly', 'allegedly'), "
                "or adding a citation."
            )
        else:
            prompt = _CRITIQUE_TEMPLATE.format(
                claim=v.claim,
                label=v.label.value,
                confidence=v.confidence,
                source=v.evidence_title or "Unknown source",
                evidence=v.evidence_used or "No evidence text available",
            )
            response = await self._provider.generate(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=200,
            )
            instruction = response.text.strip()

        logger.debug(
            "Critique for claim [%s]: %.100s…", v.label.value, instruction
        )

        return ClaimCritique(
            claim=v.claim,
            label=v.label,
            confidence=v.confidence,
            instruction=instruction,
            evidence_text=v.evidence_used,
            evidence_source=v.evidence_title,
        )
