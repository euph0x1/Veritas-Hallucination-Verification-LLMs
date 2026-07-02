"""
core/pipeline/critic.py  —  Step 6: Targeted critique generation.

V1 approach: rule-based critique generation.

Instead of relying on the LLM to interpret evidence (which causes hallucinated
correction instructions when evidence chunks are low-quality), this version uses
a structured rule-based system:

1. For CONTRADICTION with high-similarity evidence (>= 0.75):
   Call the LLM with strict rules and evidence relevance check.

2. For CONTRADICTION with low-similarity evidence (< 0.75):
   Use a safe generic instruction — the evidence is probably about a
   different aspect of the topic, not the specific claim.

3. For NEUTRAL / UNCERTAIN:
   Always use generic instruction — no strong evidence either way.

4. For NO_EVIDENCE:
   Template-only, no LLM call.

This eliminates the "LL.D. from Edinburgh → remove birthplace claim" bug
by never trusting low-similarity evidence for generating specific corrections.
"""

import logging
from dataclasses import dataclass

from core.pipeline.scorer import HallucinationReport
from core.pipeline.verifier import ClaimVerification, NLILabel
from core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Only call LLM for high-similarity evidence
HIGH_SIMILARITY_THRESHOLD = 0.75

_SYSTEM_PROMPT = """You are a fact-checker writing one correction instruction.

You will receive a CLAIM and EVIDENCE that directly contradicts it.

Your task: write ONE sentence identifying the specific factual error and what 
the correct value is, based only on what the evidence explicitly states.

STRICT RULES:
- Output the instruction sentence only. Nothing else.
- Only reference facts explicitly stated in the evidence text.
- Never say "remove this claim" unless the evidence explicitly states the 
  claim's subject never did or never was the thing claimed.
- Never fabricate a correction value not present in the evidence.
- If you cannot identify the specific error from the evidence, output:
  "This claim could not be reliably verified — consider qualifying it with 'reportedly'."

EXAMPLES:
Claim: "Bell invented the television."
Evidence: "Bell did not develop television technology. Television was invented by Philo Farnsworth."
Instruction: "Remove this claim — Bell did not invent television; it was invented by Philo Farnsworth."

Claim: "Einstein was born in 1900."
Evidence: "Albert Einstein was born on March 14, 1879, in Ulm, Germany."
Instruction: "Change the birth year from 1900 to 1879, as Einstein was born on March 14, 1879."

Claim: "Paris is the capital of Germany."
Evidence: "Berlin is the capital and largest city of Germany."
Instruction: "Change 'Germany' to 'France' — Berlin is the capital of Germany, not Paris." """

_CRITIQUE_TEMPLATE = """CLAIM: "{claim}"

EVIDENCE (source: {source}):
"{evidence}"

Write the correction instruction:"""


@dataclass
class ClaimCritique:
    claim: str
    label: NLILabel
    confidence: float
    instruction: str
    evidence_text: str | None
    evidence_source: str | None
    used_llm: bool = False


@dataclass
class CritiqueResult:
    critiques: list[ClaimCritique]
    unsupported_count: int
    total_claims: int

    def to_correction_block(self) -> str:
        if not self.critiques:
            return "No corrections required."
        lines = []
        for i, c in enumerate(self.critiques, 1):
            lines.append(
                f"{i}. Find this exact text in the original: \"{c.claim}\"\n"
                f"   Correction: {c.instruction}"
            )
        return "\n\n".join(lines)


class CritiqueGenerator:

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def critique(self, report: HallucinationReport) -> CritiqueResult:
        unsupported = report.unsupported_verifications

        if not unsupported:
            logger.info("No unsupported claims — no critique needed.")
            return CritiqueResult(
                critiques=[],
                unsupported_count=0,
                total_claims=report.total_claims,
            )

        logger.info("Generating critiques for %d unsupported claims…", len(unsupported))

        critiques = []
        for v in unsupported:
            critique = await self._critique_one(v)
            critiques.append(critique)

        return CritiqueResult(
            critiques=critiques,
            unsupported_count=len(unsupported),
            total_claims=report.total_claims,
        )

    async def _critique_one(self, v: ClaimVerification) -> ClaimCritique:
        """
        Generate a critique using rule-based routing:

        - NO_EVIDENCE or no evidence text → generic template
        - NEUTRAL or UNCERTAIN → generic template (no strong signal)
        - CONTRADICTION with low similarity (<0.75) → generic template
          (evidence is topically related but about a different fact)
        - CONTRADICTION with high similarity (>=0.75) → LLM call
          (evidence is directly relevant to the specific claim)
        """

        # ── Case 1: No evidence at all ────────────────────────────────────────
        if v.label == NLILabel.NO_EVIDENCE or not v.evidence_used:
            return self._generic_critique(v, reason="no_evidence")

        # ── Case 2: Neutral or uncertain — no strong signal ───────────────────
        if v.label in (NLILabel.NEUTRAL, NLILabel.UNCERTAIN):
            return self._generic_critique(v, reason="weak_signal")

        # ── Case 3: Contradiction but low similarity ───────────────────────────
        # Evidence chunk is about the same topic but different aspect.
        # E.g. "Edinburgh degree" chunk contradicting "born in Edinburgh" claim.
        # Trust the similarity score — if it's below threshold, the evidence
        # is not directly addressing the specific fact in the claim.
        if v.evidence_similarity < HIGH_SIMILARITY_THRESHOLD:
            logger.debug(
                "Low similarity (%.3f) for contradiction — using generic critique: %.60s",
                v.evidence_similarity, v.claim,
            )
            return self._generic_critique(v, reason="low_similarity_contradiction")

        # ── Case 4: Contradiction with high similarity — LLM call ─────────────
        return await self._llm_critique(v)

    def _generic_critique(self, v: ClaimVerification, reason: str) -> ClaimCritique:
        """
        Safe generic instruction that never fabricates a correction.
        Used whenever evidence quality is insufficient for a specific fix.
        """
        if v.label == NLILabel.NO_EVIDENCE:
            instruction = (
                "This claim could not be verified against the knowledge base. "
                "Consider qualifying it with 'reportedly' or 'according to some sources'."
            )
        elif v.label == NLILabel.CONTRADICTION and reason == "low_similarity_contradiction":
            instruction = (
                "This claim appears to conflict with available evidence, but the "
                "specific correction cannot be determined reliably. Consider "
                "verifying this fact independently or qualifying it with 'reportedly'."
            )
        else:
            instruction = (
                "This claim could not be verified from available evidence. "
                "Consider qualifying it with 'reportedly' or 'according to some sources'."
            )

        logger.debug(
            "Generic critique [%s, reason=%s]: %.60s", v.label.value, reason, v.claim
        )
        return ClaimCritique(
            claim=v.claim,
            label=v.label,
            confidence=v.confidence,
            instruction=instruction,
            evidence_text=v.evidence_used,
            evidence_source=v.evidence_title,
            used_llm=False,
        )

    async def _llm_critique(self, v: ClaimVerification) -> ClaimCritique:
        """
        LLM-based critique for high-confidence contradictions with
        high-similarity evidence. This is the only path that calls the LLM.
        """
        prompt = _CRITIQUE_TEMPLATE.format(
            claim=v.claim,
            source=v.evidence_title or "Knowledge base",
            evidence=v.evidence_used,
        )
        response = await self._provider.generate(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=120,
        )
        instruction = response.text.strip()

        # Safety net: if instruction is too long or looks like evidence copy,
        # fall back to generic
        if len(instruction) > 250 or _is_evidence_copy(instruction, v.evidence_used):
            logger.warning(
                "LLM critique looks like evidence copy for: %.60s — using generic",
                v.claim,
            )
            return self._generic_critique(v, reason="llm_copy_detected")

        logger.debug("LLM critique [similarity=%.3f]: %.120s", v.evidence_similarity, instruction)
        return ClaimCritique(
            claim=v.claim,
            label=v.label,
            confidence=v.confidence,
            instruction=instruction,
            evidence_text=v.evidence_used,
            evidence_source=v.evidence_title,
            used_llm=True,
        )


def _is_evidence_copy(instruction: str, evidence: str) -> bool:
    """Detect verbatim copying of evidence into the instruction."""
    if not evidence or len(instruction) < 20:
        return False
    evidence_words = evidence.split()
    if len(evidence_words) < 8:
        return False
    for i in range(len(evidence_words) - 7):
        window = " ".join(evidence_words[i:i + 8]).lower()
        if window in instruction.lower():
            return True
    return False
