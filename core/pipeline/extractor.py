"""
core/pipeline/extractor.py  —  Step 2: Extract atomic claims.

Converts a response paragraph into a list of independent, verifiable
factual claims using the fixed prompt decided in the PRD.

The prompt and output schema are intentionally fixed for V1 to keep
results consistent across experiments. Do not modify the prompt
between experiments — create a new extractor version instead.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# ── Fixed V1 prompt ───────────────────────────────────────────────────────────
# This prompt is locked for V1. Changing it invalidates experiment comparisons.

_SYSTEM_PROMPT = """You are a precise claim extractor. Given a passage of text, extract every \
independent, verifiable factual claim. Each claim must be:
- Self-contained (no pronouns referring to other claims)
- A single atomic fact (do not combine two facts in one claim)
- Stated as a declarative sentence
- Verifiable against an external source

Do not extract opinions, predictions, procedural instructions, or definitions.

Return ONLY a JSON array of strings. No explanation, no preamble, no markdown fences.

Example input:
"Marie Curie was born in Warsaw in 1867. She won two Nobel Prizes and pioneered research into radioactivity."

Example output:
["Marie Curie was born in Warsaw.", "Marie Curie was born in 1867.", "Marie Curie won two Nobel Prizes.", "Marie Curie pioneered research into radioactivity."]"""

_USER_TEMPLATE = 'Extract all atomic factual claims from the following text:\n\n"""\n{text}\n"""'


@dataclass
class ExtractorResult:
    claims: list[str]
    raw_response: str
    parse_error: str | None = None    # set if JSON parsing fell back to heuristics


class ClaimExtractor:
    """
    Extracts atomic factual claims from a text passage.

    Uses the LLM with a structured JSON output prompt. Includes
    a fallback parser in case the model returns imperfect JSON.
    """

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def extract(self, text: str) -> ExtractorResult:
        """
        Extract atomic claims from the given text.

        Args:
            text: A paragraph or multi-sentence response to decompose.

        Returns:
            ExtractorResult with a list of claim strings.
            Returns empty list if the text contains no verifiable claims.
        """
        if not text or not text.strip():
            return ExtractorResult(claims=[], raw_response="", parse_error="Empty input")

        prompt = _USER_TEMPLATE.format(text=text.strip())
        logger.debug("Extracting claims from %d-char text", len(text))

        response = await self._provider.generate(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,        # deterministic — claim extraction must be stable
            max_tokens=1024,
        )

        raw = response.text.strip()
        claims, parse_error = self._parse(raw)

        logger.info("Extracted %d claims (parse_error=%s)", len(claims), parse_error)
        return ExtractorResult(
            claims=claims,
            raw_response=raw,
            parse_error=parse_error,
        )

    def _parse(self, raw: str) -> tuple[list[str], str | None]:
        """
        Parse the LLM output into a list of claim strings.

        Tries strict JSON first, then falls back to heuristics for
        common failure modes (e.g. model wraps output in ```json fences).

        Returns:
            (claims, parse_error_message_or_None)
        """
        # ── Attempt 1: strict JSON ────────────────────────────────────────────
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                claims = [str(c).strip() for c in data if str(c).strip()]
                return claims, None
        except json.JSONDecodeError:
            pass

        # ── Attempt 2: strip markdown fences then parse ───────────────────────
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                claims = [str(c).strip() for c in data if str(c).strip()]
                return claims, "stripped_markdown_fences"
        except json.JSONDecodeError:
            pass

        # ── Attempt 3: extract JSON array substring ───────────────────────────
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    claims = [str(c).strip() for c in data if str(c).strip()]
                    return claims, "extracted_json_substring"
            except json.JSONDecodeError:
                pass

        # ── Attempt 4: line-by-line heuristic ────────────────────────────────
        # Treat each non-empty line as a claim (last resort)
        lines = [
            line.strip().lstrip("-•*0123456789.) ")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 10
        ]
        if lines:
            return lines, "line_heuristic_fallback"

        return [], "parse_failed_no_claims"
