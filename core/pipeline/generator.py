"""
core/pipeline/generator.py  —  Step 1: Generate initial response.

Takes a user query, calls the LLM provider, and returns a structured
response. The system prompt encourages factual, specific answers —
which makes claims easier to extract and verify downstream.
"""

import logging
from dataclasses import dataclass

from core.providers.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable and precise assistant.
When answering questions:
- State facts clearly and specifically (include dates, names, numbers where relevant)
- Do not hedge with vague language like "it might be" or "some say"
- If you are confident about a fact, state it directly
- Keep your response focused and factual"""


@dataclass
class GeneratorResult:
    query: str
    response_text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float | None


class ResponseGenerator:
    """Wraps the LLM provider to produce an initial draft response."""

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def generate(self, query: str) -> GeneratorResult:
        """
        Generate a response to the user query.

        Args:
            query: The user's question or instruction.

        Returns:
            GeneratorResult with the response text and metadata.
        """
        logger.info("Generating response for query: %.80s…", query)

        result: LLMResponse = await self._provider.generate(
            prompt=query,
            system_prompt=_SYSTEM_PROMPT,
        )

        logger.info(
            "Generated %d chars in %.0fms",
            len(result.text),
            result.latency_ms or 0,
        )

        return GeneratorResult(
            query=query,
            response_text=result.text,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
        )
