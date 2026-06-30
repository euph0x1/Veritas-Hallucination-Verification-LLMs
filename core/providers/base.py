"""
Provider abstraction interfaces.

Any new LLM or embedding backend must implement these protocols.
The rest of the pipeline depends only on these interfaces, never on
concrete provider classes directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Structured response from any LLM provider."""
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    raw: Optional[dict] = field(default=None, repr=False)


@dataclass
class EmbeddingResponse:
    """Structured response from any embedding provider."""
    vectors: np.ndarray          # shape: (n_texts, embedding_dim)
    model: str
    latency_ms: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.vectors, np.ndarray):
            self.vectors = np.array(self.vectors, dtype=np.float32)


# ── Provider interfaces ───────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """
    Interface every LLM backend must implement.

    V1: OllamaProvider
    V2: OpenAIProvider, AnthropicProvider, HuggingFaceProvider
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt or instruction.
            system_prompt: Optional system-level instruction.
            temperature: Override the default temperature.
            max_tokens: Override the default max token limit.

        Returns:
            LLMResponse with generated text and metadata.

        Raises:
            ProviderError: On network failure, timeout, or model error.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the model is loaded."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging."""
        ...


class BaseEmbeddingProvider(ABC):
    """
    Interface every embedding backend must implement.

    V1: SentenceTransformerProvider
    V2: OpenAIEmbeddingProvider
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """
        Produce dense vector embeddings for a list of texts.

        Args:
            texts: One or more strings to embed.

        Returns:
            EmbeddingResponse with (n_texts, dim) float32 array.

        Raises:
            ProviderError: On model failure or empty input.
        """
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience wrapper: embed one text and return the 1-D vector.

        Returns:
            1-D float32 numpy array of length embedding_dim.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of vectors this provider produces."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging."""
        ...


# ── Exceptions ────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Raised when a provider call fails in an unrecoverable way."""

    def __init__(self, provider: str, message: str, cause: Optional[Exception] = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}")


class ProviderUnavailableError(ProviderError):
    """Raised when the provider service cannot be reached (network/timeout)."""


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available on the provider."""
