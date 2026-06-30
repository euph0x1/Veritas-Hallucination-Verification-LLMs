"""
V1 provider implementations.

OllamaProvider     — LLM via local Ollama server
SentenceTransformerProvider — Embeddings via sentence-transformers library

Both implement the base interfaces; nothing else in the pipeline
imports these classes directly (only via the factory in __init__).
"""

import time
import json
import asyncio
import logging
from typing import Optional

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from core.providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    LLMResponse,
    EmbeddingResponse,
    ProviderError,
    ProviderUnavailableError,
    ModelNotFoundError,
)
from config import settings

logger = logging.getLogger(__name__)


# ── Ollama LLM Provider ───────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    """
    Calls the Ollama /api/generate endpoint.

    Ollama streams tokens by default; we set stream=False to get a
    single JSON response, which is simpler and sufficient for V1.
    """

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
        timeout: int = settings.llm_timeout,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.llm_temperature,
                "num_predict": max_tokens if max_tokens is not None else settings.llm_max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.monotonic()
        try:
            response = await self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                "ollama", f"Cannot reach Ollama at {self._base_url}. Is it running?", cause=e
            )
        except httpx.TimeoutException as e:
            raise ProviderUnavailableError(
                "ollama", f"Request timed out after {self._timeout}s", cause=e
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(
                    "ollama", f"Model '{self._model}' not found. Run: ollama pull {self._model}", cause=e
                )
            raise ProviderError("ollama", f"HTTP {e.response.status_code}: {e.response.text}", cause=e)

        latency_ms = (time.monotonic() - t0) * 1000
        data = response.json()

        return LLMResponse(
            text=data.get("response", "").strip(),
            model=self.model_name,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model (or a version of it) is available
            model_base = self._model.split(":")[0]
            available = any(model_base in m for m in models)
            if not available:
                logger.warning(
                    "Ollama is running but model '%s' not found. "
                    "Available: %s. Run: ollama pull %s",
                    self._model, models, self._model
                )
            return available
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False

    async def close(self):
        await self._client.aclose()


# ── SentenceTransformer Embedding Provider ────────────────────────────────────

class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local embeddings via the sentence-transformers library.

    The model is loaded once on first use (lazy init) to avoid
    blocking the event loop at import time.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
    ):
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dim: Optional[int] = None

    def _load(self):
        """Lazy-load the model on first embed call."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            # Determine dimension from a test encode
            probe = self._model.encode(["test"], convert_to_numpy=True)
            self._dim = probe.shape[1]
            logger.info("Embedding model loaded. Dimension: %d", self._dim)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            raise ProviderError("sentence_transformer", "embed() called with empty text list")

        # Run blocking model inference in thread pool to not block the event loop
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, self._encode, texts)
        latency_ms = (time.monotonic() - t0) * 1000

        return EmbeddingResponse(
            vectors=vectors,
            model=self._model_name,
            latency_ms=latency_ms,
        )

    async def embed_single(self, text: str) -> np.ndarray:
        response = await self.embed([text])
        return response.vectors[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Synchronous encode — called from thread pool."""
        self._load()
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine sim via dot product after this
            show_progress_bar=False,
        ).astype(np.float32)

    async def health_check(self) -> bool:
        try:
            await self.embed_single("health check")
            return True
        except Exception as e:
            logger.error("SentenceTransformer health check failed: %s", e)
            return False


# ── Provider factory ──────────────────────────────────────────────────────────

def get_llm_provider() -> BaseLLMProvider:
    """
    Return the configured LLM provider.
    Add new providers here as V2 options.
    """
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaProvider()
    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        f"Supported in V1: ['ollama']"
    )


def get_embedding_provider() -> BaseEmbeddingProvider:
    """
    Return the configured embedding provider.
    Add new providers here as V2 options.
    """
    provider = settings.embedding_provider.lower()
    if provider == "sentence_transformer":
        return SentenceTransformerProvider()
    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        f"Supported in V1: ['sentence_transformer']"
    )
