"""
Veritas configuration.
All settings are read from environment variables or .env file.
Override any value by setting the corresponding env var.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # ── LLM provider ─────────────────────────────────────────────────────────
    llm_provider: str = Field("ollama", description="LLM backend: 'ollama'")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field("gemma3", description="Model name pulled in Ollama")
    llm_temperature: float = Field(0.1, description="Low temp for factual generation")
    llm_max_tokens: int = Field(1024, description="Max tokens for LLM responses")
    llm_timeout: int = Field(120, description="Request timeout in seconds")

    # ── Embedding provider ────────────────────────────────────────────────────
    embedding_provider: str = Field("sentence_transformer", description="Embedding backend")
    embedding_model: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="SentenceTransformer model name"
    )
    embedding_dimension: int = Field(384, description="Output dimension of embedding model")

    # ── FAISS / retrieval ─────────────────────────────────────────────────────
    faiss_index_path: Path = Field(
        Path("data/index/faiss.index"),
        description="Path to saved FAISS index file"
    )
    faiss_metadata_path: Path = Field(
        Path("data/index/metadata.json"),
        description="Path to chunk metadata (text, source, etc.)"
    )
    retrieval_top_k: int = Field(
        10,
        description=(
            "Number of chunks FAISS retrieves per claim. "
            "Set to 10 when reranking is enabled so the reranker "
            "has enough candidates to select from."
        )
    )
    retrieval_min_similarity: float = Field(
        0.55,
        description=(
            "Minimum cosine similarity to include a chunk as evidence. "
            "0.55 filters out loosely-related corpus chunks that would produce "
            "misleading NLI labels. Lower only if corpus coverage is poor."
        )
    )

    # ── Reranker ──────────────────────────────────────────────────────────────
    enable_reranking: bool = Field(
        True,
        description=(
            "Enable cross-encoder reranking of FAISS results before NLI. "
            "Reranking scores (claim, evidence) pairs jointly for much higher "
            "relevance precision than bi-encoder retrieval alone."
        )
    )
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description=(
            "Cross-encoder model for reranking. "
            "ms-marco-MiniLM-L-6-v2 is fast and accurate. "
            "Upgrade to ms-marco-MiniLM-L-12-v2 for higher accuracy."
        )
    )
    rerank_top_k: int = Field(
        3,
        description=(
            "Number of chunks passed to NLI after reranking. "
            "Top-3 gives the verifier focused, high-quality evidence "
            "without overwhelming it with marginal chunks."
        )
    )

    # ── NLI verifier ─────────────────────────────────────────────────────────
    nli_model: str = Field(
        "cross-encoder/nli-deberta-v3-small",
        description="HuggingFace NLI model"
    )
    nli_confidence_threshold: float = Field(
        0.65,
        description="Below this confidence the label is overridden to 'uncertain'"
    )
    nli_batch_size: int = Field(8, description="Pairs per NLI forward pass")

    # ── Corpus ────────────────────────────────────────────────────────────────
    corpus_chunk_size: int = Field(
        100,
        description="Chunk size in words — smaller = more precise NLI evidence"
    )
    corpus_chunk_overlap: int = Field(20, description="Overlap between consecutive chunks")
    corpus_max_articles: int = Field(500, description="Wikipedia articles to index in V1")

    # ── Self-correction loop ──────────────────────────────────────────────────
    max_correction_iterations: int = Field(3, description="Hard cap on correction rounds")
    improvement_threshold: float = Field(
        0.05,
        description="Minimum hallucination score reduction to continue iterating"
    )

    # ── Experiment logging ────────────────────────────────────────────────────
    log_dir: Path = Field(Path("data/logs"), description="Directory for JSON experiment logs")
    log_enabled: bool = Field(True, description="Set False to disable experiment logging")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)
    api_debug: bool = Field(False)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Module-level singleton — import this everywhere
settings = Settings()