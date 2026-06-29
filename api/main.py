"""
api/main.py  —  FastAPI application factory.

Startup sequence:
    1. Load corpus index from disk (fails fast if index not built)
    2. Initialise provider singletons
    3. Warm up NLI model (avoids cold-start latency on first request)
    4. Register routes

All singletons are stored on app.state so routes can access them
without global variables.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable when running `python api/main.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.providers import get_llm_provider, get_embedding_provider
from core.orchestrator import VeritasPipeline
from corpus.loader import CorpusIndex
from logging_module.logger import ExperimentLogger
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""

    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting Veritas…")

    # 1. Corpus index
    logger.info("Loading corpus index…")
    corpus_index = CorpusIndex()
    try:
        corpus_index.load()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Build the index first: python -m corpus.build_index")
        sys.exit(1)

    # 2. Providers
    logger.info("Initialising providers…")
    llm_provider = get_llm_provider()
    embedding_provider = get_embedding_provider()

    # 3. Warm up embedding model (loads weights on first call)
    logger.info("Warming up embedding model…")
    await embedding_provider.embed_single("warmup")

    # 4. Warm up NLI model
    logger.info("Warming up NLI model…")
    from core.pipeline.verifier import NLIVerifier
    nli_verifier = NLIVerifier()
    nli_verifier._load()

    # 5. Pipeline + logger
    pipeline = VeritasPipeline(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        corpus_index=corpus_index,
    )
    exp_logger = ExperimentLogger()

    # Store on app.state for route access
    app.state.pipeline          = pipeline
    app.state.llm_provider      = llm_provider
    app.state.embedding_provider = embedding_provider
    app.state.corpus_index      = corpus_index
    app.state.exp_logger        = exp_logger

    logger.info("Veritas ready. Corpus: %d chunks.", corpus_index.size)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down Veritas…")
    if hasattr(llm_provider, "close"):
        await llm_provider.close()
    logger.info("Goodbye.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Veritas",
        description=(
            "Evidence-Guided Hallucination Verification & Self-Correction Framework. "
            "Detects unsupported claims in LLM responses using NLI and retrieval-augmented verification."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # tighten for production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info",
    )
