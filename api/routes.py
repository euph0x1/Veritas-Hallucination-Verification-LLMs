"""
api/routes.py  —  FastAPI route definitions.

All routes delegate to the VeritasPipeline and ExperimentLogger
singletons stored in app.state (set up in main.py).

Endpoints:
    POST /generate      — generate initial response only
    POST /verify        — verify arbitrary text
    POST /critique      — verify + generate critiques
    POST /correct       — full pipeline on existing text (no generation step)
    POST /pipeline      — complete end-to-end pipeline
    GET  /health        — service + provider health check
    GET  /metrics       — aggregate experiment statistics
    GET  /runs          — list recent experiment runs
    GET  /runs/{run_id} — retrieve a specific run
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    GenerateRequest, GenerateResponse,
    VerifyRequest, VerifyResponse,
    CritiqueRequest, CritiqueResponse,
    CorrectRequest, CorrectResponse,
    PipelineRequest, PipelineResponse,
    HealthResponse, MetricsResponse,
    HallucinationMetrics, ClaimDetail, CritiqueDetail,
)
from core.pipeline.verifier import VerifierResult

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics_from_report(report) -> HallucinationMetrics:
    return HallucinationMetrics(**report.to_dict())


def _claim_details_from_report(report) -> list[ClaimDetail]:
    return [
        ClaimDetail(
            claim=v.claim,
            label=v.label.value,
            confidence=round(v.confidence, 4),
            entailment_score=round(v.entailment_score, 4),
            neutral_score=round(v.neutral_score, 4),
            contradiction_score=round(v.contradiction_score, 4),
            evidence_title=v.evidence_title,
            evidence_similarity=round(v.evidence_similarity, 4),
        )
        for v in report.verifications
    ]


def _critique_details(critique_result) -> list[CritiqueDetail]:
    return [
        CritiqueDetail(
            claim=c.claim,
            label=c.label.value,
            instruction=c.instruction,
            evidence_source=c.evidence_source,
        )
        for c in critique_result.critiques
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/", tags=["System"])
async def root():
    """Veritas API root — links to documentation."""
    return {
        "name": "Veritas",
        "version": "1.0.0",
        "description": "Evidence-Guided Hallucination Verification & Self-Correction Framework",
        "docs": "http://127.0.0.1:8000/docs",
        "redoc": "http://127.0.0.1:8000/redoc",
        "health": "http://127.0.0.1:8000/health",
        "endpoints": {
            "POST /pipeline": "Full end-to-end pipeline",
            "POST /generate": "Generate response only",
            "POST /verify": "Verify text for hallucinations",
            "POST /critique": "Verify + generate critiques",
            "POST /correct": "Verify + critique + self-correct",
            "GET  /health": "Service health check",
            "GET  /metrics": "Aggregate experiment statistics",
            "GET  /runs": "List recent experiment runs"
        }
    }

@router.post("/generate", response_model=GenerateResponse, tags=["Pipeline"])
async def generate(request: Request, body: GenerateRequest):
    """Generate an initial LLM response without verification."""
    pipeline = request.app.state.pipeline
    result = await pipeline._generator.generate(body.query)
    return GenerateResponse(
        response_text=result.response_text,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@router.post("/verify", response_model=VerifyResponse, tags=["Pipeline"])
async def verify(request: Request, body: VerifyRequest):
    """Verify factual claims in arbitrary text. Returns NLI labels per claim."""
    pipeline = request.app.state.pipeline
    report = await pipeline.verify_only(body.text)
    return VerifyResponse(
        metrics=_metrics_from_report(report),
        claim_details=_claim_details_from_report(report),
        is_clean=report.is_clean,
    )


@router.post("/critique", response_model=CritiqueResponse, tags=["Pipeline"])
async def critique(request: Request, body: CritiqueRequest):
    """Verify text and generate targeted correction instructions for unsupported claims."""
    pipeline = request.app.state.pipeline

    report = await pipeline.verify_only(body.text)
    critique_result = await pipeline._critic.critique(report)

    return CritiqueResponse(
        metrics=_metrics_from_report(report),
        critiques=_critique_details(critique_result),
        correction_block=critique_result.to_correction_block(),
    )


@router.post("/correct", response_model=CorrectResponse, tags=["Pipeline"])
async def correct(request: Request, body: CorrectRequest):
    """
    Verify, critique, and self-correct existing text.
    Skips the generation step — useful when you already have text to verify.
    """
    pipeline = request.app.state.pipeline

    # Verify original
    initial_report = await pipeline.verify_only(body.text)
    critique_result = await pipeline._critic.critique(initial_report)

    # Self-correct
    correction = await pipeline._corrector.correct(
        original_response=body.text,
        critique_result=critique_result,
        initial_report=initial_report,
        reverify_fn=pipeline._reverify,
    )

    final_report = await pipeline.verify_only(correction.best_response)

    return CorrectResponse(
        original_text=body.text,
        corrected_text=correction.best_response,
        initial_metrics=_metrics_from_report(initial_report),
        final_metrics=_metrics_from_report(final_report),
        iterations_run=correction.iterations_run,
        total_improvement=correction.total_improvement,
        stop_reason=correction.stop_reason,
    )


@router.post("/pipeline", response_model=PipelineResponse, tags=["Pipeline"])
async def pipeline_endpoint(request: Request, body: PipelineRequest):
    """
    Run the complete Veritas pipeline end-to-end.
    Generate → Extract → Retrieve → Verify → Critique → Correct → Re-Verify
    """
    pipeline = request.app.state.pipeline
    exp_logger = request.app.state.exp_logger

    result = await pipeline.run(body.query)
    run_id = await exp_logger.log(result)

    return PipelineResponse(
        run_id=run_id,
        query=result.query,
        initial_response=result.initial_response,
        final_response=result.final_response,
        claims=result.claims,
        claim_parse_error=result.claim_parse_error,
        initial_metrics=_metrics_from_report(result.initial_report),
        final_metrics=_metrics_from_report(result.final_report),
        claim_details=_claim_details_from_report(result.initial_report),
        critiques=_critique_details(result.critique),
        correction={
            "iterations_run": result.correction.iterations_run,
            "initial_score": result.correction.initial_score,
            "best_score": result.correction.best_score,
            "total_improvement": result.correction.total_improvement,
            "stop_reason": result.correction.stop_reason,
        },
        total_latency_ms=result.total_latency_ms,
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    """Check service health and provider availability."""
    llm_provider = request.app.state.llm_provider
    embedding_provider = request.app.state.embedding_provider
    corpus_index = request.app.state.corpus_index

    llm_ok = await llm_provider.health_check()
    emb_ok = await embedding_provider.health_check()

    return HealthResponse(
        status="ok" if (llm_ok and emb_ok and corpus_index.is_loaded) else "degraded",
        llm_provider=llm_provider.model_name,
        llm_healthy=llm_ok,
        embedding_healthy=emb_ok,
        corpus_loaded=corpus_index.is_loaded,
        corpus_size=corpus_index.size,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics(request: Request):
    """Return aggregate statistics across all experiment runs."""
    exp_logger = request.app.state.exp_logger
    data = exp_logger.aggregate_metrics()
    return MetricsResponse(**data)


@router.get("/runs", tags=["System"])
async def list_runs(request: Request, limit: int = 20):
    """List recent experiment runs (most recent first)."""
    exp_logger = request.app.state.exp_logger
    return {"runs": exp_logger.list_runs(limit=limit)}


@router.get("/runs/{run_id}", tags=["System"])
async def get_run(request: Request, run_id: str):
    """Retrieve the full record for a specific experiment run."""
    exp_logger = request.app.state.exp_logger
    record = exp_logger.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return record
