"""
api/schemas.py  —  Pydantic request and response models.

Explicit schemas for every endpoint ensure:
- Input validation with clear error messages
- Consistent response shape across all endpoints
- Auto-generated OpenAPI docs that accurately reflect the API
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator


# ── Request models ────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question or instruction")


class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to verify for hallucinations")


class CritiqueRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to critique")


class CorrectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to correct")


class PipelineRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")


# ── Shared sub-models ─────────────────────────────────────────────────────────

class HallucinationMetrics(BaseModel):
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    neutral_claims: int
    uncertain_claims: int
    no_evidence_claims: int
    hallucination_score: float = Field(..., ge=0.0, le=1.0)
    avg_nli_confidence: float
    avg_retrieval_similarity: float


class ClaimDetail(BaseModel):
    claim: str
    label: str
    confidence: float
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    evidence_title: str | None
    evidence_similarity: float


class CritiqueDetail(BaseModel):
    claim: str
    label: str
    instruction: str
    evidence_source: str | None


class CorrectionIteration(BaseModel):
    iteration: int
    hallucination_score: float
    improvement: float
    was_best: bool


# ── Response models ───────────────────────────────────────────────────────────

class GenerateResponse(BaseModel):
    response_text: str
    model: str
    latency_ms: float | None


class VerifyResponse(BaseModel):
    metrics: HallucinationMetrics
    claim_details: list[ClaimDetail]
    is_clean: bool


class CritiqueResponse(BaseModel):
    metrics: HallucinationMetrics
    critiques: list[CritiqueDetail]
    correction_block: str


class CorrectResponse(BaseModel):
    original_text: str
    corrected_text: str
    initial_metrics: HallucinationMetrics
    final_metrics: HallucinationMetrics
    iterations_run: int
    total_improvement: float
    stop_reason: str


class PipelineResponse(BaseModel):
    run_id: str
    query: str
    initial_response: str
    final_response: str
    claims: list[str]
    claim_parse_error: str | None
    initial_metrics: HallucinationMetrics
    final_metrics: HallucinationMetrics
    claim_details: list[ClaimDetail]
    critiques: list[CritiqueDetail]
    correction: dict[str, Any]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_healthy: bool
    embedding_healthy: bool
    corpus_loaded: bool
    corpus_size: int


class MetricsResponse(BaseModel):
    total_runs: int
    avg_initial_hallucination_score: float = 0.0
    avg_final_hallucination_score: float = 0.0
    avg_improvement: float = 0.0
    avg_correction_iterations: float = 0.0
    avg_latency_ms: float = 0.0
    correction_success_rate: float = 0.0
    stop_reason_distribution: dict[str, int] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
