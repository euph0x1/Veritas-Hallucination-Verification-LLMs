"""
logging_module/logger.py  —  Lightweight JSON experiment logger.

V1 decision: log to structured JSON files, one file per experiment run.
MLflow / PostgreSQL deferred to V2.

Log files are written to settings.log_dir/{run_id}.json.
An index file (index.jsonl) appends one line per run for fast listing.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def _serialise(obj):
    """JSON serialiser for types not handled by default encoder."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):          # Enum
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return str(obj)


class ExperimentLogger:
    """
    Writes pipeline results to structured JSON files.

    Each run gets a unique ID and a dedicated JSON file.
    A lightweight JSONL index allows listing past runs without
    reading every file.

    Usage:
        logger = ExperimentLogger()
        run_id = await logger.log(pipeline_result)
    """

    def __init__(self, log_dir: Path = settings.log_dir, enabled: bool = settings.log_enabled):
        self._log_dir = log_dir
        self._enabled = enabled
        if enabled:
            log_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = log_dir / "index.jsonl"

    async def log(self, pipeline_result) -> str:
        """
        Persist a full pipeline result to disk.

        Args:
            pipeline_result: PipelineResult from VeritasPipeline.run().

        Returns:
            run_id string (UUID4).
        """
        if not self._enabled:
            return "logging_disabled"

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = self._build_record(run_id, timestamp, pipeline_result)

        # ── Write full run file ───────────────────────────────────────────────
        run_path = self._log_dir / f"{run_id}.json"
        try:
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(record, f, default=_serialise, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error("Failed to write experiment log %s: %s", run_path, e)
            return run_id

        # ── Append to index ───────────────────────────────────────────────────
        index_entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": pipeline_result.query[:120],
            "initial_score": pipeline_result.initial_report.hallucination_score,
            "final_score": pipeline_result.final_report.hallucination_score,
            "improvement": pipeline_result.correction.total_improvement,
            "iterations": pipeline_result.correction.iterations_run,
            "stop_reason": pipeline_result.correction.stop_reason,
            "total_latency_ms": round(pipeline_result.total_latency_ms, 1),
        }
        try:
            with open(self._index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(index_entry) + "\n")
        except IOError as e:
            logger.warning("Failed to update experiment index: %s", e)

        logger.info("Experiment logged: %s → %s", run_id, run_path)
        return run_id

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return the most recent `limit` run summaries from the index."""
        if not self._index_path.exists():
            return []
        try:
            lines = self._index_path.read_text(encoding="utf-8").strip().splitlines()
            runs = [json.loads(line) for line in lines if line.strip()]
            return runs[-limit:][::-1]      # most recent first
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Failed to read experiment index: %s", e)
            return []

    def get_run(self, run_id: str) -> dict | None:
        """Load the full record for a specific run."""
        run_path = self._log_dir / f"{run_id}.json"
        if not run_path.exists():
            return None
        try:
            with open(run_path, encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Failed to read run %s: %s", run_id, e)
            return None

    def aggregate_metrics(self) -> dict:
        """
        Compute aggregate statistics across all logged runs.
        Used by the GET /metrics endpoint.
        """
        runs = self.list_runs(limit=1000)
        if not runs:
            return {"total_runs": 0}

        initial_scores = [r["initial_score"] for r in runs]
        final_scores   = [r["final_score"] for r in runs]
        improvements   = [r["improvement"] for r in runs]
        iterations     = [r["iterations"] for r in runs]
        latencies      = [r["total_latency_ms"] for r in runs]

        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

        stop_reasons = {}
        for r in runs:
            stop_reasons[r["stop_reason"]] = stop_reasons.get(r["stop_reason"], 0) + 1

        return {
            "total_runs": len(runs),
            "avg_initial_hallucination_score": avg(initial_scores),
            "avg_final_hallucination_score": avg(final_scores),
            "avg_improvement": avg(improvements),
            "avg_correction_iterations": avg(iterations),
            "avg_latency_ms": avg(latencies),
            "correction_success_rate": round(
                sum(1 for r in runs if r["improvement"] > 0) / len(runs), 4
            ),
            "stop_reason_distribution": stop_reasons,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_record(self, run_id: str, timestamp: str, result) -> dict:
        """Build the full JSON record for a pipeline run."""

        # Per-claim detail
        claim_details = []
        for v in result.initial_report.verifications:
            claim_details.append({
                "claim": v.claim,
                "label": v.label.value,
                "confidence": round(v.confidence, 4),
                "entailment_score": round(v.entailment_score, 4),
                "neutral_score": round(v.neutral_score, 4),
                "contradiction_score": round(v.contradiction_score, 4),
                "evidence_title": v.evidence_title,
                "evidence_similarity": round(v.evidence_similarity, 4),
            })

        critique_details = []
        for c in result.critique.critiques:
            critique_details.append({
                "claim": c.claim,
                "label": c.label.value,
                "instruction": c.instruction,
                "evidence_source": c.evidence_source,
            })

        iteration_history = []
        for it in result.correction.iteration_history:
            iteration_history.append({
                "iteration": it.iteration,
                "hallucination_score": it.hallucination_score,
                "improvement": round(it.improvement, 4),
                "was_best": it.was_best,
            })

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": result.query,
            "initial_response": result.initial_response,
            "final_response": result.final_response,
            "claims": result.claims,
            "claim_parse_error": result.claim_parse_error,
            "initial_metrics": result.initial_report.to_dict(),
            "final_metrics": result.final_report.to_dict(),
            "claim_details": claim_details,
            "critiques": critique_details,
            "correction": {
                "iterations_run": result.correction.iterations_run,
                "initial_score": result.correction.initial_score,
                "best_score": result.correction.best_score,
                "total_improvement": result.correction.total_improvement,
                "stop_reason": result.correction.stop_reason,
                "iteration_history": iteration_history,
            },
            "total_latency_ms": round(result.total_latency_ms, 1),
        }
