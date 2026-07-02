"""
core/pipeline/verifier.py  —  Step 4: NLI verification.

For each (claim, evidence) pair, runs Natural Language Inference to
determine whether the evidence supports, contradicts, or is neutral
towards the claim.

Key V1 decisions:
    - Confidence threshold: below nli_confidence_threshold → "uncertain"
    - Evidence aggregation: run NLI against all top-k chunks, not just top-1
    - Aggregation priority: strong contradiction (>=0.80) > strong entailment
      (>=0.80) > highest confidence overall
    - Similarity floor: chunks below 0.50 cosine similarity are skipped
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.pipeline.retriever import ClaimEvidence
from config import settings

logger = logging.getLogger(__name__)


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"
    UNCERTAIN = "uncertain"
    NO_EVIDENCE = "no_evidence"


@dataclass
class ClaimVerification:
    """NLI result for a single claim."""
    claim: str
    label: NLILabel
    confidence: float
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    evidence_used: str | None
    evidence_title: str | None
    evidence_similarity: float


@dataclass
class VerifierResult:
    verifications: list[ClaimVerification]

    @property
    def total(self) -> int:
        return len(self.verifications)

    @property
    def entailed(self) -> list[ClaimVerification]:
        return [v for v in self.verifications if v.label == NLILabel.ENTAILMENT]

    @property
    def contradicted(self) -> list[ClaimVerification]:
        return [v for v in self.verifications if v.label == NLILabel.CONTRADICTION]

    @property
    def neutral(self) -> list[ClaimVerification]:
        return [v for v in self.verifications if v.label == NLILabel.NEUTRAL]

    @property
    def uncertain(self) -> list[ClaimVerification]:
        return [v for v in self.verifications if v.label == NLILabel.UNCERTAIN]

    @property
    def no_evidence(self) -> list[ClaimVerification]:
        return [v for v in self.verifications if v.label == NLILabel.NO_EVIDENCE]

    @property
    def unsupported(self) -> list[ClaimVerification]:
        return [
            v for v in self.verifications
            if v.label in (NLILabel.CONTRADICTION, NLILabel.NEUTRAL, NLILabel.UNCERTAIN)
        ]


class NLIVerifier:
    """
    Runs DeBERTa-v3 NLI on (claim, evidence) pairs.

    Evidence aggregation: runs NLI against all top-k chunks above the
    similarity floor, then picks the most meaningful result using a
    confidence-weighted priority system.
    """

    def __init__(
        self,
        model_name: str = settings.nli_model,
        confidence_threshold: float = settings.nli_confidence_threshold,
        batch_size: int = settings.nli_batch_size,
    ):
        self._model_name = model_name
        self._threshold = confidence_threshold
        self._batch_size = batch_size
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer = None
        self._label_map: dict[int, str] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading NLI model: %s on %s", self._model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        id2label = self._model.config.id2label
        self._label_map = {idx: label.lower() for idx, label in id2label.items()}
        logger.info("NLI model loaded. Label map: %s", self._label_map)

    async def verify(self, claim_evidences: list[ClaimEvidence]) -> VerifierResult:
        loop = asyncio.get_event_loop()
        verifications = await loop.run_in_executor(
            None, self._verify_sync, claim_evidences
        )
        return VerifierResult(verifications=verifications)

    def _verify_sync(self, claim_evidences: list[ClaimEvidence]) -> list[ClaimVerification]:
        self._load()
        results: list[ClaimVerification] = []

        for ce in claim_evidences:
            if not ce.has_evidence:
                results.append(ClaimVerification(
                    claim=ce.claim,
                    label=NLILabel.NO_EVIDENCE,
                    confidence=0.0,
                    entailment_score=0.0,
                    neutral_score=0.0,
                    contradiction_score=0.0,
                    evidence_used=None,
                    evidence_title=None,
                    evidence_similarity=0.0,
                ))
                continue

            # Filter to chunks above similarity floor
            SIMILARITY_FLOOR = 0.50
            valid_evidence = [e for e in ce.evidence if e.similarity >= SIMILARITY_FLOOR]

            if not valid_evidence:
                results.append(ClaimVerification(
                    claim=ce.claim,
                    label=NLILabel.NO_EVIDENCE,
                    confidence=0.0,
                    entailment_score=0.0,
                    neutral_score=0.0,
                    contradiction_score=0.0,
                    evidence_used=None,
                    evidence_title=None,
                    evidence_similarity=ce.evidence[0].similarity if ce.evidence else 0.0,
                ))
                logger.debug(
                    "Claim skipped — no evidence above similarity floor %.2f: %.60s",
                    SIMILARITY_FLOOR, ce.claim,
                )
                continue

            # Run NLI against every valid chunk
            chunk_results = []
            for ev_chunk in valid_evidence:
                chunk_scores = self._run_nli(ev_chunk.text, ce.claim)
                chunk_results.append((chunk_scores, ev_chunk))

            # Aggregate across chunks
            best_scores, best_chunk = self._aggregate_nli_results(chunk_results)
            label, confidence = self._decide_label(best_scores)

            results.append(ClaimVerification(
                claim=ce.claim,
                label=label,
                confidence=confidence,
                entailment_score=best_scores.get("entailment", 0.0),
                neutral_score=best_scores.get("neutral", 0.0),
                contradiction_score=best_scores.get("contradiction", 0.0),
                evidence_used=best_chunk.text,
                evidence_title=best_chunk.title,
                evidence_similarity=best_chunk.similarity,
            ))

        logger.info(
            "NLI complete: %d entailed, %d contradicted, %d neutral, "
            "%d uncertain, %d no_evidence",
            sum(1 for r in results if r.label == NLILabel.ENTAILMENT),
            sum(1 for r in results if r.label == NLILabel.CONTRADICTION),
            sum(1 for r in results if r.label == NLILabel.NEUTRAL),
            sum(1 for r in results if r.label == NLILabel.UNCERTAIN),
            sum(1 for r in results if r.label == NLILabel.NO_EVIDENCE),
        )
        return results

    def _aggregate_nli_results(
        self,
        chunk_results: list[tuple[dict, object]],
    ) -> tuple[dict, object]:
        """
        Select the most meaningful NLI result from multiple (scores, chunk) pairs.

        Logic:
        - Strong contradiction (>= 0.80) wins first.
        - Strong entailment (>= 0.80) wins second.
        - Otherwise return highest confidence overall (likely neutral).

        This prevents weak contradictions from loosely-related chunks
        overriding strong entailments from highly relevant chunks.
        """
        STRONG_THRESHOLD = 0.80

        best_overall_scores, best_overall_chunk, best_overall_conf = None, None, -1.0
        best_contra_scores, best_contra_chunk, best_contra_conf = None, None, -1.0
        best_entail_scores, best_entail_chunk, best_entail_conf = None, None, -1.0

        for scores, chunk in chunk_results:
            if not scores:
                continue
            top_label = max(scores, key=scores.__getitem__)
            top_conf = scores[top_label]

            if top_conf > best_overall_conf:
                best_overall_conf = top_conf
                best_overall_scores = scores
                best_overall_chunk = chunk

            if top_label == "contradiction" and top_conf > best_contra_conf:
                best_contra_conf = top_conf
                best_contra_scores = scores
                best_contra_chunk = chunk

            if top_label == "entailment" and top_conf > best_entail_conf:
                best_entail_conf = top_conf
                best_entail_scores = scores
                best_entail_chunk = chunk

        if best_contra_conf >= STRONG_THRESHOLD:
            return best_contra_scores, best_contra_chunk

        if best_entail_conf >= STRONG_THRESHOLD:
            return best_entail_scores, best_entail_chunk

        if best_overall_scores is None:
            best_overall_scores, best_overall_chunk = chunk_results[0]
        return best_overall_scores, best_overall_chunk

    def _run_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
        return {self._label_map[i]: float(p) for i, p in enumerate(probs)}

    def _decide_label(self, scores: dict[str, float]) -> tuple[NLILabel, float]:
        """
        Apply confidence threshold. Below threshold → UNCERTAIN.
        """
        if not scores:
            return NLILabel.UNCERTAIN, 0.0

        best_label_str = max(scores, key=scores.__getitem__)
        best_score = scores[best_label_str]

        if best_score < self._threshold:
            return NLILabel.UNCERTAIN, best_score

        label_map = {
            "entailment": NLILabel.ENTAILMENT,
            "neutral": NLILabel.NEUTRAL,
            "contradiction": NLILabel.CONTRADICTION,
        }
        return label_map.get(best_label_str, NLILabel.UNCERTAIN), best_score