"""
core/pipeline/verifier.py  —  Step 4: NLI verification.

For each (claim, evidence) pair, runs Natural Language Inference to
determine whether the evidence supports, contradicts, or is neutral
towards the claim.

Key V1 decision:
    If the model's max confidence is below nli_confidence_threshold,
    the label is overridden to "uncertain" rather than forcing a
    low-confidence entailment/contradiction.

Model:
    cross-encoder/nli-deberta-v3-small
    — Purpose-built NLI cross-encoder
    — Takes (premise, hypothesis) pairs
    — Outputs logits for [contradiction, entailment, neutral]
    — Faster than full DeBERTa-v3-large, sufficient for V1
"""

import asyncio
import logging
import time
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
    ENTAILMENT = "entailment"       # evidence supports the claim
    NEUTRAL = "neutral"             # evidence neither supports nor contradicts
    CONTRADICTION = "contradiction" # evidence contradicts the claim
    UNCERTAIN = "uncertain"         # max confidence below threshold
    NO_EVIDENCE = "no_evidence"     # no evidence chunks were retrieved


@dataclass
class ClaimVerification:
    """NLI result for a single claim."""
    claim: str
    label: NLILabel
    confidence: float               # probability of the winning label [0, 1]
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    evidence_used: str | None       # text of the best-scoring evidence chunk
    evidence_title: str | None      # source title for logging
    evidence_similarity: float      # retrieval cosine similarity


@dataclass
class VerifierResult:
    verifications: list[ClaimVerification]

    # Convenience counts
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
        """Claims that need correction: contradicted + neutral + uncertain."""
        return [
            v for v in self.verifications
            if v.label in (NLILabel.CONTRADICTION, NLILabel.NEUTRAL, NLILabel.UNCERTAIN)
        ]


class NLIVerifier:
    """
    Runs DeBERTa-v3 NLI on (claim, evidence) pairs.

    The model is loaded lazily on first call. For V1 this is fine
    since FastAPI startup triggers the first warm-up call.

    Evidence aggregation strategy (V1):
        Run NLI against the top-1 evidence chunk only.
        If we have multiple chunks, pick the one with highest
        retrieval similarity as the "best evidence."
        V2 extension: aggregate NLI across all top-k chunks.
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
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return
        logger.info("Loading NLI model: %s on %s", self._model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

        # Map model label indices to NLI label names
        # DeBERTa-v3 NLI label order: {0: contradiction, 1: entailment, 2: neutral}
        # Always read from model config to handle variations
        id2label = self._model.config.id2label
        self._label_map = {idx: label.lower() for idx, label in id2label.items()}
        logger.info("NLI model loaded. Label map: %s", self._label_map)

    async def verify(self, claim_evidences: list[ClaimEvidence]) -> VerifierResult:
        """
        Verify all claims against their retrieved evidence.

        Args:
            claim_evidences: Output from the EvidenceRetriever.

        Returns:
            VerifierResult with one ClaimVerification per claim.
        """
        loop = asyncio.get_event_loop()
        verifications = await loop.run_in_executor(
            None, self._verify_sync, claim_evidences
        )
        return VerifierResult(verifications=verifications)

    def _verify_sync(self, claim_evidences: list[ClaimEvidence]) -> list[ClaimVerification]:
        """Synchronous NLI inference — runs in thread pool."""
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

            # Use the highest-similarity evidence chunk as premise
            best_evidence = ce.evidence[0]
            premise = best_evidence.text
            hypothesis = ce.claim

            scores = self._run_nli(premise, hypothesis)
            label, confidence = self._decide_label(scores)

            results.append(ClaimVerification(
                claim=ce.claim,
                label=label,
                confidence=confidence,
                entailment_score=scores.get("entailment", 0.0),
                neutral_score=scores.get("neutral", 0.0),
                contradiction_score=scores.get("contradiction", 0.0),
                evidence_used=premise,
                evidence_title=best_evidence.title,
                evidence_similarity=best_evidence.similarity,
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

    def _run_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """
        Run one (premise, hypothesis) pair through the NLI model.

        Returns a dict mapping label name → softmax probability.
        """
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

        # Map index → label name → probability
        return {self._label_map[i]: float(p) for i, p in enumerate(probs)}

    def _decide_label(self, scores: dict[str, float]) -> tuple[NLILabel, float]:
        """
        Choose the final NLI label with confidence threshold.

        If the winning label's probability is below nli_confidence_threshold,
        return UNCERTAIN instead of forcing a low-confidence label.
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
        label = label_map.get(best_label_str, NLILabel.UNCERTAIN)
        return label, best_score
