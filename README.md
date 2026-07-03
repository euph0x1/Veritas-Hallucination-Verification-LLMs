# Veritas

**Evidence-Guided Hallucination Verification & Self-Correction Framework**

Veritas detects unsupported factual claims in LLM responses, retrieves supporting evidence, verifies consistency with NLI, generates targeted critiques, and iteratively self-corrects the final response.

> For complete endpoint testing instructions, sample requests, and troubleshooting, see `TESTING_GUIDE.md`.

---

## Project Overview

Veritas is a research-first API for hallucination detection and correction. It connects an Ollama LLM with a FAISS-powered Wikipedia index and a DeBERTa NLI model to verify factual claims and improve responses before returning them.

## Motivation

Large language models are fluent but often hallucinate facts. Veritas is designed to catch those hallucinations with evidence retrieval and NLI verification, then guide the model toward grounded corrections.

## Architecture

```
User Query
    │
    ▼
[1] Response Generator      (Ollama LLM)
    │
    ▼
[2] Claim Extractor         (LLM → atomic JSON claims)
    │
    ▼
[3] Evidence Retriever      (FAISS top-k cosine search)
    │
    ▼
[4] NLI Verifier            (DeBERTa-v3: entailment / neutral / contradiction / uncertain)
    │
    ▼
[5] Hallucination Scorer    (6 metrics including hallucination_score)
    │
    ▼
[6] Critique Generator      (claim-level, evidence-grounded instructions)
    │
    ▼
[7] Self-Correction Loop    (max 3 iterations, early exit, best-seen tracking)
    │
    ▼
Final Response + Full Report
```

---

## Prerequisites

- Python 3.12+
- FastAPI
- Uvicorn
- Ollama for LLM inference
- SentenceTransformers for embeddings
- FAISS for retrieval
- HuggingFace NLI model: `cross-encoder/nli-deberta-v3-small`
- Pydantic settings and schemas

## Installation

```bash
git clone <repo-url>
cd veritas
python -m venv .venv

.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

cp .env.example .env
```

### One-time setup

Pull the Ollama model and build the FAISS index once before running the API:

```bash
ollama pull gemma3

python -m corpus.build_index --max-articles 500
```

## Quick Start

Start the project in two terminals:

```bash
# Terminal 1
ollama serve

# Terminal 2
python -m uvicorn api.main:app --reload --port 8000
```

Then open the main interactive documentation:

```text
http://127.0.0.1:8000/docs
```

The FAISS index only needs to be rebuilt if you change the corpus, embedding model, or chunking strategy.

## Useful URLs

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/` | API landing page with available endpoints |
| `http://127.0.0.1:8000/docs` | Swagger UI for interactive testing |
| `http://127.0.0.1:8000/redoc` | Read-only API documentation |
| `http://127.0.0.1:8000/health` | Verify Ollama, embeddings, and corpus |
| `http://127.0.0.1:8000/metrics` | Aggregate experiment statistics |
| `http://127.0.0.1:8000/runs` | List previous pipeline executions |
| `http://127.0.0.1:8000/runs/{run_id}` | View a specific run record |

## API Overview

The main endpoints are:

- `POST /pipeline` — complete generate + verify + critique + correct workflow
- `POST /generate` — raw generation only
- `POST /verify` — verify claims in provided text
- `POST /critique` — generate correction instructions
- `POST /correct` — self-correct existing text
- `GET /health` — system health check
- `GET /metrics` — experiment metrics
- `GET /runs` / `GET /runs/{run_id}` — saved run history

## Configuration

Copy `.env.example` to `.env` and update values as needed.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3` | Ollama model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model |
| `RETRIEVAL_TOP_K` | `5` | Evidence chunks per claim |
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | NLI verification model |
| `NLI_CONFIDENCE_THRESHOLD` | `0.65` | Confidence threshold for uncertain labels |
| `MAX_CORRECTION_ITERATIONS` | `3` | Max self-correction rounds |
| `IMPROVEMENT_THRESHOLD` | `0.05` | Minimum improvement to continue iterations |
| `LOG_DIR` | `data/logs` | Experiment log directory |
| `CORPUS_MAX_ARTICLES` | `500` | Number of Wikipedia articles to index |

## Project Structure

```
veritas/
├── config.py                  # All settings (Pydantic)
├── core/
│   ├── providers/             # LLM + embedding abstractions
│   │   ├── base.py            # Interfaces
│   │   └── ollama.py          # OllamaProvider, SentenceTransformerProvider
│   ├── pipeline/              # One module per pipeline step
│   │   ├── generator.py       # Step 1
│   │   ├── extractor.py       # Step 2
│   │   ├── retriever.py       # Step 3
│   │   ├── verifier.py        # Step 4
│   │   ├── scorer.py          # Step 5
│   │   ├── critic.py          # Step 6
│   │   └── corrector.py       # Step 7
│   └── orchestrator.py        # Full pipeline coordinator
├── corpus/
│   ├── build_index.py         # Offline: Wikipedia → FAISS (run once)
│   └── loader.py              # Online: load index for retrieval
├── api/
│   ├── main.py                # FastAPI app factory + lifespan
│   ├── routes.py              # All endpoint handlers
│   └── schemas.py             # Pydantic request/response models
├── logging_module/
│   └── logger.py              # JSON experiment logger
├── tests/
│   └── unit/
│       └── test_pipeline.py
├── docker/
│   ├── Dockerfile
│   └── compose.yml
├── data/
│   ├── index/                 # FAISS index (gitignored)
│   └── logs/                  # Experiment JSON logs (gitignored)
└── requirements.txt
```

---

## V2 Extensions (Future Work)

- PostgreSQL / MLflow experiment tracking
- Cross-encoder reranking of retrieved evidence
- Claim type classifier (factual / opinion / procedural)
- Re-verification after each correction iteration
- Hybrid BM25 + dense retrieval
- Multiple NLI model comparison
- Additional provider implementations (OpenAI / Anthropic)
- Authentication and rate limiting
- Batch evaluation with TruthfulQA / HaluEval
