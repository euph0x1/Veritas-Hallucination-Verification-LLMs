# Veritas Testing Guide

This guide contains the full testing instructions for Veritas, including endpoint workflows, example JSON requests, expected responses, sample prompts, and troubleshooting.

## Startup Checklist

1. Clone the repository and install dependencies.
2. Copy `.env.example` to `.env`.
3. Pull the Ollama model once:

```bash
ollama pull gemma3
```

4. Build the FAISS corpus index once:

```bash
python -m corpus.build_index --max-articles 100
```

5. Start Ollama in a separate terminal:

```bash
ollama serve
```

6. Start the API server:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

## Which URLs to Open

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/` | Veritas home page with API links |
| `http://127.0.0.1:8000/docs` | Swagger UI — primary interface for testing |
| `http://127.0.0.1:8000/redoc` | Read-only API docs |
| `http://127.0.0.1:8000/health` | Verify service and provider health |
| `http://127.0.0.1:8000/metrics` | View aggregated experiment metrics |
| `http://127.0.0.1:8000/runs` | List previous pipeline executions |
| `http://127.0.0.1:8000/runs/{run_id}` | View details for a specific execution |

> Use `/docs` as your main testing interface. Click any endpoint, then click `Try it out`.

## How to Use Swagger UI

1. Open `http://127.0.0.1:8000/docs` in your browser.
2. Find the endpoint you want to test.
3. Click `Try it out`.
4. Paste the example JSON into the request body.
5. Click `Execute`.
6. Review the response body and status code.

## Recommended Testing Order

1. `GET /health`
2. `POST /generate`
3. `POST /verify`
4. `POST /critique`
5. `POST /correct`
6. `POST /pipeline`
7. `GET /runs`
8. `GET /runs/{run_id}`
9. `GET /metrics`

This order is recommended because it verifies the foundation first, then builds confidence step-by-step toward the full end-to-end pipeline.

## Endpoint Testing Instructions

### GET /health

Purpose: Verify that Ollama, embeddings, and the FAISS corpus are loaded successfully.

Expected response body:

```json
{
  "status": "ok",
  "llm_provider": "ollama/gemma3",
  "llm_healthy": true,
  "embedding_healthy": true,
  "corpus_loaded": true,
  "corpus_size": 3293
}
```

Success looks like: all health flags are `true` and the status is `ok`.

### POST /generate

Purpose: Generate a raw LLM response without verification.

Example request:

```json
{
  "query": "Who invented the telephone and when was it first used?"
}
```

Expected response fields:

- `response_text`
- `model`
- `latency_ms`

What success looks like: a coherent response in `response_text`.

### POST /verify

Purpose: Extract claims from a text passage and verify them using NLI.

Example request:

```json
{
  "text": "Alexander Graham Bell invented the telephone in 1876. He was born in Edinburgh in 1847. He also invented the television."
}
```

Expected response fields:

- `metrics`
- `claim_details`
- `is_clean`

What success looks like: each claim is labeled, and `hallucination_score` is present.

### POST /critique

Purpose: Verify text and generate evidence-grounded correction instructions.

Example request:

```json
{
  "text": "Albert Einstein won the Nobel Prize in Physics in 1921 for his theory of relativity. He was born in Berlin in 1879."
}
```

Expected response fields:

- `metrics`
- `critiques`
- `correction_block`

What success looks like: critiques contain specific instructions tied to evidence.

### POST /correct

Purpose: Verify, critique, and self-correct existing text.

Example request:

```json
{
  "text": "The Eiffel Tower was built in 1901 and stands 250 metres tall. It is located in Berlin."
}
```

Expected response fields:

- `original_text`
- `corrected_text`
- `initial_metrics`
- `final_metrics`
- `iterations_run`
- `total_improvement`
- `stop_reason`

What success looks like: `corrected_text` fixes the errors and `final_metrics.hallucination_score` is lower than `initial_metrics.hallucination_score`.

### POST /pipeline

Purpose: Execute the full Veritas pipeline end-to-end.

Example request:

```json
{
  "query": "Tell me about Marie Curie's life and scientific achievements."
}
```

Expected response fields:

- `run_id`
- `query`
- `initial_response`
- `final_response`
- `claims`
- `claim_parse_error`
- `initial_metrics`
- `final_metrics`
- `claim_details`
- `critiques`
- `correction`
- `total_latency_ms`

What success looks like: the pipeline completes, `run_id` is returned, and `final_metrics.hallucination_score` is lower than `initial_metrics.hallucination_score`.

### GET /runs

Purpose: List previous `/pipeline` executions.

Expected response fields:

- `runs`

What success looks like: the response contains one or more run summaries with `run_id` values.

### GET /runs/{run_id}

Purpose: Retrieve a specific pipeline execution record.

How to test:

1. Call `GET /runs` and copy a `run_id`.
2. Use `GET /runs/{run_id}` and paste the copied ID.

Expected response fields:

- `run_id`
- `query`
- `initial_response`
- `final_response`
- `claim_details`
- `critiques`
- `correction`

What success looks like: the record contains details for the selected pipeline run.

### GET /metrics

Purpose: View aggregated statistics across all pipeline runs.

Expected response fields:

- `total_runs`
- `avg_initial_hallucination_score`
- `avg_final_hallucination_score`
- `avg_improvement`
- `avg_correction_iterations`
- `avg_latency_ms`
- `correction_success_rate`
- `stop_reason_distribution`

What success looks like: metrics are populated and `avg_improvement` is positive after running several `/pipeline` calls.

## Example Request JSON

Use these requests in Swagger for manual testing.

### Generate

```json
{
  "query": "Who discovered penicillin?"
}
```

### Verify

```json
{
  "text": "Alexander Fleming discovered penicillin in 1928."
}
```

### Critique

```json
{
  "text": "Albert Einstein invented the internet in 1950."
}
```

### Correct

```json
{
  "text": "The Eiffel Tower was built in 1901 and stands 250 metres tall. It is located in Berlin."
}
```

### Pipeline

```json
{
  "query": "When was the Eiffel Tower built and how tall is it?"
}
```

## Sample Test Prompts

### Factual Prompts

1. "When was the Eiffel Tower built and how tall is it?"
2. "Who discovered penicillin and when was it discovered?"
3. "What is the capital of Australia and why is it not Sydney?"
4. "Explain how quantum computing differs from classical computing."
5. "Who invented the telephone and what year was the first call made?"

### Hallucination-Triggering Prompts

1. "Tell me about the Moon Treaty signed in 2024."
2. "Who won the Nobel Prize in Physics in 2032?"
3. "Explain why Albert Einstein invented the internet."
4. "Describe Apple's headquarters on Mars."
5. "What happened during the Second Moon War?"

These prompts are useful for evaluating Veritas's hallucination detection and correction behavior.

## How to Interpret Hallucination Metrics

- `hallucination_score`: lower is better.
- `supported_claims`: claims confirmed by evidence.
- `contradicted_claims`: claims contradicted by retrieved evidence.
- `neutral_claims`: claims not verifiable from retrieved evidence.
- `uncertain_claims`: low-confidence NLI decisions.
- `no_evidence_claims`: retrieval failed to find support.

A successful run typically shows a lower `final_metrics.hallucination_score` than `initial_metrics.hallucination_score`.

## Troubleshooting

### 404 Not Found on `/`

- If `/` returns 404, ensure `api/routes.py` has a root route and the server is restarted.

### `/docs` not opening

- Check the API terminal for startup errors.
- Ensure the server is running on port 8000.

### Ollama not running

- Start Ollama with:

```bash
ollama serve
```

### Model not found

- Pull the model:

```bash
ollama pull gemma3
```

### Corpus index missing

- Build the FAISS index:

```bash
python -m corpus.build_index --max-articles 100
```

### Slow first request

- The first request may be slow because the embedding and NLI models are loading.

### Empty `/runs`

- `GET /runs` only shows results from `/pipeline` calls.
- Run at least one `/pipeline` request first.

### `GET /favicon.ico` 404 Not Found

- This is normal. Browsers request `/favicon.ico` automatically and the API does not serve a favicon by default.

## Complete Demo Walkthrough

1. Start Ollama and the API server.
2. Open `http://127.0.0.1:8000/docs`.
3. Run `GET /health`.
4. Run `POST /generate` with a factual query.
5. Run `POST /verify` on a short text sample.
6. Run `POST /critique` on a text with a false claim.
7. Run `POST /correct` on a text that needs self-correction.
8. Run `POST /pipeline` for a full end-to-end test.
9. Check `GET /runs`, then `GET /runs/{run_id}`.
10. Review `GET /metrics` to confirm the overall pipeline behavior.
