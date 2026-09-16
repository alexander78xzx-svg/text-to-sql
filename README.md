# Execution-Guided Text-to-SQL Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python)](https://python.org)

An asynchronous microservice that translates natural language questions into executable SQL using a fine-tuned 117M parameter local transformer (GPT-2 base). The pipeline implements an execution-guided candidate selection layer using an ephemeral SQLite sandbox to filter out hallucinated schemas, malformed syntax, and broken relational joins.

---
## Live Demo & Web Interface

The system is deployed and accessible via the production web application:

* **Live Interactive Demo:** [https://text-to-sql-web-chi.vercel.app](https://text-to-sql-web-chi.vercel.app)
* **Frontend Source Code:** [Text-to-SQL Web Client](https://github.com/alexander78xzx-svg/text-to-sql-web)
---

## Architectural Motivation & Model Selection

* **Resource-Constrained Design:** The 117M parameter base was deliberately selected to optimize for constrained edge or local CPU environments, maintaining sub-100ms response times and an operational RAM footprint under 1.2 GB.
* **Model Agnostic Pipeline:** The underlying execution-guided validation architecture is fully model-agnostic. While evaluated here on a lightweight transformer, the same candidate generation, DDL injection, and `EXPLAIN QUERY PLAN` verification engine directly scales to larger foundation models (e.g., CodeLlama 7B/13B, Mistral, or deep-seek variants) to achieve higher Exact Match rates without architectural changes.

---

## Core Architecture

1. **Ingress Layer (FastAPI):** Ingests relational schema DDL, foreign key maps, and natural language queries. CPU-bound inference and database operations are offloaded to worker threads via `asyncio.to_thread` to maintain a non-blocking event loop.
2. **Inference Pipeline (PyTorch):** Generates candidate query sequences using sampled autoregressive decoding ($T=0.4$) constrained by schema prompts.
3. **Execution Sandbox (SQLite):** Dynamically hydrates an in-memory (`:memory:`) database with caller-defined tables, primary keys, and foreign keys. It dry-runs candidate sequences through `EXPLAIN QUERY PLAN` to deterministically discard structurally invalid SQL before returning a payload.

---

## Empirical Evaluation & Benchmarks

The model was evaluated against a 50-query holdout validation set across complex multi-table schemas:

| Metric | Measurement | Description |
| :--- | :--- | :--- |
| **Model Parameters** | 117 Million | Local GPT-2 architecture fine-tuned on relational schemas |
| **Exact Match (EM)** | **36.0%** (18/50) | Character-level identical match against gold queries |
| **Syntactic Validity** | **66.0%** (33/50) | Queries that successfully compile against target schema DDL |

---

## Reproducing the Weights

Model weights (`.pt` files) are excluded from the repository. To regenerate them locally:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Tokenize dataset and structural schemas
python dataset.py

# 2. Fine-tune the local weights
python train.py

# 3. Run validation benchmark
python validate.py
```

---

## API Contract

**Endpoint:** `POST /v1/generate`

**Request Payload:**
```json
{
  "question": "Find the names of active customers.",
  "tables": [
    {
      "name": "customers",
      "columns": ["id", "name", "status"],
      "primary_keys": ["id"]
    }
  ],
  "foreign_keys": []
}
```

**Response (`200 OK`):**
```json
{
  "output": "SELECT name FROM customers WHERE status = 'active';",
  "compiles": true
}
```

---

## Local Development & Docker

```bash
# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run via Docker
docker build -t text-to-sql-api .
docker run -p 8000:8000 text-to-sql-api
```
