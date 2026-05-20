# 🔬 Multi-Agent RAG Operations & Deployment Playbook

This document provides a highly detailed, minute step-by-step operational manual for maintaining, testing, and deploying the **Multi-Agent Scientific RAG System**.

---

## 🏗️ 1. Complete System Architecture & Flow

The pipeline executes a highly defensive, multi-stage RAG logic designed to ensure high recall, perfect citation groundings, and error resilience.

```
                  [ USER QUERY ]
                        │
                        ▼
                ┌───────────────┐      [ YES ]
                │  Redis Cache  ├───────────────┐
                │  Fingerprint  │               │
                └───────┬───────┘               │
                        │ [ NO ]                ▼
                        ▼               [ Instant Return ]
             ┌─────────────────────┐       (<2ms Latency)
             │ Query Decomposition │
             │   (Gemini Flash)    │
             └──────────┬──────────┘
                        │ (Sub-Questions Array)
                        ▼
             ┌─────────────────────┐
             │ Parallel Retrieval  │
             │ (Qdrant + Postgres) │
             └──────────┬──────────┘
                        │ (Deduplicated Evidence)
                        ▼
             ┌─────────────────────┐
             │   Reasoning Agent   │
             │ (Gemini 2.5 Flash)  │
             └──────────┬──────────┘
                        │ (If Gemini fails with 429 Rate Limit)
                        ▼
             ┌─────────────────────┐
             │  DeepSeek Fallback  │
             │ (HF Serverless API) │
             └──────────┬──────────┘
                        │ (Raw Answer Draft)
                        ▼
             ┌─────────────────────┐
             │ Grounding Validator │
             │  (Anti-Hallucination)│
             └──────────┬──────────┘
                        │ (Grounding Score audit)
                        ▼
            [ Grounded Answer + Sources ]
```

---

## 🛠️ 2. Step-by-Step Local Setup & Execution

### Step 2.1: Python Package Architecture (Astral `uv`)
We use `uv` for lightning-fast package dependency resolution:
```bash
# 1. Install uv on your system
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the project dependencies (creates .venv and installs all lockfile requirements)
uv sync
```

### Step 2.2: Infrastructure Ingestion (Docker)
Start the supporting services (PostgreSQL, Redis, Qdrant, and Neo4j):
```bash
# Bring up all supporting databases in the background
docker compose up -d
```
*Verify that the containers are active:*
```bash
docker ps
```
*Expected containers running: `rag_api`, `rag_postgres`, `rag_redis`, `rag_qdrant`, and `rag_neo4j`.*

### Step 2.3: Initialize the Schema & DB Migration
Tables must be established in PostgreSQL before starting traffic:
```bash
# Initialize DB tables
uv run python -c "from src.database.init_db import init_db; init_db()"
```

### Step 2.4: Ingest Scientific Data
Populate the vector database (Qdrant) and graph database (Neo4j) with your papers:
```bash
# Ingest PDF and CSV files located in data/ directory
uv run python test_ingest.py
```

### Step 2.5: Run Live CLI Performance Audits
Test the entire pipeline directly in your terminal using the real-time reporting tool:
```bash
uv run python quick_test.py "What is the phase and target of the Alzheimer's trial?"
```

---

## 🚦 3. Live Observability & Operations

Our FastAPI server exposes active health metrics to keep tabs on your system's operational health.

### 3.1 Health Endpoint (`GET /health`)
Used by target groups/load-balancers to check container health. 
* **Command:** `curl http://localhost:8000/health`
* **Checks Executed:**
  * **Postgres**: Executes a native SQL `SELECT 1` query to verify active db pooling.
  * **Redis**: Executes a `.ping()` command to check local caching health.
  * **Qdrant**: Accesses the REST cluster using `.get_collections()`.
  * **LLM**: Validates that active credentials are loaded in memory.
* **Response Format:**
  ```json
  {
    "status": "healthy",
    "postgres": "up",
    "redis": "up",
    "qdrant": "up",
    "llm_provider": "active"
  }
  ```

### 3.2 Metrics Endpoint (`GET /metrics`)
Tracks real-time system performance in memory using Redis.
* **Command:** `curl http://localhost:8000/metrics`
* **Response Format:**
  ```json
  {
    "average_latency": "15057.05 ms",
    "cache_hit_rate": 0.5,
    "fallback_count": 1,
    "request_count": 2,
    "hallucination_score_averages": 1.0
  }
  ```

---

## 🧠 4. Dynamic Semantic Caching & Invalidation

To protect your Gemini and DeepSeek rate limits, the system features a **Semantic Versioned Query Cache**:

1. **Caching Mechanics**: When a user queries, the text is normalized (whitespace removed, lowercased) and hashed using **SHA-256** to create a unique cache key.
2. **Version Isolation**: Redis maintains a global key `"rag:doc_version"`. Every query cache key is prefixed with this version (e.g., `rag:query:v2:hash`).
3. **Smart Invalidation**: When you upload/ingest a new document (via `/ingest`), the system automatically increments `"rag:doc_version"`. This instantly invalidates all old cache keys (as the prefix changes), ensuring stale data is never served, without needing costly cache deletion scripts!

---

## 🛡️ 5. Gemini 2.5 Flash ⇄ DeepSeek Fallback Logic

When a query is received:
1. **Decomposition**: `QueryDecompositionAgent` splits the query using `gemini-2.5-flash`.
2. **Execution**: The `ReasoningAgent` attempts generation using `gemini-2.5-flash`.
3. **429 Mitigation**: If the Gemini API throws a quota/rate-limit error (e.g., *RESOURCE_EXHAUSTED*):
   * The exception is caught gracefully.
   * `llm_fallback_used` is set to `True`.
   * The query is dynamically re-routed to the **DeepSeek Hugging Face Serverless Inference endpoint**.
   * If both APIs fail, the system falls back to a deterministic summary of local vector sources to guarantee an answer.

---

## 🐳 6. GitHub CI/CD continuous deployment (GHCR)

The project includes an active GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. **Trigger**: On any push to the `main` branch.
2. **Quality Check**: Builds the environment, syncs packages using `uv`, and compiles the core code to catch syntax issues.
3. **Deployment Package**: Builds a slim, production-grade Docker image, registers metadata, and publishes it securely to the **GitHub Container Registry (GHCR)**:
   ```bash
   ghcr.io/sathvik-ai/multi-agent-rag:latest
   ```

To deploy this package on any cloud provider, simply set your environment keys and pull the container:
```bash
docker pull ghcr.io/sathvik-ai/multi-agent-rag:latest
docker run -d -p 8000:8000 --env-file .env ghcr.io/sathvik-ai/multi-agent-rag:latest
```
