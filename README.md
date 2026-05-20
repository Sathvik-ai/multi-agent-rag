# 🔬 Multi-Agent RAG Pipeline over Scientific Data

A production-style Multi-Agent Scientific RAG system with FastAPI, Qdrant, PostgreSQL, Redis, Neo4j, dual-LLM fallback, hallucination detection, and source-grounded response validation.

## ✨ What Makes This Different

Most RAG systems are "single-shot": one question → one search → one answer. This system does more:

| Feature | Standard RAG | This System |
|---|---|---|
| Query Strategy | Single search | **Multi-Hop Decomposition** |
| Knowledge Gaps | Returns "I don't know" | **Live ArXiv Auto-Download & Ingest** |
| Graph Reasoning | None | **Neo4j Author → Paper Traversal** |
| Caching | Simple key-value | **Version-fingerprinted, auto-invalidated** |
| Source Attribution | Generic | **Each source tagged to the sub-query that found it** |
| LLM | Black-box | **Transparent prompt chain, no LangChain** |

---

## 🏗️ Architecture

```
User Query
    ↓
FastAPI
    ↓
Query Decomposition Agent
    ↓
Retrieval Agent
    ↓
Qdrant + PostgreSQL + Neo4j
    ↓
Reasoning Agent
    ↓
Gemini
      ↓(429)
DeepSeek Fallback
    ↓
Validation Agent
    ↓
Grounded Response + Sources
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API** | FastAPI + asyncio | Web server & async endpoints |
| **Embeddings** | SentenceTransformers `all-MiniLM-L6-v2` | Free, local, 384-dim vectors |
| **Vector Search** | Qdrant | Local semantic similarity search |
| **Relational DB** | PostgreSQL | Document lineage & query logs |
| **Graph DB** | Neo4j | Author → Paper entity relationships |
| **Cache** | Redis | Fingerprinted query caching |
| **LLM** | DeepSeek-V4 | Reasoning & decomposition |
| **Deployment** | Docker Compose | Containerized infrastructure |

---

## 📊 Live Results

### Phase 4 Complete: Multi-Hop RAG with Real Gemini 2.5 Flash

**Query:** *"How does FlashAttention improve upon the original Transformer attention mechanism, and what memory bottlenecks does it solve?"*

**Real Gemini 2.5 Flash Answer (grounded in our local corpus):**
```
FlashAttention improves upon the original Transformer attention in several ways:

1. Training Speed: Up to 3x faster than standard attention implementations
   for common sequence lengths (128-2K). [Source 1, 2]

2. Quality: Scales Transformers to longer sequences — 0.7 improvement in 
   perplexity on GPT-2, 6.4 points lift on long-document classification. [Source 1]

3. Memory Efficiency: Up to 20x more memory-efficient than exact attention 
   baselines. Memory footprint scales LINEARLY (not quadratically). [Source 3]

FlashAttention solves the GPU memory hierarchy bottleneck by keeping 
computations in fast on-chip SRAM instead of slower HBM. [Source 2]
```

**API Response Metadata:**
```json
{
  "confidence": 0.6312,
  "cache_status": "miss",
  "arxiv_fallback_used": false,
  "latency": {
    "decomposition_ms": 1316.09,
    "retrieval_ms": 113.88,
    "llm_ms": 11653.32,
    "total_ms": 13092.19
  }
}
```

**Second identical query (Redis cache hit):**
```json
{ "cache_status": "hit", "latency": { "total_ms": 2.1 } }
```

### Graph API — All Authors & Papers in Neo4j
```
GET /graph/papers → 32 Author-Paper relationships
Ashish Vaswani     → Attention Is All You Need
Noam Shazeer       → Attention Is All You Need
Tri Dao            → FlashAttention: Fast and Memory-Efficient Exact Attention
Patrick Lewis      → Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Shunyu Yao         → ReAct: Synergizing Reasoning and Acting in Language Models
... (27 more)
```

---

## 📚 Sample Data

| File | Type | Content |
|---|---|---|
| `data/papers/attention_is_all_you_need.pdf` | PDF | Original Transformer paper |
| `data/papers/rag_original.pdf` | PDF | RAG for Knowledge-Intensive NLP |
| `data/papers/react.pdf` | PDF | ReAct: Reasoning + Acting in LLMs |
| `data/papers/flash_attention.pdf` | PDF | FlashAttention memory-efficient attention |
| `data/datasets/medical_trials.csv` | CSV | 10 mock Phase 1-4 clinical drug trials |

---

## 🚀 Quickstart

### 📦 Run Pre-Built Production Package (Fastest!)
You can pull and run the fully compiled production package directly from **GitHub Container Registry (GHCR)**:
```bash
# 1. Pull the official production-ready container image
docker pull ghcr.io/sathvik-ai/multi-agent-rag:latest

# 2. Add your environment credentials in .env file
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY and HF_TOKEN

# 3. Start the entire ecosystem (API, Postgres, Redis, Qdrant, Neo4j) 
docker compose up -d
```

### 🛠️ Run Local Development Mode
If you prefer running in local python/development mode:

#### 1. Prerequisites & Installation
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository and install all packages in a sync virtual environment
uv sync

# Configure environment
cp .env.example .env
# Paste your GEMINI_API_KEY and HF_TOKEN in .env
```

#### 2. Start Infrastructure
```bash
docker compose up -d
```

#### 3. Run the API Server
```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

#### 4. Ingest Sample Papers
```bash
uv run python test_ingest.py
```

### 5. Query the System
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does RAG work?", "use_multihop": true}'
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home Welcome status |
| `GET` | `/health` | Service health status |
| `GET` | `/metrics` | System analytics and metrics |
| `POST` | `/ingest` | Upload & ingest PDF or CSV |
| `POST` | `/query` | Multi-hop RAG query |
| `GET` | `/graph/papers` | Neo4j Author→Paper relationships |

---

## 📊 Local Dashboards

| Service | URL | Credentials |
|---|---|---|
| **API Swagger UI** | http://localhost:8000/docs | — |
| **Qdrant Vector DB** | http://localhost:6333/dashboard | — |
| **Neo4j Graph Browser** | http://localhost:7474 | `neo4j` / `rag_password` |

---

## 🚦 Build Status

| Phase | Status | Description |
|---|---|---|
| Phase 1: Infrastructure | ✅ Done | Docker, PostgreSQL, Qdrant, Neo4j, Redis |
| Phase 2: Ingestion Pipeline | ✅ Done | PDF + CSV parser, chunker, embedder |
| Phase 3: Retrieval & Reasoning | ✅ Done | Semantic search + Gemini agent |
| Phase 4: API & Caching | ✅ Done | FastAPI, Redis cache, ArXiv fallback |
| Phase 5: Robustness & CD | ✅ Done | Grounding audit agent, metrics endpoint, GHCR continuous CD |