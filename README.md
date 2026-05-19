# Multi-Agent RAG Pipeline over Scientific Data

This repository implements a **multi-agent retrieval and reasoning pipeline** designed to ingest, process, and query heterogeneous scientific data (such as PDFs of academic papers and structured CSV datasets).

## 🚀 Key Features

*   **Multi-Source Ingestion:** Natively parses PDFs (`PyMuPDF`) and CSVs (`pandas`).
*   **Transparent Processing:** Uses a custom logic chunker without relying on opaque "black-box" wrappers.
*   **Local Embeddings:** Embeds data using `all-MiniLM-L6-v2` locally via `SentenceTransformers` for fast, cost-free vectorization.
*   **Advanced Database Architecture:**
    *   **Qdrant:** Semantic vector search storage.
    *   **PostgreSQL:** Document metadata, chunk lineage, and query latency tracking.
    *   **Neo4j:** Entity relationship mapping (Graph database - coming soon).
    *   **Redis:** Query caching (coming soon).

## 🛠️ Current Status & Results

### Phase 2 Complete: Ingestion Pipeline (Level 1)
The data ingestion pipeline has been completely built and tested. It successfully parses documents, splits them intelligently, generates vectors, and securely links metadata between Postgres and Qdrant.

**Test Ingestion Results (Success Level: 100%)**
```
Testing ingestion pipeline on data/papers/attention_is_all_you_need.pdf...
Parsing attention_is_all_you_need.pdf...
Chunking text...
Embedding 23 chunks...
Storing in PostgreSQL and Qdrant...
SUCCESS! Document successfully ingested with ID: 400ce09a-f64a-4681-b387-e526b18c3bc8

Testing ingestion pipeline on data/datasets/medical_trials.csv...
Parsing medical_trials.csv...
Chunking text...
Embedding 2 chunks...
Storing in PostgreSQL and Qdrant...
SUCCESS! Document successfully ingested with ID: 7d543d39-6099-441a-9a24-e77a1a4547fa
```

## 📚 Sample Data
The `data/` directory contains sample data used for testing:
*   `data/papers/rag_original.pdf`: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
*   `data/papers/react.pdf`: *ReAct: Synergizing Reasoning and Acting in Language Models*
*   `data/papers/flash_attention.pdf`: *FlashAttention: Fast and Memory-Efficient Exact Attention*
*   `data/datasets/medical_trials.csv`: A mock dataset of 10 Phase 1-4 clinical drug trials.

## 🐳 Quickstart

To run the infrastructure (PostgreSQL, Qdrant, Neo4j, Redis):
```bash
docker compose up -d
```

To run the Python pipeline, ensure you use the `uv` package manager:
```bash
uv run python test_ingest.py
```