import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.database.connection import SessionLocal, get_qdrant_client
from src.database.models import QueryLog
from src.database.init_db import init_db
from src.ingestion.pipeline import IngestionPipeline
from src.agents.reasoning import ReasoningAgent
from src.agents.hallucination import HallucinationDetector
from src.api.cache import CacheManager
from src.api.eval_data import EVAL_DATASET

# ─────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent RAG Pipeline",
    description=(
        "A multi-agent retrieval and reasoning API for scientific data. "
        "Features: Multi-Hop Reasoning, Query Decomposition, ArXiv Live Fallback, "
        "Redis Caching, and Neo4j Knowledge Graph traversal."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons — initialized once at startup
cache = CacheManager()
reasoning_agent = ReasoningAgent()
hallucination_detector = HallucinationDetector()

@app.on_event("startup")
def startup_event():
    """Initialize database tables on server startup."""
    init_db()
    print("RAG Pipeline API is ready.")

# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_multihop: bool = True          # Enable query decomposition
    enable_arxiv_fallback: bool = True # Enable ArXiv self-healing

class QueryResponse(BaseModel):
    question: str
    sub_questions: list[str]
    answer: str
    confidence: float
    arxiv_fallback_used: bool
    cache_status: str                  # 'hit' or 'miss'
    sources: list[dict]
    latency: dict
    hallucination: Optional[dict] = None  # Grounding audit result

class IngestResponse(BaseModel):
    message: str
    document_id: Optional[str]

# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Multi-Agent RAG Pipeline"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a scientific document (PDF or CSV).
    Automatically parses, chunks, embeds, and stores in Qdrant + PostgreSQL + Neo4j.
    Also invalidates the Redis cache so stale results aren't served.
    """
    db = SessionLocal()
    qdrant = get_qdrant_client()
    
    # Save the uploaded file to a temp location
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ['.pdf', '.csv']:
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")
    
    tmp_path = Path(f"data/uploads/{file.filename}")
    tmp_path.parent.mkdir(exist_ok=True)
    
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        
        pipeline = IngestionPipeline(db, qdrant)
        doc_id = pipeline.process_file(tmp_path)
        
        # Invalidate Redis cache — new data is now available
        cache.invalidate_on_ingest()
        
        return IngestResponse(
            message=f"Successfully ingested '{file.filename}'",
            document_id=doc_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """
    Ask a scientific question. 
    
    Unique Features:
    - Multi-Hop: Decomposes complex questions into sub-queries for deeper retrieval.
    - ArXiv Fallback: If confidence is low, automatically searches ArXiv live.
    - Redis Cache: Returns cached responses instantly on repeated queries.
    - Latency Breakdown: Shows time spent in decomposition, retrieval, and LLM.
    """
    t_start = time.time()
    
    # 1. Check Redis cache first
    cached = cache.get(request.question)
    if cached:
        cached['cache_status'] = 'hit'
        return cached
    
    # 2. Run the multi-hop reasoning pipeline
    db = SessionLocal()
    qdrant = get_qdrant_client()
    ingest_pipeline = IngestionPipeline(db, qdrant) if request.enable_arxiv_fallback else None
    
    try:
        result = reasoning_agent.ask(
            question=request.question,
            ingest_pipeline=ingest_pipeline
        )
        
        total_ms = round((time.time() - t_start) * 1000, 2)
        result['latency']['total_ms'] = total_ms
        result['cache_status'] = 'miss'
        
        # 3. Hallucination Detection (Level 3)
        hall_result = hallucination_detector.evaluate(
            question=request.question,
            answer=result['answer'],
            sources=result['sources']
        )
        result['hallucination'] = hall_result
        
        # 4. Log query to PostgreSQL for observability
        log = QueryLog(
            query_text=request.question,
            latency_ms=total_ms,
            cache_hit=result['cache_status'],
            sources_cited=[s.get('document_id') for s in result.get('sources', [])]
        )
        db.add(log)
        db.commit()
        
        # 5. Store result in Redis cache for next time
        cache.set(request.question, result)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/graph/papers", tags=["Graph"])
def list_papers_in_graph():
    """
    Returns all papers and their authors from the Neo4j Knowledge Graph.
    """
    from src.database.connection import get_neo4j_driver
    driver = get_neo4j_driver()
    
    results = []
    with driver.session() as session:
        records = session.run(
            "MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name AS author, p.title AS title, p.id AS doc_id"
        )
        for record in records:
            results.append({
                "author": record["author"],
                "title": record["title"],
                "document_id": record["doc_id"]
            })
    return {"papers": results, "count": len(results)}


@app.post("/evaluate", tags=["Evaluation"])
def evaluate_pipeline():
    """
    Level 3: Systematic Evaluation Endpoint.
    Runs our built-in ground-truth dataset against the live pipeline and 
    reports accuracy (keyword hit rate) and hallucination scores per question.
    """
    results = []
    passed = 0

    for item in EVAL_DATASET:
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        # Run the RAG pipeline on this question
        rag_result = reasoning_agent.ask(question=question, ingest_pipeline=None)
        answer = rag_result.get("answer", "").lower()

        # Keyword coverage check
        hits = [kw for kw in expected_keywords if kw.lower() in answer]
        accuracy = round(len(hits) / len(expected_keywords), 2)
        if accuracy >= 0.5:
            passed += 1

        # Hallucination audit
        hall = hallucination_detector.evaluate(
            question=question,
            answer=rag_result.get("answer", ""),
            sources=rag_result.get("sources", [])
        )

        results.append({
            "question": question,
            "keyword_accuracy": accuracy,
            "keywords_hit": hits,
            "keywords_expected": expected_keywords,
            "retrieval_confidence": rag_result.get("confidence", 0),
            "hallucination": hall,
            "answer_snippet": rag_result.get("answer", "")[:300]
        })

    return {
        "total_questions": len(EVAL_DATASET),
        "passed": passed,
        "overall_accuracy": round(passed / len(EVAL_DATASET), 2),
        "results": results
    }
