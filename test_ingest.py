import os
from pathlib import Path
from src.database.connection import SessionLocal, get_qdrant_client
from src.ingestion.pipeline import IngestionPipeline

def run_test():
    db = SessionLocal()
    qdrant = get_qdrant_client()
    
    pipeline = IngestionPipeline(db_session=db, qdrant_client=qdrant)
    
    paper_path = Path("data/datasets/medical_trials.csv")
    
    if not paper_path.exists():
        print("Paper not found!")
        return
        
    print(f"Testing ingestion pipeline on {paper_path}...")
    doc_id = pipeline.process_file(paper_path)
    
    if doc_id:
        print(f"\nSUCCESS! Document successfully ingested with ID: {doc_id}")
    else:
        print("\nFAILED: No document ID returned.")

if __name__ == "__main__":
    run_test()
