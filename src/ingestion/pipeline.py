import uuid
from typing import List, Dict, Any
from pathlib import Path
from qdrant_client.models import Distance, VectorParams, PointStruct
from sqlalchemy.orm import Session

from .parser import DocumentParser
from .chunker import TextChunker
from .embedder import Embedder
from ..database.models import DocumentMetadata, DocumentChunk

class IngestionPipeline:
    """
    Orchestrates the ingestion of scientific data:
    1. Parse (PDF/CSV -> Text)
    2. Chunk (Text -> Chunks)
    3. Embed (Chunks -> Vectors)
    4. Store (Vectors -> Qdrant, Metadata -> PostgreSQL)
    """
    
    def __init__(self, db_session: Session, qdrant_client, collection_name="scientific_papers"):
        self.db = db_session
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self):
        """Creates the Qdrant collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.vector_size,
                    distance=Distance.COSINE
                )
            )

    def process_file(self, file_path: str | Path) -> str:
        """
        Process a single file and store it in both DBs.
        Returns the Document ID.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 1. Parse
        print(f"Parsing {path.name}...")
        parsed_data = self.parser.parse_file(path)
        text = parsed_data["text"]
        metadata = parsed_data["metadata"]
        
        # 2. Chunk
        print("Chunking text...")
        chunks = self.chunker.split_text(text)
        
        if not chunks:
            print("No text extracted to chunk.")
            return None

        # 3. Embed
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed_texts(chunks)

        # 4. Store
        print("Storing in PostgreSQL and Qdrant...")
        doc_id = str(uuid.uuid4())
        
        # Write to PostgreSQL (Lineage/Metadata)
        db_doc = DocumentMetadata(
            id=doc_id,
            title=metadata["title"],
            source_type=metadata["source_type"],
            file_path=str(path),
            num_chunks=len(chunks)
        )
        self.db.add(db_doc)
        
        # Write to Qdrant & link chunks in PostgreSQL
        qdrant_points = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            
            # DB Lineage
            db_chunk = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=i,
                qdrant_point_id=chunk_id
            )
            self.db.add(db_chunk)
            
            # Qdrant Vector
            qdrant_points.append(
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "document_id": doc_id,
                        "chunk_index": i,
                        "text": chunk_text,
                        "title": metadata["title"]
                    }
                )
            )
            
        # Commit to Postgres
        self.db.commit()
        
        # Upload to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=qdrant_points
        )
        
        print(f"Successfully ingested {path.name}. Document ID: {doc_id}")
        return doc_id
