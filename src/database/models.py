from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import relationship
from .connection import Base

class DocumentMetadata(Base):
    """
    Stores metadata about ingested documents/datasets.
    """
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True) # Unique ID (hash)
    title = Column(String, nullable=False)
    source_type = Column(String, nullable=False) # 'pdf', 'csv', etc.
    file_path = Column(String, nullable=False)
    num_chunks = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """
    Tracks lineage of text chunks stored in the vector database.
    We don't store the full text here to save space (that's in Qdrant),
    but we keep the metadata and lineage.
    """
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True) # Unique chunk ID
    document_id = Column(String, ForeignKey("documents.id"))
    chunk_index = Column(Integer, nullable=False)
    qdrant_point_id = Column(String, nullable=False) # ID in vector DB

    document = relationship("DocumentMetadata", back_populates="chunks")


class QueryLog(Base):
    """
    Logs queries for observability, latency tracking, and cache hits.
    """
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_text = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    cache_hit = Column(String, default="miss") # 'hit', 'miss'
    sources_cited = Column(JSON, nullable=True) # Array of document IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
