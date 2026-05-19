from typing import List, Dict, Any
from qdrant_client.models import Filter
from src.database.connection import get_qdrant_client
from src.ingestion.embedder import Embedder

class RetrievalAgent:
    """
    Agent responsible for searching the Qdrant vector database.
    It takes a natural language query, converts it to a vector, 
    and retrieves the most relevant semantic chunks.
    """
    
    def __init__(self, collection_name: str = "scientific_papers"):
        self.qdrant = get_qdrant_client()
        self.embedder = Embedder()
        self.collection_name = collection_name
        
    def search(self, query: str, limit: int = 5, doc_filter: str = None) -> List[Dict[str, Any]]:
        """
        Performs a semantic vector search.
        """
        print(f"Retrieving context for query: '{query}'")
        
        # 1. Embed the user query
        query_vector = self.embedder.embed_query(query)
        
        # 2. Build Optional Filters (e.g., search only in a specific document)
        query_filter = None
        if doc_filter:
            # Requires Qdrant Filter models for exact match
            from qdrant_client.models import FieldCondition, MatchValue
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_filter)
                    )
                ]
            )
            
        # 3. Query Qdrant
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        # 4. Format Results
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "title": hit.payload.get("title", "Unknown"),
                "document_id": hit.payload.get("document_id", "")
            })
            
        return results
