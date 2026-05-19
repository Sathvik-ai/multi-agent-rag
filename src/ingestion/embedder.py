from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    """
    Generates vector embeddings for text chunks using SentenceTransformers.
    We use all-MiniLM-L6-v2 as requested, as it's fast, local, and requires no API keys.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # This will download the model weights on first run
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Takes a list of string chunks and returns a list of float vectors.
        """
        # encode returns a numpy array, we convert to python lists for DB ingestion
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
        
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single search query.
        """
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist()
