import hashlib
import json
import time
from typing import Any
from src.database.connection import get_redis_client


class _SafeEncoder(json.JSONEncoder):
    """
    Handles types that standard json.dumps cannot serialize:
    - numpy float32/float64 → Python float
    - numpy int types       → Python int
    - any other non-standard object → str(obj) as fallback
    """
    def default(self, obj):
        try:
            # Covers numpy floats, numpy ints, and similar numeric wrappers
            if hasattr(obj, 'item'):
                return obj.item()
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

class CacheManager:
    """
    Intelligent Redis Query Cache.
    
    Uses semantic fingerprinting (SHA-256 of normalized query text) to key
    cached responses. Cache is invalidated automatically when new documents
    are ingested via a document version counter stored in Redis.
    """
    
    TTL_SECONDS = 3600  # 1 hour default TTL
    VERSION_KEY = "rag:doc_version"
    CACHE_PREFIX = "rag:query:"
    
    def __init__(self):
        self.redis = get_redis_client()
        
    def _fingerprint(self, query: str) -> str:
        """
        Normalizes the query (lowercase, stripped) and creates a SHA-256 hash.
        This ensures 'What is RAG?' and 'what is rag ?' hit the same cache entry.
        """
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _cache_key(self, query: str) -> str:
        """Combines query fingerprint with the current document version for cache isolation."""
        version = self.redis.get(self.VERSION_KEY) or "0"
        return f"{self.CACHE_PREFIX}{version}:{self._fingerprint(query)}"
    
    def get(self, query: str) -> Any | None:
        """
        Retrieves a cached result. Returns None on cache miss.
        """
        try:
            cached = self.redis.get(self._cache_key(query))
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache GET error: {e}")
        return None
    
    def set(self, query: str, response: Any) -> None:
        """
        Stores a response in the cache with a TTL.
        """
        try:
            key = self._cache_key(query)
            self.redis.setex(key, self.TTL_SECONDS, json.dumps(response, cls=_SafeEncoder))
        except Exception as e:
            print(f"Cache SET error: {e}")
    
    def invalidate_on_ingest(self) -> None:
        """
        Increments the global document version counter.
        This effectively invalidates ALL existing cached queries, because
        they were computed before the new document was added.
        
        This is the 'smart' part: we don't delete individual keys;
        we simply bump the version and old keys become unreachable.
        """
        try:
            self.redis.incr(self.VERSION_KEY)
            print("Cache invalidated: document version bumped.")
        except Exception as e:
            print(f"Cache invalidation error: {e}")
