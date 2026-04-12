"""
Redis caching layer for search results
"""
import json
import hashlib
from typing import Optional, Dict, Any
import redis
from functools import wraps

class SearchCache:
    """Redis-based cache for search results"""
    
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db)
            self.redis_client.ping()
            self.enabled = True
            print("✅ Redis cache connected")
        except:
            print("⚠️ Redis not available, caching disabled")
            self.enabled = False
        
        self.ttl = ttl
    
    def _get_cache_key(self, query: str, top_k: int, **kwargs) -> str:
        """Generate cache key from query parameters"""
        key_data = f"{query}:{top_k}:{json.dumps(kwargs, sort_keys=True)}"
        return f"search:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def get(self, query: str, top_k: int, **kwargs) -> Optional[Any]:
        """Get cached search results"""
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(query, top_k, **kwargs)
        cached = self.redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, query: str, top_k: int, results: Any, **kwargs):
        """Cache search results"""
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(query, top_k, **kwargs)
        self.redis_client.setex(
            cache_key,
            self.ttl,
            json.dumps(results)
        )
    
    def clear(self, pattern: str = "*"):
        """Clear cache entries"""
        if not self.enabled:
            return
        
        for key in self.redis_client.scan_iter(f"search:{pattern}"):
            self.redis_client.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        keys = list(self.redis_client.scan_iter("search:*"))
        return {
            "enabled": True,
            "total_keys": len(keys),
            "ttl_seconds": self.ttl
        }

def cached_search(ttl: int = 3600):
    """Decorator for caching search results"""
    def decorator(func):
        cache = SearchCache(ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract query and top_k from args/kwargs
            if args and hasattr(args[0], 'query'):
                query = args[0].query
            else:
                query = kwargs.get('query', '')
            
            top_k = kwargs.get('top_k', 10)
            
            # Try to get from cache
            cached_result = cache.get(query, top_k)
            if cached_result:
                print(f"Cache hit for query: {query}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(query, top_k, result)
            print(f"Cached result for query: {query}")
            
            return result
        
        return wrapper
    return decorator
