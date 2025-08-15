"""Memory manager for coordinating different memory backends."""

from typing import Dict, Optional, Union
from .base import MemoryBackend
from .faiss_backend import FAISSMemoryBackend
from .manual_backend import ManualMemoryBackend


class MemoryManager:
    """Manages memory backends for different models."""
    
    def __init__(self, default_backend: str = "faiss", embedding_dim: int = 128):
        """
        Initialize memory manager.
        
        Args:
            default_backend: Default backend type to use
            embedding_dim: Default embedding dimension
        """
        self.default_backend = default_backend
        self.embedding_dim = embedding_dim
        self.backends: Dict[str, MemoryBackend] = {}
        self._initialize_default_backend()
    
    def _initialize_default_backend(self):
        """Initialize the default backend."""
        if self.default_backend == "faiss":
            self.backends["faiss"] = FAISSMemoryBackend(self.embedding_dim)
        elif self.default_backend == "manual":
            self.backends["manual"] = ManualMemoryBackend(self.embedding_dim)
        else:
            raise ValueError(f"Unknown backend type: {self.default_backend}")
    
    def get_backend(self, backend_type: Optional[str] = None) -> MemoryBackend:
        """
        Get a memory backend instance.
        
        Args:
            backend_type: Type of backend to get (defaults to default_backend)
            
        Returns:
            MemoryBackend instance
        """
        backend_type = backend_type or self.default_backend
        
        if backend_type not in self.backends:
            if backend_type == "faiss":
                self.backends[backend_type] = FAISSMemoryBackend(self.embedding_dim)
            elif backend_type == "manual":
                self.backends[backend_type] = ManualMemoryBackend(self.embedding_dim)
            else:
                raise ValueError(f"Unknown backend type: {backend_type}")
        
        return self.backends[backend_type]
    
    def set_embedding_dim(self, dim: int):
        """Update the embedding dimension for new backends."""
        self.embedding_dim = dim
        
    def clear_all(self):
        """Clear all backends."""
        for backend in self.backends.values():
            # Clear all model memories in each backend
            # This is a simplified approach - in practice we'd track model IDs
            pass
