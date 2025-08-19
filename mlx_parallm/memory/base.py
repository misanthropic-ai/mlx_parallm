"""Base abstract class for memory backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import mlx.core as mx


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def add_memories(
        self, 
        model_id: str, 
        memory_keys: mx.array, 
        memory_values: mx.array,
        memory_ids: Optional[List[int]] = None
    ) -> None:
        """
        Add memory embeddings to the backend.
        
        Args:
            model_id: Unique identifier for the model instance
            memory_keys: Key embeddings of shape (num_memories, num_heads, head_dim)
            memory_values: Value embeddings of shape (num_memories, num_heads, head_dim)
            memory_ids: Optional list of token IDs for the memories
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        model_id: str, 
        queries: mx.array, 
        topk: int = 10,
        layer_idx: Optional[int] = None
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Retrieve top-k memories for given queries.
        
        Args:
            model_id: Unique identifier for the model instance
            queries: Query embeddings of shape (batch_size, num_heads, seq_len, head_dim)
            topk: Number of memories to retrieve per query
            layer_idx: Optional layer index for layer-specific retrieval
            
        Returns:
            Tuple of:
                - selected_keys: Retrieved key embeddings
                - selected_values: Retrieved value embeddings
                - similarities: Similarity scores
                - indices: Indices of retrieved memories
        """
        pass
    
    @abstractmethod
    def clear(self, model_id: str) -> None:
        """Clear all memories for a specific model instance."""
        pass
    
    @abstractmethod
    def list_memories(self, model_id: str) -> Dict:
        """
        List memory metadata for a specific model.
        
        Returns:
            Dictionary containing memory statistics and metadata
        """
        pass
    
    @abstractmethod
    def memory_exists(self, model_id: str) -> bool:
        """Check if memories exist for a specific model."""
        pass

    # Optional fast-path to fetch all stored memories for a model_id
    def get_all(self, model_id: str) -> Tuple[mx.array, mx.array]:
        """
        Return all stored memory keys and values for a model_id.
        Shapes: (num_memories, n_kv_heads, head_dim) for both.
        Implementations may override; default raises NotImplementedError.
        """
        raise NotImplementedError
