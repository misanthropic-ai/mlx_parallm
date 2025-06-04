"""FAISS-based memory backend implementation."""

import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
from .base import MemoryBackend


class FAISSMemoryBackend(MemoryBackend):
    """FAISS-based memory storage backend."""
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize FAISS backend.
        
        Args:
            embedding_dim: Dimension of the embeddings (head_dim)
        """
        self.embedding_dim = embedding_dim
        self.indexes: Dict[str, Dict] = {}  # model_id -> {index, keys, values, ids}
        
    def _create_index(self, model_id: str, num_heads: int):
        """Create a new FAISS index for a model."""
        if model_id not in self.indexes:
            self.indexes[model_id] = {
                'faiss_indexes': [],
                'memory_keys': [],
                'memory_values': [],
                'memory_ids': [],
                'num_heads': num_heads
            }
            
            # Create separate index for each attention head
            for _ in range(num_heads):
                index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                self.indexes[model_id]['faiss_indexes'].append(index)
    
    def add_memories(
        self,
        model_id: str,
        memory_keys: mx.array,
        memory_values: mx.array,
        memory_ids: Optional[List[int]] = None
    ) -> None:
        """Add memory embeddings to the FAISS index."""
        # Convert MLX arrays to numpy
        keys_np = np.array(memory_keys)
        values_np = np.array(memory_values)
        
        # Shape: (num_memories, num_heads, head_dim)
        num_memories, num_heads, head_dim = keys_np.shape
        
        if head_dim != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {head_dim}")
        
        # Create index if doesn't exist
        self._create_index(model_id, num_heads)
        
        # Normalize keys for cosine similarity
        keys_normalized = keys_np / (np.linalg.norm(keys_np, axis=-1, keepdims=True) + 1e-8)
        
        # Add to each head's index
        for head_idx in range(num_heads):
            head_keys = keys_normalized[:, head_idx, :]  # (num_memories, head_dim)
            self.indexes[model_id]['faiss_indexes'][head_idx].add(head_keys.astype(np.float32))
        
        # Store original keys and values
        self.indexes[model_id]['memory_keys'].append(memory_keys)
        self.indexes[model_id]['memory_values'].append(memory_values)
        
        if memory_ids is not None:
            self.indexes[model_id]['memory_ids'].extend(memory_ids)
    
    def search(
        self,
        model_id: str,
        queries: mx.array,
        topk: int = 10,
        layer_idx: Optional[int] = None
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Retrieve top-k memories using FAISS."""
        if model_id not in self.indexes:
            raise ValueError(f"No memories found for model {model_id}")
        
        # Convert queries to numpy
        queries_np = np.array(queries)
        batch_size, num_heads, seq_len, head_dim = queries_np.shape
        
        # Normalize queries
        queries_normalized = queries_np / (np.linalg.norm(queries_np, axis=-1, keepdims=True) + 1e-8)
        
        # Prepare output arrays
        all_selected_keys = []
        all_selected_values = []
        all_similarities = []
        all_indices = []
        
        # Concatenate all stored memories
        all_keys = mx.concatenate(self.indexes[model_id]['memory_keys'], axis=0)
        all_values = mx.concatenate(self.indexes[model_id]['memory_values'], axis=0)
        
        # Search for each head
        for head_idx in range(num_heads):
            head_queries = queries_normalized[:, head_idx, :, :].reshape(-1, head_dim)  # (batch_size * seq_len, head_dim)
            
            # Search in FAISS
            similarities, indices = self.indexes[model_id]['faiss_indexes'][head_idx].search(
                head_queries.astype(np.float32), topk
            )
            
            # Reshape results
            similarities = similarities.reshape(batch_size, seq_len, topk)
            indices = indices.reshape(batch_size, seq_len, topk)
            
            # Gather selected keys and values
            selected_keys = []
            selected_values = []
            
            for b in range(batch_size):
                for s in range(seq_len):
                    idx = indices[b, s, :]
                    selected_keys.append(all_keys[idx, head_idx, :])
                    selected_values.append(all_values[idx, head_idx, :])
            
            selected_keys = mx.stack(selected_keys).reshape(batch_size, seq_len, topk, head_dim)
            selected_values = mx.stack(selected_values).reshape(batch_size, seq_len, topk, head_dim)
            
            all_selected_keys.append(selected_keys)
            all_selected_values.append(selected_values)
            all_similarities.append(mx.array(similarities))
            all_indices.append(mx.array(indices))
        
        # Stack across heads
        selected_keys = mx.stack(all_selected_keys, axis=1)  # (batch_size, num_heads, seq_len, topk, head_dim)
        selected_values = mx.stack(all_selected_values, axis=1)
        similarities = mx.stack(all_similarities, axis=1)  # (batch_size, num_heads, seq_len, topk)
        indices = mx.stack(all_indices, axis=1)
        
        # Reshape to match expected format
        selected_keys = selected_keys.reshape(batch_size, num_heads, seq_len * topk, head_dim)
        selected_values = selected_values.reshape(batch_size, num_heads, seq_len * topk, head_dim)
        
        return selected_keys, selected_values, similarities, indices
    
    def clear(self, model_id: str) -> None:
        """Clear all memories for a model."""
        if model_id in self.indexes:
            del self.indexes[model_id]
    
    def list_memories(self, model_id: str) -> Dict:
        """List memory metadata."""
        if model_id not in self.indexes:
            return {"num_memories": 0, "exists": False}
        
        num_memories = len(self.indexes[model_id]['memory_ids']) if self.indexes[model_id]['memory_ids'] else 0
        if not num_memories and self.indexes[model_id]['memory_keys']:
            num_memories = sum(len(k) for k in self.indexes[model_id]['memory_keys'])
        
        return {
            "num_memories": num_memories,
            "num_heads": self.indexes[model_id]['num_heads'],
            "embedding_dim": self.embedding_dim,
            "exists": True
        }
    
    def memory_exists(self, model_id: str) -> bool:
        """Check if memories exist for a model."""
        return model_id in self.indexes and len(self.indexes[model_id]['memory_keys']) > 0