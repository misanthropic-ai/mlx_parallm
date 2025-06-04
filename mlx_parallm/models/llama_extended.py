"""Extended Mind Transformer implementation for Llama models in MLX."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, BatchedKVCache, create_additive_causal_mask
from .llama import ModelArgs as LlamaModelArgs, MLP, TransformerBlock, LlamaModel, Model
from ..memory.manager import MemoryManager


@dataclass
class ExtendedModelArgs(LlamaModelArgs):
    """Extended model arguments with memory configuration."""
    # Memory configuration
    use_external_mind: bool = True
    use_external_mind_by_layer: Optional[List[bool]] = None
    memory_topk: int = 10
    mask_by_sim: bool = False
    sim_threshold: float = 0.25
    memory_backend: str = "faiss"
    remove_special_tokens: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if self.use_external_mind_by_layer is None:
            self.use_external_mind_by_layer = [True] * self.num_hidden_layers


class ExtendedAttention(nn.Module):
    """Attention module with external memory retrieval."""
    
    def __init__(self, args: ExtendedModelArgs):
        super().__init__()
        
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        
        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5
        
        attention_bias = getattr(args, "attention_bias", False)
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)
        
        rope_scale = 1
        if args.rope_scaling is not None:
            # Handle both Llama 2 and Llama 3 style
            if "type" in args.rope_scaling and args.rope_scaling["type"] == "linear":
                rope_scale = 1 / args.rope_scaling["factor"]
            elif "rope_type" in args.rope_scaling and args.rope_scaling["rope_type"] == "llama3":
                # For Llama 3, we use the factor differently
                rope_scale = 1 / args.rope_scaling.get("factor", 1)
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )
        
        # Memory configuration
        self.use_external_mind = args.use_external_mind
        self.memory_topk = args.memory_topk
        self.mask_by_sim = args.mask_by_sim
        self.sim_threshold = args.sim_threshold
        
    def _repeat_kv(self, hidden_states: mx.array, n_rep: int) -> mx.array:
        """Repeat key/value heads to match number of query heads."""
        if n_rep == 1:
            return hidden_states
        B, n_kv_heads, L, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, 2)
        hidden_states = mx.tile(hidden_states, [1, 1, n_rep, 1, 1])
        return hidden_states.reshape(B, n_kv_heads * n_rep, L, head_dim)
    
    def _create_memory_mask(self, topk: int, seq_len: int, dtype) -> mx.array:
        """Create mask for external memories where queries only attend to their own memories."""
        mask = mx.ones((seq_len, seq_len * topk), dtype=dtype)
        for i in range(seq_len):
            mask[i, i * topk:(i + 1) * topk] = 0
        return mask * (-1e9)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BatchedKVCache] = None,
        memory_backend: Optional[MemoryManager] = None,
        model_id: Optional[str] = None,
        layer_idx: Optional[int] = None,
        output_retrieved_memory_idx: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, D = x.shape
        
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        # Repeat k/v heads if needed
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            keys = self._repeat_kv(keys, n_rep)
            values = self._repeat_kv(values, n_rep)
        
        # Compute attention scores
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # Extended Mind: Memory retrieval
        retrieved_indices = None
        if self.use_external_mind and memory_backend is not None and model_id is not None:
            backend = memory_backend.get_backend()
            if backend.memory_exists(model_id):
                # Normalize queries for similarity search
                queries_norm = queries / (mx.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
                
                # Search for top-k memories
                selected_keys, selected_values, similarities, indices = backend.search(
                    model_id, queries_norm, self.memory_topk, layer_idx
                )
                
                # Compute attention scores with retrieved memories
                memory_scores = (queries @ selected_keys.transpose(0, 1, 3, 2)) * self.scale
                
                # Apply similarity masking if enabled
                if self.mask_by_sim:
                    # Reshape similarities to match memory_scores
                    sim_mask = similarities.reshape(B, self.n_heads, L, self.memory_topk)
                    sim_mask = (sim_mask < self.sim_threshold).astype(scores.dtype)
                    memory_scores = memory_scores - sim_mask * 1e9
                
                # For concatenation, we need to match dimensions properly
                # memory_scores shape: (B, n_heads, L, topk)
                # scores shape: (B, n_heads, L, kv_seq_len)
                # We concatenate along the last dimension (key dimension)
                scores = mx.concatenate([memory_scores, scores], axis=-1)
                
                # For values, we need to handle the sequence length mismatch
                # selected_values shape: (B, n_heads, L * topk, head_dim)
                # but we need to reshape it to (B, n_heads, topk, head_dim) for each query position
                # Then we can concatenate with the full cached values
                
                # Reshape selected_values from (B, n_heads, L * topk, head_dim) 
                # to (B, n_heads, L, topk, head_dim) then back to (B, n_heads, L * topk, head_dim)
                # This is already in the right format for attention computation
                
                # The trick is that we don't actually concatenate the values here
                # Instead, we'll handle them separately in the attention computation
                memory_values = selected_values
                
                # Update mask to include memory mask
                if mask is not None:
                    memory_mask = self._create_memory_mask(self.memory_topk, L, mask.dtype)
                    # Ensure masks have the same shape by adding batch and head dimensions if needed
                    if mask.ndim == 2:
                        # mask is (L, kv_seq_len), expand to (1, 1, L, kv_seq_len)
                        mask = mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)
                        memory_mask = mx.expand_dims(mx.expand_dims(memory_mask, axis=0), axis=0)
                    elif mask.ndim == 3:
                        # mask is (1, L, kv_seq_len), expand to (1, 1, L, kv_seq_len)
                        mask = mx.expand_dims(mask, axis=0)
                        memory_mask = mx.expand_dims(mx.expand_dims(memory_mask, axis=0), axis=0)
                    # Now concatenate along the key dimension (last axis)
                    full_mask = mx.concatenate([memory_mask, mask], axis=-1)
                    mask = full_mask
                
                if output_retrieved_memory_idx:
                    retrieved_indices = indices
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Compute attention weights
        weights = mx.softmax(scores, axis=-1)
        
        # Apply attention to values
        if 'memory_values' in locals() and memory_values is not None:
            # Split attention weights for memory and regular values
            memory_weights = weights[:, :, :, :self.memory_topk * L]
            regular_weights = weights[:, :, :, self.memory_topk * L:]
            
            # For memory values, we need to handle the fact that each query position
            # has its own set of retrieved memories
            # memory_values shape: (B, n_heads, L * topk, head_dim)
            # We reshape it to group memories by query position
            memory_values_reshaped = memory_values.reshape(B, self.n_heads, L, self.memory_topk, self.head_dim)
            
            # Similarly reshape memory weights to align with the structure
            # memory_weights shape: (B, n_heads, L, topk * L)
            # But we only want each query to attend to its own topk memories
            
            # Create a mask to ensure queries only attend to their own memories
            # This is more efficient than the loop approach
            memory_output_parts = []
            for i in range(L):
                # Extract weights for position i attending to its memories
                start_idx = i * self.memory_topk
                end_idx = (i + 1) * self.memory_topk
                pos_weights = memory_weights[:, :, i:i+1, start_idx:end_idx]  # (B, n_heads, 1, topk)
                
                # Get memory values for position i
                pos_values = memory_values_reshaped[:, :, i, :, :]  # (B, n_heads, topk, head_dim)
                
                # Compute weighted sum
                memory_output = pos_weights @ pos_values  # (B, n_heads, 1, head_dim)
                memory_output_parts.append(memory_output)
            
            # Concatenate outputs from all positions
            memory_output_full = mx.concatenate(memory_output_parts, axis=2)  # (B, n_heads, L, head_dim)
            
            # Compute regular attention with cached values
            regular_output = regular_weights @ values  # (B, n_heads, L, head_dim)
            
            # Combine outputs
            output = memory_output_full + regular_output
        else:
            output = weights @ values
        
        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)
        
        return output, retrieved_indices


class ExtendedTransformerBlock(nn.Module):
    """Transformer block with extended attention."""
    
    def __init__(self, args: ExtendedModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = ExtendedAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BatchedKVCache] = None,
        memory_backend: Optional[MemoryManager] = None,
        model_id: Optional[str] = None,
        layer_idx: Optional[int] = None,
        output_retrieved_memory_idx: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        r, retrieved_idx = self.self_attn(
            self.input_layernorm(x), 
            mask, 
            cache,
            memory_backend,
            model_id,
            layer_idx,
            output_retrieved_memory_idx
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, retrieved_idx


class ExtendedLlamaModel(nn.Module):
    """Extended Llama model with memory-augmented transformer blocks."""
    
    def __init__(self, args: ExtendedModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        
        # Create layers based on use_external_mind_by_layer
        self.layers = []
        for i in range(args.num_hidden_layers):
            if args.use_external_mind_by_layer[i]:
                self.layers.append(ExtendedTransformerBlock(args))
            else:
                self.layers.append(TransformerBlock(args))
        
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Memory manager
        self.memory_manager = MemoryManager(
            default_backend=args.memory_backend,
            embedding_dim=args.hidden_size // args.num_attention_heads
        )
    
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        model_id: Optional[str] = None,
        output_retrieved_memory_idx: bool = False,
    ):
        h = self.embed_tokens(inputs)
        
        mask = None
        if h.shape[1] > 1:
            mask = create_additive_causal_mask(
                h.shape[1], cache[0].offset if cache is not None else 0
            )
            mask = mask.astype(h.dtype)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        all_retrieved_indices = []
        
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if isinstance(layer, ExtendedTransformerBlock):
                h, retrieved_idx = layer(
                    h, mask, cache=c,
                    memory_backend=self.memory_manager,
                    model_id=model_id,
                    layer_idx=i,
                    output_retrieved_memory_idx=output_retrieved_memory_idx
                )
                if output_retrieved_memory_idx:
                    all_retrieved_indices.append(retrieved_idx)
            else:
                h = layer(h, mask, cache=c)
                if output_retrieved_memory_idx:
                    all_retrieved_indices.append(None)
        
        output = self.norm(h)
        
        if output_retrieved_memory_idx:
            return output, all_retrieved_indices
        return output


class ExtendedModel(nn.Module):
    """Extended Mind Transformer model wrapper."""
    
    def __init__(self, args: ExtendedModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ExtendedLlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        
        # Track model instance ID
        self._model_id = None
        
    def set_model_id(self, model_id: str):
        """Set the model instance ID for memory management."""
        self._model_id = model_id
    
    def add_memories(self, memory_tokens: mx.array, tokenizer=None):
        """
        Add memories to the model.
        
        Args:
            memory_tokens: Token IDs of memories, shape (num_memories, seq_len)
            tokenizer: Optional tokenizer for preprocessing
        """
        if self._model_id is None:
            raise ValueError("Model ID must be set before adding memories")
        
        # Ensure memory_tokens is 2D
        if memory_tokens.ndim == 1:
            memory_tokens = memory_tokens.reshape(1, -1)
        
        # Encode memories through the model to get key-value pairs
        memory_embeds = self.model.embed_tokens(memory_tokens)
        h = memory_embeds
        
        # Process through each layer to get key-value representations
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, ExtendedTransformerBlock):
                # Apply layer normalization (from the transformer block)
                h_norm = layer.input_layernorm(h)
                
                # Project to keys and values using the attention module
                keys = layer.self_attn.k_proj(h_norm)
                values = layer.self_attn.v_proj(h_norm)
                
                # Get shape info
                B, L, D = keys.shape
                
                # Reshape to (B, L, n_kv_heads, head_dim) then transpose to (B, n_kv_heads, L, head_dim)
                keys = keys.reshape(B, L, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                
                # Now flatten batch and sequence dimensions: (B * L, n_kv_heads, head_dim)
                keys = keys.transpose(0, 2, 1, 3).reshape(B * L, layer.self_attn.n_kv_heads, -1)
                values = values.transpose(0, 2, 1, 3).reshape(B * L, layer.self_attn.n_kv_heads, -1)
                
                # Add to memory backend
                backend = self.model.memory_manager.get_backend()
                backend.add_memories(
                    self._model_id,
                    keys,
                    values,
                    memory_ids=memory_tokens.reshape(-1).tolist()
                )
                
                # Forward through the rest of the layer to get the next hidden state
                # This is important to propagate the representation through layers
                attn_out, _ = layer.self_attn(h_norm, memory_backend=None)
                h = h + attn_out
                mlp_out = layer.mlp(layer.post_attention_layernorm(h))
                h = h + mlp_out
            else:
                # For non-extended layers, just do a forward pass
                h = layer(h)
    
    def clear_memories(self):
        """Clear all memories for this model instance."""
        if self._model_id is not None:
            backend = self.model.memory_manager.get_backend()
            backend.clear(self._model_id)
    
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        output_retrieved_memory_idx: bool = False,
    ):
        out = self.model(inputs, cache, self._model_id, output_retrieved_memory_idx)
        
        if output_retrieved_memory_idx:
            out, retrieved_indices = out
        
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        
        if output_retrieved_memory_idx:
            return out, retrieved_indices
        return out
    
    def sanitize(self, weights):
        """Remove unused precomputed rotary freqs."""
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
    
    @property
    def layers(self):
        return self.model.layers
    
    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads
    
    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
    
    @property
    def memory_ids(self):
        """Get memory token IDs."""
        if self._model_id is None:
            return None
        backend = self.model.memory_manager.get_backend()
        if backend.memory_exists(self._model_id):
            metadata = backend.list_memories(self._model_id)
            # This is a simplified version - in practice we'd track the actual token IDs
            return metadata
        return None
    
    @memory_ids.setter
    def memory_ids(self, value):
        """Set memory token IDs (for compatibility with reference implementation)."""
        if isinstance(value, list):
            value = mx.array(value)
        self.add_memories(value.reshape(1, -1) if value.ndim == 1 else value)