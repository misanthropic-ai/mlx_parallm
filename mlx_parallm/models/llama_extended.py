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
    memory_topk: int = 2
    mask_by_sim: bool = True
    sim_threshold: float = 0.25
    memory_backend: str = "manual"
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
            if "type" in args.rope_scaling and args.rope_scaling["type"] == "linear":
                rope_scale = 1 / args.rope_scaling["factor"]
            elif "rope_type" in args.rope_scaling:
                if args.rope_scaling["rope_type"] == "linear":
                    rope_scale = 1 / args.rope_scaling["factor"]
                # For llama3 rope_type, we don't scale
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
        # Create mask in the requested dtype
        mask = mx.ones((seq_len, seq_len * topk), dtype=dtype)
        for i in range(seq_len):
            mask[i, i * topk:(i + 1) * topk] = 0
        # Apply large negative value to mask out positions
        # Use -10000 instead of -1e9 to avoid overflow in float16
        mask = mask * (-10000.0)
        return mask
    
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
        
        # Apply RoPE with pre-cache update semantics (align with reference)
        if cache is not None:
            prev_offset = cache.offset
            keys, values = cache.update_and_fetch(keys, values)
            queries = self.rope(queries, offset=prev_offset)
            keys = self.rope(keys)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        # Repeat k/v heads if needed
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            keys = self._repeat_kv(keys, n_rep)
            values = self._repeat_kv(values, n_rep)
        
        # Compute regular attention scores (with RoPE)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # Extended Mind: Memory retrieval AFTER RoPE (like reference implementation)
        retrieved_indices = None
        if self.use_external_mind and memory_backend is not None and model_id is not None:
            backend = memory_backend.get_backend()
            # Use layer-specific namespace for memory retrieval
            layer_model_id = f"{model_id}__L{layer_idx}" if layer_idx is not None else model_id
            if backend.memory_exists(layer_model_id):
                # Normalize RoPE'd queries for similarity search
                queries_norm = queries / (mx.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
                
                # Search for top-k memories (memories don't have RoPE)
                # Backend returns: (B, n_heads, L*topk, head_dim) - already handles head repetition
                selected_keys, selected_values, similarities, indices = backend.search(
                    layer_model_id, queries_norm, self.memory_topk, layer_idx
                )

                # Some backends may return memories with n_kv_heads instead of n_heads.
                # If so, repeat to match n_heads (Grouped Query Attention compatibility).
                if selected_keys.shape[1] != self.n_heads:
                    # Expect selected_* heads to equal n_kv_heads; repeat by n_rep
                    if (self.n_heads % self.n_kv_heads) != 0:
                        raise ValueError("n_heads must be a multiple of n_kv_heads for repetition")
                    n_rep = self.n_heads // self.n_kv_heads
                    # selected_* shapes: (B, n_kv_heads, L*topk, D)
                    # Convert to (B, n_kv_heads, L*topk, D) explicitly and repeat along head axis
                    Bk, Hk, LK, Hd = selected_keys.shape
                    selected_keys = mx.expand_dims(selected_keys, 2)
                    selected_keys = mx.tile(selected_keys, [1, 1, n_rep, 1, 1])
                    selected_keys = selected_keys.reshape(Bk, Hk * n_rep, LK, Hd)
                    Bv, Hv, LV, Hdv = selected_values.shape
                    selected_values = mx.expand_dims(selected_values, 2)
                    selected_values = mx.tile(selected_values, [1, 1, n_rep, 1, 1])
                    selected_values = selected_values.reshape(Bv, Hv * n_rep, LV, Hdv)
                    # Repeat similarities across heads if needed: (B, n_kv_heads, L, topk)
                    if similarities is not None and similarities.shape[1] != self.n_heads:
                        sims = similarities
                        sims = mx.expand_dims(sims, 2)
                        sims = mx.tile(sims, [1, 1, n_rep, 1, 1])
                        similarities = sims.reshape(sims.shape[0], sims.shape[1] * n_rep, sims.shape[3], sims.shape[4])
                
                # Compute memory attention scores via dot product, matching reference flow
                memory_scores = (queries @ selected_keys.transpose(0, 1, 3, 2)) * self.scale
                
                # Apply similarity masking if enabled
                if self.mask_by_sim and similarities is not None:
                    # similarities: (B, n_heads, L, topk) align to n_heads if needed
                    sims = similarities
                    if sims.shape[1] != self.n_heads:
                        if (self.n_heads % sims.shape[1]) != 0:
                            raise ValueError("Cannot align similarities heads to n_heads")
                        rep = self.n_heads // sims.shape[1]
                        sims = mx.expand_dims(sims, 2)
                        sims = mx.tile(sims, [1, 1, rep, 1, 1])
                        sims = sims.reshape(sims.shape[0], sims.shape[1] * rep, sims.shape[3], sims.shape[4])
                    # Build mask in MLX: start from ones and fill per-position block
                    sim_mask = mx.ones((B, self.n_heads, L, L * self.memory_topk), dtype=mx.bool_)
                    for i in range(L):
                        start_idx = i * self.memory_topk
                        end_idx = (i + 1) * self.memory_topk
                        cond = sims[:, :, i, :] < self.sim_threshold
                        sim_mask[:, :, i, start_idx:end_idx] = cond
                    sim_mask = sim_mask.astype(memory_scores.dtype)
                    memory_scores = memory_scores - sim_mask * (10000.0 if memory_scores.dtype in (mx.float16, mx.bfloat16) else 1e9)
                
                # Concatenate memory scores with regular scores
                scores = mx.concatenate([memory_scores, scores], axis=-1)
                
                # Concatenate memory values with regular values
                values = mx.concatenate([selected_values, values], axis=2)
                
                # Update mask to include memory mask
                if mask is not None:
                    # Use the same dtype as scores for the memory mask
                    memory_mask = self._create_memory_mask(self.memory_topk, L, scores.dtype)
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
        
        # Compute attention weights with improved numerical stability (upcast to fp32)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        
        
        # Apply attention to values
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
        
        # Build caches per layer and run a forward pass to populate them (no retrieval)
        if self._model_id is None:
            raise ValueError("Model ID must be set before adding memories")

        # Ensure 2D
        if memory_tokens.ndim == 1:
            memory_tokens = memory_tokens.reshape(1, -1)

        B = memory_tokens.shape[0]
        seq_len = memory_tokens.shape[1]

        # Prepare KV caches for each layer
        kv_caches: List[Optional[BatchedKVCache]] = []
        for layer in self.model.layers:
            if isinstance(layer, ExtendedTransformerBlock):
                kv_caches.append(BatchedKVCache(layer.self_attn.head_dim, layer.self_attn.n_kv_heads, batch_size=B))
            else:
                kv_caches.append(None)

        # Run the model to fill caches; pass model_id=None to disable retrieval
        _ = self.model(memory_tokens, cache=kv_caches, model_id=None, output_retrieved_memory_idx=False)

        # Now, for each extended layer, extract un-RoPE'd KV from caches and store as memories
        backend = self.model.memory_manager.get_backend()
        for layer_idx, (layer, c) in enumerate(zip(self.model.layers, kv_caches)):
            if not isinstance(layer, ExtendedTransformerBlock):
                continue
            if c is None or c.keys is None or c.values is None:
                continue
            # c.keys/c.values: (B, n_kv, L, D) with offset == seq_len
            k = c.keys[..., :c.offset, :]
            v = c.values[..., :c.offset, :]
            # (B, n_kv, L, D) -> (B, L, n_kv, D) -> (B*L, n_kv, D)
            k_store = k.transpose(0, 2, 1, 3).reshape(B * k.shape[2], k.shape[1], k.shape[3])
            v_store = v.transpose(0, 2, 1, 3).reshape(B * v.shape[2], v.shape[1], v.shape[3])
            layer_model_id = f"{self._model_id}__L{layer_idx}"
            backend.add_memories(
                layer_model_id,
                k_store,
                v_store,
                memory_ids=memory_tokens.reshape(-1).tolist(),
            )
    
    def clear_memories(self):
        """Clear all memories for this model instance."""
        if self._model_id is not None:
            backend = self.model.memory_manager.get_backend()
            # Clear base model_id and all layer-specific memories
            backend.clear(self._model_id)
            for layer_idx in range(len(self.model.layers)):
                layer_model_id = f"{self._model_id}__L{layer_idx}"
                if backend.memory_exists(layer_model_id):
                    backend.clear(layer_model_id)
    
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
