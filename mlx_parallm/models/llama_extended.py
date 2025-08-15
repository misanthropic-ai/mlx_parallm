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
    memory_alpha: float = 1.0
    debug_extended: bool = False
    memory_value_alpha: float = 1.0
    memory_weight_cap: float = 1.0
    memory_calibrate: bool = True
    strict_in_attention: bool = False
    
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
        self.memory_alpha = getattr(args, "memory_alpha", 1.0)
        self.memory_value_alpha = getattr(args, "memory_value_alpha", 1.0)
        self.debug_extended = getattr(args, "debug_extended", False)
        self.memory_weight_cap = getattr(args, "memory_weight_cap", 1.0)
        self.memory_calibrate = getattr(args, "memory_calibrate", True)
        self.strict_in_attention = getattr(args, "strict_in_attention", False)
        
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
        parent_model: Optional["ExtendedLlamaModel"] = None,
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
        
        # Compute attention scores (local)
        local_scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        scores = local_scores
        had_memory = False
        
        # Extended Mind: Memory retrieval
        retrieved_indices = None
        selected_keys = None
        selected_values = None
        similarities = None
        if self.use_external_mind and model_id is not None:
            # Strict in-module retrieval path
            if getattr(self, 'strict_in_attention', False) and parent_model is not None and hasattr(parent_model, '_strict_memories') and layer_idx in parent_model._strict_memories:
                kk, vv = parent_model._strict_memories[layer_idx]  # (n_kv_heads, N, D)
                # Normalize queries for similarity
                qn = queries / (mx.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)  # (B,H,L,D)
                import numpy as _np
                qn_np = _np.array(qn)
                kk_np = _np.array(kk)
                vv_np = _np.array(vv)
                Bn, Hn, Ln, Dn = qn_np.shape
                n_kv = kk_np.shape[0]
                TK = self.memory_topk
                sel_k_heads = []
                sel_v_heads = []
                sims_heads = []
                idx_heads = []
                for h in range(Hn):
                    kv_h = h % n_kv
                    MK = kk_np[kv_h]  # (N,D)
                    MV = vv_np[kv_h]
                    q2 = qn_np[:, h, :, :].reshape(Bn*Ln, Dn)
                    s = q2 @ MK.T
                    tk = min(TK, MK.shape[0])
                    idx = _np.argpartition(-s, kth=tk-1, axis=1)[:, :tk]
                    row = _np.arange(idx.shape[0])[:, None]
                    part = s[row, idx]
                    order = _np.argsort(-part, axis=1)
                    idx_sorted = idx[row, order]
                    val_sorted = part[row, order]
                    sk = MK[idx_sorted].reshape(Bn, Ln, tk, Dn)
                    sv = MV[idx_sorted].reshape(Bn, Ln, tk, Dn)
                    sel_k_heads.append(sk)
                    sel_v_heads.append(sv)
                    sims_heads.append(val_sorted.reshape(Bn, Ln, tk))
                    idx_heads.append(idx_sorted.reshape(Bn, Ln, tk))
                selected_keys = _np.stack(sel_k_heads, axis=1)  # (B,H,L,tk,D)
                selected_values = _np.stack(sel_v_heads, axis=1)
                similarities = _np.stack(sims_heads, axis=1)  # (B,H,L,tk)
                indices = _np.stack(idx_heads, axis=1)
                selected_keys = mx.array(selected_keys.reshape(Bn, Hn, Ln*tk, Dn))
                selected_values = mx.array(selected_values.reshape(Bn, Hn, Ln*tk, Dn))
                similarities = mx.array(similarities)
                indices = mx.array(indices)
            else:
                # Backend retrieval path
                backend = memory_backend.get_backend()
            # Use per-layer memory namespace to avoid cross-layer retrieval
                layer_model_id = f"{model_id}__L{layer_idx}" if layer_idx is not None else model_id
                if not (getattr(self, 'strict_in_attention', False) and parent_model is not None and hasattr(parent_model, '_strict_memories') and layer_idx in parent_model._strict_memories):
                    if backend.memory_exists(layer_model_id):
                        queries_norm = queries / (mx.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
                        selected_keys, selected_values, similarities, indices = backend.search(
                            layer_model_id, queries_norm, self.memory_topk, layer_idx
                        )
                
                # Compute attention scores with retrieved memories
                if selected_keys is not None:
                    memory_scores = (queries @ selected_keys.transpose(0, 1, 3, 2)) * self.scale
                
                # Apply similarity masking if enabled
                if self.mask_by_sim and similarities is not None and selected_keys is not None:
                    # similarities: (B, H, L, topk)
                    # Build a block-diagonal mask for memory_scores of shape (B, H, L, L*topk)
                    sims_np = np.array(similarities)
                    Bn, Hn, Ln, TK = sims_np.shape
                    # mask True where we should mask (similarity < threshold)
                    mask_blocks = (sims_np < self.sim_threshold)
                    full_mask = np.ones((Bn, Hn, Ln, Ln * TK), dtype=bool)
                    for i in range(Ln):
                        full_mask[:, :, i, i * TK : (i + 1) * TK] = mask_blocks[:, :, i, :]
                    full_mask_mx = mx.array(full_mask, dtype=memory_scores.dtype)
                    memory_scores = memory_scores - full_mask_mx * 1e9

                # Optionally calibrate memory score distribution to match local scores (masked moments)
                if selected_keys is not None and self.memory_calibrate:
                    eps = 1e-6
                    # Local stats
                    loc_mu = local_scores.mean(axis=-1, keepdims=True)
                    loc_sigma = mx.sqrt(mx.maximum(0.0, local_scores.var(axis=-1, keepdims=True)))
                    # Memory stats: if similarity mask exists, ignore masked positions
                    if self.mask_by_sim and similarities is not None:
                        sims_np = np.array(similarities)
                        Bn, Hn, Ln, TK = sims_np.shape
                        valid = (sims_np >= self.sim_threshold).astype(np.float32)
                        full = np.zeros((Bn, Hn, Ln, Ln * TK), dtype=np.float32)
                        for i in range(Ln):
                            full[:, :, i, i * TK : (i + 1) * TK] = valid[:, :, i, :]
                        valid_mx = mx.array(full, dtype=memory_scores.dtype)
                        count = mx.maximum(valid_mx.sum(axis=-1, keepdims=True), eps)
                        mem_mu = (memory_scores * valid_mx).sum(axis=-1, keepdims=True) / count
                        diff = (memory_scores - mem_mu) * valid_mx
                        mem_var = (diff * (memory_scores - mem_mu)).sum(axis=-1, keepdims=True) / count
                        mem_sigma = mx.sqrt(mx.maximum(0.0, mem_var))
                    else:
                        mem_mu = memory_scores.mean(axis=-1, keepdims=True)
                        mem_sigma = mx.sqrt(mx.maximum(0.0, memory_scores.var(axis=-1, keepdims=True)))
                    # Z-score memory to local
                    memory_scores = (memory_scores - mem_mu) / (mem_sigma + eps)
                    memory_scores = memory_scores * (loc_sigma + eps) + loc_mu
                # Scale memory contribution (overall)
                if selected_keys is not None:
                    memory_scores = memory_scores * self.memory_alpha
                
                # Concatenate memory scores before regular scores along key dimension
                # memory_scores: (B, n_heads, L, L*topk), local_scores: (B, n_heads, L, kv_seq_len)
                if selected_keys is not None:
                    scores = mx.concatenate([memory_scores, local_scores], axis=-1)

                # Concatenate memory values before regular values along sequence/key dimension
                # selected_values: (B, n_heads, L*topk, head_dim), values: (B, n_heads, kv_seq_len, head_dim)
                if selected_values is not None:
                    values = mx.concatenate([selected_values, values], axis=2)
                    had_memory = True
                
                # Update mask to include memory mask only when memory is used
                if mask is not None and selected_keys is not None:
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
            # Ensure scores is defined in both memory/non-memory cases
            try:
                _ = scores
            except NameError:
                scores = local_scores
            scores = scores + mask
        
        # Compute attention weights (upcast to float32 for stability like reference)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        if self.debug_extended:
            # Basic diagnostics
            import numpy as _np
            ws = _np.array(weights)
            if ws.size:
                mem_len = L * getattr(self, "memory_topk", 0)
                if had_memory and mem_len > 0 and mem_len < ws.shape[-1]:
                    mem_w = ws[..., :mem_len].sum(axis=-1).mean()
                    loc_w = ws[..., mem_len:].sum(axis=-1).mean()
                    print(f"[EXTDBG] mean mem_w={mem_w:.4f}, mean loc_w={loc_w:.4f}")
                print(f"[EXTDBG] weights finite={_np.isfinite(ws).all()}")
        # If memory is present, split and optionally gate/cap its contribution
        mem_len = L * self.memory_topk if hasattr(self, "memory_topk") else 0
        if had_memory and mem_len > 0 and mem_len < weights.shape[-1]:
            memory_weights = weights[:, :, :, :mem_len]
            local_weights = weights[:, :, :, mem_len:]
            memory_values = values[:, :, :mem_len, :]
            local_values = values[:, :, mem_len:, :]
            # Apply memory weight cap to prevent collapse
            if self.memory_weight_cap < 1.0:
                eps = 1e-6
                sum_m = memory_weights.sum(axis=-1, keepdims=True)
                sum_l = local_weights.sum(axis=-1, keepdims=True)
                cap = mx.array(self.memory_weight_cap, dtype=weights.dtype)
                one = mx.array(1.0, dtype=weights.dtype)
                factor_m = mx.minimum(one, cap / (sum_m + eps))
                # Remaining probability goes to local
                remaining = one - factor_m * sum_m
                factor_l = remaining / (sum_l + eps)
                memory_weights = memory_weights * factor_m
                local_weights = local_weights * factor_l
            output = (local_weights @ local_values) + self.memory_value_alpha * (memory_weights @ memory_values)
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
        parent_model: Optional["ExtendedLlamaModel"] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        r, retrieved_idx = self.self_attn(
            self.input_layernorm(x), 
            mask, 
            cache,
            memory_backend,
            model_id,
            layer_idx,
            output_retrieved_memory_idx,
            parent_model
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
        # Strict in-attention memory caches per layer (kv_head-major)
        self.strict_in_attention = getattr(args, "strict_in_attention", False)
        self._strict_memories = {}
    
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
            offset = 0
            if cache is not None:
                try:
                    offset = next((c.offset for c in cache if c is not None), 0)
                except Exception:
                    offset = 0
            mask = create_additive_causal_mask(h.shape[1], offset)
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
                    output_retrieved_memory_idx=output_retrieved_memory_idx,
                    parent_model=self
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
        
        # Build per-layer caches and run a forward pass to populate them, then store post-RoPE KV
        caches = []
        B = memory_tokens.shape[0]
        for layer in self.model.layers:
            if isinstance(layer, ExtendedTransformerBlock):
                caches.append(BatchedKVCache(layer.self_attn.head_dim, layer.self_attn.n_kv_heads, batch_size=B))
            else:
                caches.append(None)

        _ = self.model(memory_tokens, cache=caches, model_id=None, output_retrieved_memory_idx=False)

        backend = self.model.memory_manager.get_backend()
        # Optional special-token filtering setup
        filter_special = getattr(self.model.args, 'remove_special_tokens', False)
        special_ids = set()
        if filter_special and tokenizer is not None:
            try:
                if getattr(tokenizer, 'bos_token_id', None) is not None:
                    special_ids.add(int(tokenizer.bos_token_id))
                if getattr(tokenizer, 'eos_token_id', None) is not None:
                    special_ids.add(int(tokenizer.eos_token_id))
                if getattr(tokenizer, 'pad_token_id', None) is not None and tokenizer.pad_token_id is not None:
                    special_ids.add(int(tokenizer.pad_token_id))
                # common zero id
                special_ids.add(0)
            except Exception:
                special_ids = set()

        # Convert memory tokens to numpy for indexing
        mem_tok_np = None
        if filter_special and tokenizer is not None:
            try:
                mem_tok_np = np.array(memory_tokens)
            except Exception:
                mem_tok_np = None

        for layer_idx, (layer, c) in enumerate(zip(self.model.layers, caches)):
            if isinstance(layer, ExtendedTransformerBlock) and c is not None and c.keys is not None and c.values is not None:
                # c.keys: (B, n_kv_heads, L, head_dim)
                Bk, Hkv, Lseq, Dhd = c.keys.shape
                # Build per-batch keep indices to drop special-token positions
                if mem_tok_np is not None and len(special_ids) > 0:
                    keep_lists = []
                    for b in range(Bk):
                        Lmem = mem_tok_np.shape[1]
                        keep = []
                        for i in range(Lseq):
                            if i < Lmem:
                                if int(mem_tok_np[b, i]) not in special_ids:
                                    keep.append(i)
                            else:
                                keep.append(i)
                        if len(keep) == 0:
                            # fallback: keep all if everything filtered
                            keep = list(range(Lseq))
                        keep_lists.append(np.array(keep, dtype=np.int64))
                    # Gather per-batch, then concat
                    k_batch = []
                    v_batch = []
                    for b in range(Bk):
                        idx = mx.array(keep_lists[b])
                        k_b = mx.take(c.keys[b], idx, axis=1)  # (n_kv_heads, L_kept, D)
                        v_b = mx.take(c.values[b], idx, axis=1)
                        # transpose to (L_kept, n_kv_heads, D)
                        k_b = k_b.transpose(1, 0, 2)
                        v_b = v_b.transpose(1, 0, 2)
                        k_batch.append(k_b)
                        v_batch.append(v_b)
                    k = mx.concatenate(k_batch, axis=0)  # (sum L_kept, n_kv_heads, D)
                    v = mx.concatenate(v_batch, axis=0)
                else:
                    # Flatten without filtering: (B*L, n_kv_heads, head_dim)
                    k = c.keys.transpose(0, 2, 1, 3).reshape(Bk * Lseq, Hkv, Dhd)
                    v = c.values.transpose(0, 2, 1, 3).reshape(Bk * Lseq, Hkv, Dhd)
                layer_model_id = f"{self._model_id}__L{layer_idx}"
                backend.add_memories(layer_model_id, k, v, memory_ids=memory_tokens.reshape(-1).tolist())
                # Also store strict per-layer caches keyed by layer index (n_kv_heads, N, D)
                if getattr(self.model, 'strict_in_attention', False):
                    kk = k.transpose(1, 0, 2)  # (n_kv_heads, N, D)
                    vv = v.transpose(1, 0, 2)
                    if not hasattr(self.model, '_strict_memories'):
                        self.model._strict_memories = {}
                    self.model._strict_memories[layer_idx] = (kk, vv)
    
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
