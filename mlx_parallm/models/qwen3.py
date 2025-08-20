from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# Import most components from mlx_lm's qwen3 implementation
from mlx_lm.models.qwen3 import (
    ModelArgs,
    MLP,
    TransformerBlock as OriginalTransformerBlock,
    Qwen3Model as OriginalQwen3Model,
    Model as OriginalModel,
)

# Import BatchedKVCache and mask creation from local base
from .base import (
    BatchedKVCache,
    create_additive_causal_mask,
    create_additive_causal_mask_variable,
)


class Attention(nn.Module):
    """Attention module adapted for BatchedKVCache."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        
        head_dim = args.head_dim
        self.scale = head_dim**-0.5
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        
        # Import rope initialization from mlx_lm
        from mlx_lm.models.rope_utils import initialize_rope
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BatchedKVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape
        
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        if cache is not None:
            # Apply RoPE to both queries and keys using per-row offsets
            try:
                offsets = getattr(cache, "offsets", [getattr(cache, "offset", 0)])
            except Exception:
                offsets = [getattr(cache, "offset", 0)]

            if isinstance(offsets, list) and len(offsets) == queries.shape[0]:
                Bq = queries.shape[0]
                q_list = []
                k_list = []
                for b in range(Bq):
                    off_b = int(offsets[b])
                    q_list.append(self.rope(queries[b:b+1], offset=off_b))
                    k_list.append(self.rope(keys[b:b+1], offset=off_b))
                queries = mx.concatenate(q_list, axis=0)
                keys = mx.concatenate(k_list, axis=0)
            else:
                prev_offset = int(getattr(cache, "offset", 0))
                queries = self.rope(queries, offset=prev_offset)
                keys = self.rope(keys, offset=prev_offset)

            # Now append roped keys/values to cache and fetch full KV up to max length
            keys, values = cache.update_and_fetch(keys, values)
        else:
            # No cache: standard RoPE on current chunk
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        # Ensure mask shape broadcasts with (B, n_heads, L_q, L_kv)
        if mask is not None:
            if mask.ndim == 2:
                # (L_q, L_kv) -> (1, 1, L_q, L_kv)
                mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)
            elif mask.ndim == 3:
                # (B, L_q, L_kv) -> (B, 1, L_q, L_kv)
                mask = mx.expand_dims(mask, 1)
        
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    """TransformerBlock using our custom Attention."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)  # Use our custom Attention
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
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
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3Model(nn.Module):
    """Qwen3Model using our custom TransformerBlock."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        
        mask = None
        if h.shape[1] > 1:
            if cache is not None and hasattr(cache, "__getitem__") and getattr(cache[0], "offsets", None) is not None:
                offsets = getattr(cache[0], "offsets")
                total_len = int(max(offsets)) + int(h.shape[1])
                mask = create_additive_causal_mask_variable(int(h.shape[1]), offsets, total_len)
            else:
                off = cache[0].offset if cache is not None else 0
                mask = create_additive_causal_mask(int(h.shape[1]), int(off))
            mask = mask.astype(h.dtype)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        
        return self.norm(h)


class Model(nn.Module):
    """Main model class with required properties for mlx_parallm."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out
    
    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
    
    @property
    def layers(self):
        return self.model.layers
    
    @property
    def head_dim(self):
        return self.args.head_dim
    
    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
