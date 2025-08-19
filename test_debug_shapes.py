#!/usr/bin/env python3
"""Debug script to trace shape issues in Extended Mind."""

import mlx.core as mx
from mlx_parallm.utils import load

print("Debug: Tracing shape issues in Extended Mind")
print("=" * 50)

# Load model
model_path = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
print(f"Loading model: {model_path}")
model, tokenizer = load(model_path, use_extended_mind=True)
model.set_model_id('debug')

# Check model configuration
print("\nModel configuration:")
print(f"  n_heads: {model.layers[0].self_attn.n_heads}")
print(f"  n_kv_heads: {model.layers[0].self_attn.n_kv_heads}")
print(f"  head_dim: {model.layers[0].self_attn.head_dim}")

# Add a simple memory
memory = "Test memory for debugging."
tokens = mx.array(tokenizer.encode(memory))
print(f"\nAdding memory with {len(tokens)} tokens")
model.add_memories(tokens.reshape(1, -1))

# Check what's stored in the backend
backend = model.model.memory_manager.get_backend()
# Check layer-specific namespace since memories are stored per layer
layer_model_id = 'debug__L0'  # Layer 0 namespace
info = backend.list_memories(layer_model_id)
print(f"\nMemory backend info for layer 0: {info}")

# Now let's trace through a forward pass with detailed shape printing
print("\nTracing forward pass...")

# Create a simple input
input_ids = mx.array([[1, 2, 3, 4, 5]])  # Batch size 1, seq len 5
print(f"Input shape: {input_ids.shape}")

# Manually trace through the first layer
layer = model.model.layers[0]
if hasattr(layer, 'self_attn'):
    # Get embeddings
    h = model.model.embed_tokens(input_ids)
    print(f"Embedding shape: {h.shape}")
    
    # Apply layer norm
    h_norm = layer.input_layernorm(h)
    
    # Get Q, K, V projections
    B, L, D = h_norm.shape
    queries = layer.self_attn.q_proj(h_norm)
    keys = layer.self_attn.k_proj(h_norm)
    values = layer.self_attn.v_proj(h_norm)
    
    print(f"\nBefore reshape:")
    print(f"  Q shape: {queries.shape}")
    print(f"  K shape: {keys.shape}")
    print(f"  V shape: {values.shape}")
    
    # Reshape
    queries = queries.reshape(B, L, layer.self_attn.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    
    print(f"\nAfter reshape:")
    print(f"  Q shape: {queries.shape}")
    print(f"  K shape: {keys.shape}")
    print(f"  V shape: {values.shape}")
    
    # Apply RoPE
    queries = layer.self_attn.rope(queries)
    keys = layer.self_attn.rope(keys)
    
    # Repeat KV heads
    n_rep = layer.self_attn.n_heads // layer.self_attn.n_kv_heads
    print(f"\nn_rep (repeat factor): {n_rep}")
    
    if n_rep > 1:
        keys = layer.self_attn._repeat_kv(keys, n_rep)
        values = layer.self_attn._repeat_kv(values, n_rep)
        print(f"\nAfter repeat_kv:")
        print(f"  K shape: {keys.shape}")
        print(f"  V shape: {values.shape}")
    
    # Now test memory retrieval
    print("\nTesting memory retrieval...")
    queries_norm = queries / (mx.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
    
    selected_keys, selected_values, similarities, indices = backend.search(
        'debug__L0', queries_norm, topk=3, layer_idx=0  # Use layer-specific namespace
    )
    
    print(f"\nMemory search results:")
    print(f"  Selected keys shape: {selected_keys.shape}")
    print(f"  Selected values shape: {selected_values.shape}")
    print(f"  Regular values shape: {values.shape}")
    
    # Try concatenation
    print("\nTrying concatenation...")
    try:
        concat_values = mx.concatenate([selected_values, values], axis=2)
        print(f"  Success! Concatenated shape: {concat_values.shape}")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  selected_values.ndim: {selected_values.ndim}")
        print(f"  values.ndim: {values.ndim}")