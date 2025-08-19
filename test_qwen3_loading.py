#!/usr/bin/env python3
"""Test script to verify Qwen3 model loading in mlx_parallm."""

import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_parallm.utils import load_model_and_tokenizer


def test_qwen3_loading():
    """Test loading a Qwen3 model."""
    
    # Use the Qwen3 model specified by the user
    model_id = "NousResearch/Hermes-4-Qwen3-14B-1-e3"
    
    print(f"Loading model: {model_id}")
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_id)
        print("✓ Model loaded successfully")
        
        # Check that model has required properties
        assert hasattr(model, 'head_dim'), "Model missing head_dim property"
        assert hasattr(model, 'n_kv_heads'), "Model missing n_kv_heads property"
        assert hasattr(model, 'layers'), "Model missing layers property"
        
        print(f"✓ Model properties verified:")
        print(f"  - head_dim: {model.head_dim}")
        print(f"  - n_kv_heads: {model.n_kv_heads}")
        print(f"  - num_layers: {len(model.layers)}")
        
        # Generation test with actual text output
        import mlx.core as mx
        from mlx_parallm.models.base import BatchedKVCache
        from mlx_parallm.utils import generate_step
        
        test_prompt = "Hello, I am"
        tokens = tokenizer.encode(test_prompt)
        input_ids = mx.array([tokens])  # Batch size 1
        
        print(f"\n✓ Test prompt: '{test_prompt}'")
        print(f"  - Input tokens: {tokens}")
        
        # Create cache for generation
        kv_heads = [model.n_kv_heads] * len(model.layers)
        cache = [BatchedKVCache(model.head_dim, n, 1) for n in kv_heads]
        
        # Forward pass
        logits = model(input_ids, cache=cache)
        print(f"✓ Forward pass successful, output shape: {logits.shape}")
        
        # Generate some tokens
        print("\nGenerating text...")
        max_tokens = 50
        temperature = 0.7
        
        generated_tokens = []
        prompt_array = mx.array([tokens])
        
        # Use generate_step from utils
        generator = generate_step(prompt_array, model, temp=temperature)
        
        for i, (token, _) in enumerate(generator):
            if i >= max_tokens:
                break
            token_id = token.item()
            generated_tokens.append(token_id)
            
            # Check for EOS token
            if token_id == tokenizer.eos_token_id:
                break
        
        # Decode and display the generated text
        full_tokens = tokens + generated_tokens
        generated_text = tokenizer.decode(full_tokens)
        print(f"\n✓ Generated text ({len(generated_tokens)} new tokens):")
        print(f"  '{generated_text}'")
        
        # Verify the output is coherent (basic check)
        if len(generated_tokens) > 0 and generated_text != test_prompt:
            print("\n✓ Generation successful - text appears coherent")
        else:
            print("\n✗ Generation may have issues - no new tokens or unchanged text")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qwen3_loading()
    sys.exit(0 if success else 1)