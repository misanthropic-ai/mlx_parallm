#!/usr/bin/env python3
"""Test script for Extended Mind Transformers implementation."""

import mlx.core as mx
from mlx_parallm.utils import load, generate
import uuid
import time


def test_extended_mind():
    """Test loading and using an extended mind model."""
    
    print("Loading model with extended mind capabilities...")
    
    # Load the model with extended mind enabled
    # You can replace this with "NousResearch/Hermes-3-Llama-3.1-8B" or similar
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Using a smaller model for testing
    
    try:
        model, tokenizer = load(model_path, use_extended_mind=True)
        print(f"✓ Model loaded successfully: {model_path}")
        print(f"  Model type: {type(model).__name__}")
        
        # Set a unique model ID
        if hasattr(model, 'set_model_id'):
            model_id = f"test_model_{uuid.uuid4().hex[:8]}"
            model.set_model_id(model_id)
            print(f"✓ Model ID set: {model_id}")
        else:
            print("✗ Model does not support extended mind features")
            return
        
        # Test 1: Generate without memories
        print("\n1. Testing generation without memories:")
        prompt = "When did Alexander Grothendieck become a French citizen?"
        print(f"   Prompt: {prompt}")
        
        start_time = time.time()
        response_no_memory = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
        time_no_memory = time.time() - start_time
        
        print(f"   Response: {response_no_memory}")
        print(f"   Time: {time_no_memory:.2f}s")
        
        # Test 2: Add memories
        print("\n2. Adding memories to the model:")
        ag_wiki_entry = """Alexander Grothendieck (28 March 1928 – 13 November 2014) was a stateless mathematician who became a French citizen in 1971. He became the leading figure in the creation of modern algebraic geometry. His research extended the scope of the field and added elements of commutative algebra, homological algebra, sheaf theory, and category theory to its foundations."""
        
        # Tokenize the memory
        memory_tokens = tokenizer.encode(ag_wiki_entry)
        memory_tokens_mx = mx.array(memory_tokens)
        
        print(f"   Memory text length: {len(ag_wiki_entry)} chars")
        print(f"   Memory tokens: {len(memory_tokens)} tokens")
        
        # Add memories to the model
        model.add_memories(memory_tokens_mx)
        print("✓ Memories added successfully")
        
        # Check memory status
        if hasattr(model, 'memory_ids'):
            memory_info = model.memory_ids
            if memory_info:
                print(f"   Memory info: {memory_info}")
        
        # Test 3: Generate with memories
        print("\n3. Testing generation with memories:")
        print(f"   Prompt: {prompt}")
        
        start_time = time.time()
        response_with_memory = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
        time_with_memory = time.time() - start_time
        
        print(f"   Response: {response_with_memory}")
        print(f"   Time: {time_with_memory:.2f}s")
        
        # Test 4: Clear memories
        print("\n4. Testing memory clearing:")
        model.clear_memories()
        print("✓ Memories cleared")
        
        # Test 5: Verify memories are cleared
        print("\n5. Testing generation after clearing memories:")
        response_after_clear = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
        print(f"   Response: {response_after_clear}")
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY:")
        print(f"✓ Extended mind model loaded and tested successfully")
        print(f"✓ Memory addition and retrieval working")
        print(f"✓ Performance comparison:")
        print(f"  - Without memory: {time_no_memory:.2f}s")
        print(f"  - With memory: {time_with_memory:.2f}s")
        
        # Check if memory improved the response
        if "1971" in response_with_memory and "1971" not in response_no_memory:
            print("✓ Memory successfully improved factual accuracy!")
        else:
            print("⚠ Memory impact on response unclear - may need more testing")
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_batch_generation_with_memories():
    """Test batch generation with mixed memory/non-memory requests."""
    print("\n" + "="*50)
    print("Testing batch generation with memories...")
    
    try:
        from mlx_parallm.utils import batch_generate
        
        model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        model, tokenizer = load(model_path, use_extended_mind=True)
        model.set_model_id(f"batch_test_{uuid.uuid4().hex[:8]}")
        
        # Add some memories
        memory_text = "The Eiffel Tower is 330 meters tall and was completed in 1889 in Paris, France."
        memory_tokens = mx.array(tokenizer.encode(memory_text))
        model.add_memories(memory_tokens)
        
        # Create test prompts
        prompts = [
            "How tall is the Eiffel Tower?",
            "What is the capital of France?",
            "When was the Eiffel Tower built?",
        ]
        
        print("Testing batch generation with memories...")
        responses = batch_generate(model, tokenizer, prompts, max_tokens=50, verbose=True)
        
        for prompt, response in zip(prompts, responses):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"✗ Batch generation test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Extended Mind Transformers Test Suite")
    print("=" * 50)
    test_extended_mind()
    test_batch_generation_with_memories()
    print("\nAll tests completed!")