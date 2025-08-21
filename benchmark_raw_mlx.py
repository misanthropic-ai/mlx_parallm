#!/usr/bin/env python3
"""Benchmark raw MLX generate vs MLX ParaLLM inference server."""

import time
import argparse
import asyncio
from typing import List, Tuple
from statistics import mean, stdev
from mlx_lm import load, generate


def benchmark_single_request(model, tokenizer, prompt: str, max_tokens: int = 50) -> Tuple[float, float, str]:
    """Benchmark a single request with raw MLX generate."""
    
    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted_prompt = prompt
    
    # Time the generation
    start_time = time.perf_counter()
    
    response = generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=max_tokens,
        verbose=False,  # Set to True to see token-by-token generation
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Estimate tokens generated (rough approximation)
    tokens_generated = len(response.split()) * 1.3  # Rough token estimate
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
    
    return total_time, tokens_per_second, response


def benchmark_sequential_requests(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_tokens: int = 50
) -> List[Tuple[float, float, str]]:
    """Benchmark multiple requests sequentially (no batching)."""
    
    results = []
    for prompt in prompts:
        result = benchmark_single_request(model, tokenizer, prompt, max_tokens)
        results.append(result)
        print(f".")  # Progress indicator
    print()
    
    return results


def benchmark_batch_requests(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_tokens: int = 50
) -> Tuple[float, float, List[str]]:
    """Benchmark multiple requests as a batch (if supported)."""
    
    # Note: MLX LM's generate doesn't natively support batching
    # This would need custom implementation with model forward pass
    # For now, we'll measure sequential processing as baseline
    
    print("Note: Raw MLX generate doesn't support native batching.")
    print("Running sequential processing for comparison...")
    
    start_time = time.perf_counter()
    responses = []
    
    for prompt in prompts:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted_prompt = prompt
        
        response = generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens,
            verbose=False,
        )
        responses.append(response)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Estimate total tokens
    total_tokens = sum(len(r.split()) * 1.3 for r in responses)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return total_time, tokens_per_second, responses


def print_results(model_name: str, results: List[Tuple[float, float, str]], prompts: List[str]):
    """Print benchmark results."""
    
    print("\n" + "=" * 80)
    print(f"RESULTS FOR: {model_name}")
    print("=" * 80)
    
    for i, (time_taken, tps, response) in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i][:50]}...")
        print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
        print(f"Time: {time_taken:.2f}s")
        print(f"Tokens/sec: {tps:.1f}")
    
    # Aggregate stats
    times = [r[0] for r in results]
    tps_values = [r[1] for r in results]
    
    print("\n" + "-" * 40)
    print("AGGREGATE STATISTICS")
    print("-" * 40)
    print(f"Average time per request: {mean(times):.2f}s")
    print(f"Total time for {len(results)} requests: {sum(times):.2f}s")
    print(f"Average tokens/sec: {mean(tps_values):.1f}")
    
    if len(results) > 1:
        print(f"Std dev time: {stdev(times):.2f}s")
        print(f"Std dev tokens/sec: {stdev(tps_values):.1f}")
    
    # Calculate total throughput
    total_tokens = sum(r[1] * r[0] for r in results)  # tps * time = tokens
    total_time = sum(times)
    overall_throughput = total_tokens / total_time if total_time > 0 else 0
    print(f"Overall throughput: {overall_throughput:.1f} tokens/sec")


def main():
    parser = argparse.ArgumentParser(description="Benchmark raw MLX generate performance")
    parser.add_argument("--model-path", required=True, help="Path to MLX model")
    parser.add_argument("--max-tokens", default=50, type=int, help="Max tokens to generate")
    parser.add_argument("--num-requests", default=4, type=int, help="Number of test requests")
    parser.add_argument("--batch-test", action="store_true", help="Test batch processing")
    
    args = parser.parse_args()
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms",
        "Write a haiku about artificial intelligence",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis",
        "List three interesting facts about Mars",
        "How does machine learning differ from traditional programming?",
        "What is the importance of biodiversity?",
        "Explain the concept of blockchain technology"
    ][:args.num_requests]
    
    print(f"Loading model: {args.model_path}")
    print("This may take a moment...")
    
    # Load model and tokenizer
    start_load = time.perf_counter()
    model, tokenizer = load(args.model_path)
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")
    
    # Test 1: Sequential requests (how raw MLX normally works)
    print(f"Testing {args.num_requests} sequential requests...")
    print("=" * 80)
    
    seq_results = benchmark_sequential_requests(
        model, tokenizer, test_prompts, args.max_tokens
    )
    
    print_results(f"{args.model_path} (Sequential)", seq_results, test_prompts)
    
    # Test 2: Batch simulation (if requested)
    if args.batch_test:
        print("\n\nTesting batch simulation (sequential processing of batch)...")
        print("=" * 80)
        
        batch_time, batch_tps, batch_responses = benchmark_batch_requests(
            model, tokenizer, test_prompts, args.max_tokens
        )
        
        print(f"\nBatch simulation completed:")
        print(f"Total time for {len(test_prompts)} prompts: {batch_time:.2f}s")
        print(f"Overall throughput: {batch_tps:.1f} tokens/sec")
        print(f"Average time per prompt: {batch_time/len(test_prompts):.2f}s")
    
    # Comparison note
    print("\n" + "=" * 80)
    print("COMPARISON NOTES:")
    print("=" * 80)
    print("Raw MLX generate:")
    print("- Processes requests sequentially (no native batching)")
    print("- No concurrent request handling")
    print("- No continuous batching or dynamic scheduling")
    print("- No PagedKVCache for memory efficiency")
    print("\nMLX ParaLLM Server advantages:")
    print("- True concurrent request handling")
    print("- Dynamic batching (up to max_batch_size)")
    print("- Continuous batching with PagedKVCache")
    print("- OpenAI-compatible API")
    print("- Request queuing and scheduling")


if __name__ == "__main__":
    main()