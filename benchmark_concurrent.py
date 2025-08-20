#!/usr/bin/env python3
"""Benchmark concurrent request handling for MLX ParaLLM server."""

import asyncio
import time
import json
from typing import List, Dict, Any
import aiohttp
import argparse
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class BenchmarkResult:
    request_id: int
    prompt: str
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    total_time: float
    time_to_first_token: float
    tokens_per_second: float


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> BenchmarkResult:
    """Send a single request and measure timing."""
    
    payload = {
        "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",  # Will be overridden by args
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    response_text = ""
    
    try:
        if stream:
            # Streaming request
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content:
                    buffer += chunk.decode('utf-8')
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Keep incomplete line in buffer
                    
                    for line in lines[:-1]:
                        if line.startswith('data: '):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    response_text += delta.get('content', '')
                            except json.JSONDecodeError:
                                continue
        else:
            # Non-streaming request
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                first_token_time = time.perf_counter()  # First response is all tokens
                data = await response.json()
                response_text = data['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return BenchmarkResult(
            request_id=request_id,
            prompt=prompt,
            response_text=f"ERROR: {e}",
            prompt_tokens=0,
            completion_tokens=0,
            total_time=0,
            time_to_first_token=0,
            tokens_per_second=0
        )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    time_to_first = (first_token_time - start_time) if first_token_time else total_time
    
    # Rough token count (actual tokenizer would be more accurate)
    prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
    completion_tokens = len(response_text.split()) * 1.3
    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
    
    return BenchmarkResult(
        request_id=request_id,
        prompt=prompt,
        response_text=response_text,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_time=total_time,
        time_to_first_token=time_to_first,
        tokens_per_second=tokens_per_second
    )


async def run_concurrent_benchmark(
    url: str,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> List[BenchmarkResult]:
    """Run multiple requests concurrently."""
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, url, i, prompt, max_tokens, temperature, stream)
            for i, prompt in enumerate(prompts)
        ]
        
        print(f"Sending {len(tasks)} concurrent requests...")
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        print(f"All requests completed in {total_time:.2f} seconds\n")
        
        return results


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a readable format."""
    
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    for r in results:
        print(f"\nRequest {r.request_id}:")
        print(f"  Prompt: {r.prompt[:50]}..." if len(r.prompt) > 50 else f"  Prompt: {r.prompt}")
        print(f"  Response: {r.response_text[:100]}..." if len(r.response_text) > 100 else f"  Response: {r.response_text}")
        print(f"  Total time: {r.total_time:.2f}s")
        print(f"  Time to first token: {r.time_to_first_token:.3f}s")
        print(f"  Tokens/second: {r.tokens_per_second:.1f}")
        print(f"  Estimated tokens: {r.completion_tokens}")
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    
    valid_results = [r for r in results if r.total_time > 0]
    
    if valid_results:
        avg_time = mean([r.total_time for r in valid_results])
        avg_ttft = mean([r.time_to_first_token for r in valid_results])
        avg_tps = mean([r.tokens_per_second for r in valid_results])
        total_tokens = sum([r.completion_tokens for r in valid_results])
        
        print(f"Average total time: {avg_time:.2f}s")
        print(f"Average time to first token: {avg_ttft:.3f}s")
        print(f"Average tokens/second: {avg_tps:.1f}")
        print(f"Total tokens generated: {total_tokens}")
        
        if len(valid_results) > 1:
            std_time = stdev([r.total_time for r in valid_results])
            std_tps = stdev([r.tokens_per_second for r in valid_results])
            print(f"Std dev total time: {std_time:.2f}s")
            print(f"Std dev tokens/second: {std_tps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX ParaLLM server with concurrent requests")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument("--num-requests", default=4, type=int, help="Number of concurrent requests")
    parser.add_argument("--max-tokens", default=100, type=int, help="Max tokens per response")
    parser.add_argument("--temperature", default=0.7, type=float, help="Temperature for sampling")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--custom-prompts", nargs="+", help="Custom prompts to use")
    
    args = parser.parse_args()
    
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    
    # Default prompts for testing
    default_prompts = [
        "Explain quantum computing in simple terms",
        "Write a haiku about artificial intelligence",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis",
        "List three interesting facts about Mars",
        "How does machine learning differ from traditional programming?",
        "What is the importance of biodiversity?",
        "Explain the concept of blockchain technology"
    ]
    
    # Use custom prompts if provided, otherwise use defaults
    if args.custom_prompts:
        prompts = args.custom_prompts
    else:
        prompts = default_prompts[:args.num_requests]
    
    print(f"Running benchmark with {len(prompts)} concurrent requests")
    print(f"Server: {url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Streaming: {args.stream}")
    print()
    
    # Run benchmark
    results = asyncio.run(run_concurrent_benchmark(
        url, prompts, args.max_tokens, args.temperature, args.stream
    ))
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()