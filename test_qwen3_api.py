#!/usr/bin/env python3
"""Test script to verify Qwen3 works with mlx_parallm API endpoints."""

import asyncio
import aiohttp
import json
import sys
from typing import List, Dict, Any


async def test_completion(session: aiohttp.ClientSession, prompt: str, request_id: int) -> Dict[str, Any]:
    """Test a single completion request."""
    url = "http://localhost:8000/v1/completions"
    
    payload = {
        "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            if response.status == 200:
                text = result["choices"][0]["text"]
                print(f"Request {request_id} ✓: '{prompt}' -> '{text[:100]}...'")
                return result
            else:
                print(f"Request {request_id} ✗: Error {response.status}: {result}")
                return None
    except Exception as e:
        print(f"Request {request_id} ✗: Exception: {e}")
        return None


async def test_chat_completion(session: aiohttp.ClientSession, message: str, request_id: int) -> Dict[str, Any]:
    """Test a single chat completion request."""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            if response.status == 200:
                text = result["choices"][0]["message"]["content"]
                print(f"Chat {request_id} ✓: '{message}' -> '{text[:100]}...'")
                return result
            else:
                print(f"Chat {request_id} ✗: Error {response.status}: {result}")
                return None
    except Exception as e:
        print(f"Chat {request_id} ✗: Exception: {e}")
        return None


async def test_batched_requests():
    """Test multiple concurrent requests to verify batching works."""
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "Python programming language was created by",
        "The meaning of life is",
        "Artificial intelligence will",
        "The best way to learn coding is",
        "Climate change affects",
        "Space exploration helps us"
    ]
    
    test_messages = [
        "What is 2+2?",
        "Explain quantum physics briefly",
        "Write a haiku about coding",
        "What's the weather like?"
    ]
    
    print("Starting batched request test...")
    print(f"Testing with {len(test_prompts)} completion requests and {len(test_messages)} chat requests\n")
    
    async with aiohttp.ClientSession() as session:
        # First check if server is running
        try:
            async with session.get("http://localhost:8000/health") as response:
                if response.status != 200:
                    print("✗ Server not responding on http://localhost:8000")
                    print("  Please start the server with:")
                    print("  uv run mlx_parallm_serve --model-path NousResearch/Hermes-4-Qwen3-14B-1-e3 --port 8000")
                    return False
        except:
            print("✗ Cannot connect to server on http://localhost:8000")
            print("  Please start the server with:")
            print("  uv run mlx_parallm_serve --model-path NousResearch/Hermes-4-Qwen3-14B-1-e3 --port 8000")
            return False
        
        print("✓ Server is running\n")
        
        # Test completions in parallel
        print("=== Testing Completion Endpoint (Batched) ===")
        completion_tasks = [
            test_completion(session, prompt, i) 
            for i, prompt in enumerate(test_prompts, 1)
        ]
        
        start_time = asyncio.get_event_loop().time()
        completion_results = await asyncio.gather(*completion_tasks)
        completion_time = asyncio.get_event_loop().time() - start_time
        
        successful_completions = sum(1 for r in completion_results if r is not None)
        print(f"\n✓ Completions: {successful_completions}/{len(test_prompts)} successful")
        print(f"  Time: {completion_time:.2f}s ({completion_time/len(test_prompts):.2f}s per request avg)")
        
        # Test chat completions in parallel
        print("\n=== Testing Chat Completion Endpoint (Batched) ===")
        chat_tasks = [
            test_chat_completion(session, message, i)
            for i, message in enumerate(test_messages, 1)
        ]
        
        start_time = asyncio.get_event_loop().time()
        chat_results = await asyncio.gather(*chat_tasks)
        chat_time = asyncio.get_event_loop().time() - start_time
        
        successful_chats = sum(1 for r in chat_results if r is not None)
        print(f"\n✓ Chat Completions: {successful_chats}/{len(test_messages)} successful")
        print(f"  Time: {chat_time:.2f}s ({chat_time/len(test_messages):.2f}s per request avg)")
        
        # Summary
        print("\n=== Summary ===")
        total_requests = len(test_prompts) + len(test_messages)
        total_successful = successful_completions + successful_chats
        
        if total_successful == total_requests:
            print(f"✓ All {total_requests} requests completed successfully!")
            print(f"  Total time: {completion_time + chat_time:.2f}s")
            print(f"  Batching is working correctly for Qwen3 model")
            return True
        else:
            print(f"✗ {total_requests - total_successful} requests failed")
            return False


async def main():
    """Main test function."""
    success = await test_batched_requests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)