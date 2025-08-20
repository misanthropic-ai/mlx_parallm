import asyncio
import json
import time
import statistics as stats
from dataclasses import dataclass
from typing import List, Optional

import aiohttp


@dataclass
class Result:
    ok: bool
    status: int
    latency_s: float
    error: Optional[str] = None


async def worker(session: aiohttp.ClientSession, url: str, model: str, prompt: str, max_tokens: int, streamed: bool) -> Result:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": streamed,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=None) as resp:
            if streamed:
                async for _ in resp.content:
                    pass
            else:
                await resp.text()
            dt = time.perf_counter() - t0
            return Result(ok=200 <= resp.status < 300, status=resp.status, latency_s=dt, error=None)
    except Exception as e:
        dt = time.perf_counter() - t0
        return Result(ok=False, status=0, latency_s=dt, error=str(e))


async def run_load(base_url: str, model: str, *, concurrency: int, requests: int, max_tokens: int, streamed: bool, endpoint: str = "/v1/completions"):
    url = f"{base_url}{endpoint}"
    results: List[Result] = []
    sem = asyncio.Semaphore(concurrency)

    async def run_one(i: int):
        async with sem:
            async with aiohttp.ClientSession() as session:
                r = await worker(session, url, model, f"Load test prompt #{i}.", max_tokens, streamed)
                results.append(r)

    await asyncio.gather(*[run_one(i) for i in range(requests)])

    oks = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]
    lat = [r.latency_s for r in oks]
    print("--- Load Test Summary ---")
    print(f"Endpoint: {endpoint} | Stream: {streamed}")
    print(f"Model: {model}")
    print(f"Requests: {len(results)} | Concurrency: {concurrency}")
    print(f"Success: {len(oks)} | Errors: {len(errs)}")
    if lat:
        print(f"Latency p50: {stats.median(lat):.2f}s | p95: {stats.quantiles(lat, n=20)[18]:.2f}s | max: {max(lat):.2f}s")
        print(f"Throughput (approx): {len(oks) / sum(lat):.2f} req/s (sum over successful reqs)")
    if errs:
        from collections import Counter
        print("Errors by status:")
        st = Counter([e.status for e in errs])
        print(st)
        e0 = errs[0]
        print(f"Sample error: status={e0.status} error={e0.error}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Simple load test for mlx_parallm inference server")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    p.add_argument("--model", required=True, help="Model id as known by server (usually the --model-path value)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()

    asyncio.run(run_load(args.base_url, args.model, concurrency=args.concurrency, requests=args.requests, max_tokens=args.max_tokens, streamed=args.stream))

