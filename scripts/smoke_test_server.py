#!/usr/bin/env python3
"""
Smoke test script for MLX ParaLLM inference server.

Launches the server as a subprocess, waits for health, runs a few
endpoint checks (models, completion, chat, streaming), prints concise
summaries, then cleans up the server process.

Usage (examples):
  uv run python scripts/smoke_test_server.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit --scheduler continuous --port 8000

  uv run python scripts/smoke_test_server.py \
    --model NousResearch/Hermes-4-Qwen3-14B-1-e3 --scheduler continuous --port 8001
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Optional


def http_get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read().decode()


def http_post_json(url: str, payload: dict, timeout: float = 180.0) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read().decode()


def wait_for_health(base: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = http_get(f"{base}/health", timeout=5.0)
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def launch_server(
    *,
    model: str,
    host: str,
    port: int,
    scheduler: str,
    max_batch_size: Optional[int],
    batch_timeout: Optional[float],
    max_concurrent_streams: Optional[int],
    log_path: str,
) -> subprocess.Popen:
    args = [
        sys.executable,
        "-m",
        "mlx_parallm.cli",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--scheduler",
        scheduler,
    ]
    if max_batch_size is not None:
        args += ["--max-batch-size", str(max_batch_size)]
    if batch_timeout is not None:
        args += ["--batch-timeout", str(batch_timeout)]
    if max_concurrent_streams is not None:
        args += ["--max-concurrent-streams", str(max_concurrent_streams)]

    # Ensure model is visible to the app even if CLI globals are not shared across processes
    env = os.environ.copy()
    env["MLX_PARALLM_MODEL"] = model
    env.setdefault("MLX_PARALLM_SCHEDULER", scheduler)
    log_f = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, text=True, env=env)
    return proc


def tail_file(path: str, n: int = 80) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception as e:
        return f"<failed to read {path}: {e}>"


def try_stream_chat(base: str, model: str, timeout: float = 180.0, max_lines: int = 8) -> list[str]:
    body = json.dumps(
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": "In one sentence, describe the ocean."}],
            "max_tokens": 24,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    lines: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for i, raw in enumerate(resp):
            if i >= max_lines:
                break
            try:
                lines.append(raw.decode(errors="ignore").rstrip())
            except Exception:
                pass
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch server and run smoke tests")
    ap.add_argument("--model", required=True, help="HF id or local model path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scheduler", default="continuous", choices=["default", "continuous"]) 
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--batch-timeout", type=float, default=0.05)
    ap.add_argument("--max-concurrent-streams", type=int, default=4)
    ap.add_argument("--no-kill", action="store_true", help="Leave server running after tests")
    ap.add_argument("--log", default="server_smoke.log")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"Launching server on {base} with model={args.model} scheduler={args.scheduler}")

    # Launch server
    proc = launch_server(
        model=args.model,
        host=args.host,
        port=args.port,
        scheduler=args.scheduler,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
        max_concurrent_streams=args.max_concurrent_streams,
        log_path=args.log,
    )

    # Wait for health
    print("Waiting for health...")
    if not wait_for_health(base, timeout=180.0):
        print("ERROR: health check timed out. Tail of log:")
        print(tail_file(args.log, 160))
        if not args.no_kill:
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return 1

    print("Health: ok")

    # Models
    try:
        code, txt = http_get(f"{base}/v1/models", timeout=10.0)
        print("Models status:", code)
        print(txt)
    except Exception as e:
        print("Models check failed:", e)

    # Completion (non-stream)
    try:
        code, txt = http_post_json(
            f"{base}/v1/completions",
            {
                "model": args.model,
                "prompt": "Write a haiku about sunrise.",
                "max_tokens": 16,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            timeout=300.0,
        )
        print("Completions status:", code)
        try:
            j = json.loads(txt)
            ch = j.get("choices", [])
            snippet = (ch[0].get("text") or ch[0].get("message", {}).get("content", ""))[:200] if ch else ""
            print("Completion snippet:", snippet)
        except Exception:
            print("Completion raw:", txt[:300])
    except Exception as e:
        print("Completions error:", e)

    # Chat (non-stream)
    try:
        code, txt = http_post_json(
            f"{base}/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Suggest 3 creative team names."}],
                "max_tokens": 32,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            timeout=300.0,
        )
        print("Chat status:", code)
        try:
            j = json.loads(txt)
            ch = j.get("choices", [])
            content = (ch[0].get("message", {}).get("content") or "")[:200] if ch else ""
            print("Chat snippet:", content)
        except Exception:
            print("Chat raw:", txt[:300])
    except Exception as e:
        print("Chat error:", e)

    # Chat (streaming) — first few lines
    try:
        lines = try_stream_chat(base, args.model, timeout=300.0, max_lines=8)
        print("Streaming (first lines):")
        for ln in lines:
            print(ln)
    except Exception as e:
        print("Streaming error:", e)

    # Metrics
    try:
        code, txt = http_get(f"{base}/debug/metrics", timeout=10.0)
        print("Metrics status:", code)
        print(txt)
    except Exception as e:
        print("Metrics error:", e)

    # Cleanup
    if args.no_kill:
        print(f"Leaving server running (pid={proc.pid}). Logs: {args.log}")
        return 0

    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    except Exception:
        pass
    print("Server terminated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
