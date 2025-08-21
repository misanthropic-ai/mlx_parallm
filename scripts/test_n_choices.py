#!/usr/bin/env python3
"""
Test that the server returns `n` choices for /v1/completions and /v1/chat/completions.

Actions:
- Best-effort kill any running server matching port or process name
- Launch server as subprocess via `uv run mlx_parallm_serve ...`
- Poll /health until ready
- Send requests with n=2 and print results
- Terminate server at the end
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


def kill_existing(port: int) -> None:
    cmds = [
        ["bash", "-lc", f"lsof -ti tcp:{port} | xargs -r kill -9"],
        ["bash", "-lc", "pkill -f mlx_parallm_serve || true"],
        ["bash", "-lc", "pkill -f uvicorn || true"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass


def wait_health(base_url: str, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.ok:
                print("[OK] Health:", r.text)
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Server health not ready within timeout")


def request_n2(base_url: str, model: str) -> None:
    # /v1/completions
    comp_payload = {
        "model": model,
        "prompt": "What is 2+2? Answer briefly.",
        "max_tokens": 8,
        "temperature": 0.9,
        "n": 2,
    }
    r = requests.post(f"{base_url}/v1/completions", json=comp_payload, timeout=30)
    print("/v1/completions status:", r.status_code)
    print(r.text)
    try:
        data = r.json()
        ch = data.get("choices", [])
        print("completions choices_len=", len(ch))
        for i, c in enumerate(ch):
            print(f"  choice[{i}]=", (c.get("text") or "").strip())
    except Exception:
        pass

    # /v1/chat/completions
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        "max_tokens": 8,
        "temperature": 0.9,
        "n": 2,
    }
    r2 = requests.post(f"{base_url}/v1/chat/completions", json=chat_payload, timeout=30)
    print("/v1/chat/completions status:", r2.status_code)
    print(r2.text)
    try:
        data2 = r2.json()
        ch2 = data2.get("choices", [])
        print("chat choices_len=", len(ch2))
        for i, c in enumerate(ch2):
            m = c.get("message", {})
            print(f"  choice[{i}]=", (m.get("content") or "").strip())
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/hermes-qwen3-14b-4bit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scheduler", default="continuous")
    ap.add_argument("--diverse", action="store_true", default=True)
    ap.add_argument("--logs", default="logs_test")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    Path(args.logs).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.logs) / "server_stdout.log"
    err_path = Path(args.logs) / "server_stderr.log"

    print("[Step] Killing existing server processes (best effort)…")
    kill_existing(args.port)

    cmd = [
        "uv",
        "run",
        "mlx_parallm_serve",
        "--model-path",
        args.model,
        "--scheduler",
        args.scheduler,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--diverse-mode",
        "true" if args.diverse else "false",
    ]
    print("[Step] Launching:", " ".join(cmd))

    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        try:
            print("[Step] Waiting for health…")
            wait_health(base_url, timeout=90)
            print("[Step] Querying with n=2…")
            request_n2(base_url, args.model)
        finally:
            print("[Step] Terminating server…")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    print(f"[Info] Server logs written to: {out_path} and {err_path}")


if __name__ == "__main__":
    main()

