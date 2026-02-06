from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import requests


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(base_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.25)
    raise RuntimeError(f"Server did not become healthy within {timeout_s}s (last_err={last_err})")


def build_tiny_model(dst_dir: Path, *, tokenizer_src: Optional[Path] = None) -> Path:
    tokenizer_src = tokenizer_src or (repo_root() / "models" / "hermes-qwen3-14b-4bit")
    if not tokenizer_src.exists():
        raise RuntimeError(f"tokenizer_src missing: {tokenizer_src}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "build_tiny_model.py"),
        "--tokenizer-src",
        str(tokenizer_src),
        "--dst",
        str(dst_dir),
        "--hidden-size",
        "64",
        "--layers",
        "8",
        "--heads",
        "4",
        "--kv-heads",
        "4",
        "--intermediate-size",
        "128",
        "--q-bits",
        "4",
        "--q-group-size",
        "64",
    ]
    subprocess.run(cmd, cwd=str(repo_root()), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Sanity check artifacts
    if not (dst_dir / "config.json").exists():
        raise RuntimeError("tiny model build missing config.json")
    if not any(dst_dir.glob("model*.safetensors")):
        raise RuntimeError("tiny model build missing model*.safetensors")
    return dst_dir


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    base_url: str
    log_path: Path


def start_server(
    *,
    model_path: Path,
    port: Optional[int] = None,
    lora_path: Optional[Path] = None,
    batch_timeout: float = 0.05,
    max_batch_size: int = 8,
    scheduler: str = "default",
    diverse_mode: bool = True,
    extra_args: Optional[Sequence[str]] = None,
    log_path: Optional[Path] = None,
) -> ServerHandle:
    port = port or find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    if log_path is None:
        log_dir = Path(tempfile.mkdtemp(prefix=f"mlx_parallm_server_{port}_"))
        log_path = log_dir / "server.log"

    cli_args = [
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--scheduler",
        scheduler,
        "--max-batch-size",
        str(max_batch_size),
        "--batch-timeout",
        str(batch_timeout),
        "--request-timeout-seconds",
        "86400",
        "--diverse-mode",
        "true" if diverse_mode else "false",
    ]
    if os.getenv("MLX_PARALLM_COVERAGE") in ("1", "true", "yes", "on"):
        args = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source",
            "mlx_parallm",
            "-m",
            "mlx_parallm.cli",
            *cli_args,
        ]
    else:
        args = [sys.executable, "-m", "mlx_parallm.cli", *cli_args]
    if lora_path is not None:
        args += ["--lora-path", str(lora_path)]
    if extra_args:
        args += list(extra_args)

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env["MLX_PARALLM_MODEL"] = str(model_path)
    env["MLX_PARALLM_SCHEDULER"] = str(scheduler)

    with open(log_path, "w", buffering=1) as log_f:
        proc = subprocess.Popen(args, cwd=str(repo_root()), stdout=log_f, stderr=subprocess.STDOUT, text=True, env=env)

    wait_for_health(base_url, timeout_s=180.0)
    return ServerHandle(proc=proc, base_url=base_url, log_path=log_path)


def stop_server(h: ServerHandle) -> None:
    if h.proc.poll() is not None:
        return
    # If the server was started under `coverage run`, SIGTERM can end up killing the
    # coverage runner before it writes its data file. SIGINT typically yields a
    # cleaner shutdown path.
    is_coverage = False
    try:
        args = h.proc.args
        if isinstance(args, (list, tuple)) and "coverage" in [str(a) for a in args]:
            is_coverage = True
    except Exception:
        is_coverage = False
    try:
        h.proc.send_signal(signal.SIGINT if is_coverage else signal.SIGTERM)
        h.proc.wait(timeout=30 if is_coverage else 10)
    except Exception:
        try:
            h.proc.kill()
        except Exception:
            pass
        try:
            h.proc.wait(timeout=5)
        except Exception:
            pass


def get_metrics(base_url: str) -> dict:
    r = requests.get(f"{base_url}/debug/metrics", timeout=5)
    r.raise_for_status()
    return json.loads(r.text)
