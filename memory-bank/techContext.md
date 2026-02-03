# Tech Context

## Languages and runtime
- Python 3.10+
- MLX (`mlx`) for model execution on Apple Silicon
- `mlx-lm` for tokenizer utilities and adapter tooling

## Key dependencies
- Server: FastAPI, Starlette, Uvicorn
- HF model access: `huggingface_hub`, `transformers`, `safetensors`
- RL/rollouts: optional `aiohttp` (dev extra) for Atropos API client; `requests` for mock client

## Dev workflow
- Env:
  - `uv venv && source .venv/bin/activate`
  - `uv pip install -e .`
- Run server:
  - `mlx_parallm_serve --model-path <hf_id_or_path> --port 8000`
  - Health: `curl http://127.0.0.1:8000/health`
- Run RL smoke:
  - `uv run mlx_parallm_train --model-path <path_or_hf_id> --steps 3 --batch-size 2 --checkpoint-dir checkpoints`

## Operational constraints (important)
- Unified memory is large but finite; “double-loading” models (inference+training) is expensive.
- Adapter training is the practical near-term path; full-weight training is currently not a target.
- HF download/auth is required for remote models (`HF_TOKEN` or `huggingface-cli login`).

