# Repository Guidelines

## Project Structure & Modules
- Source: `mlx_parallm/`
  - `server/`: FastAPI app, request schemas, batching worker.
  - `models/`: MLX model adapters (llama, qwen3, gemma, mixtral, phi3).
  - `cli.py`: Entrypoint for `mlx_parallm_serve`.
  - `utils.py`, `sample_utils.py`: loading, generation, sampling utilities.
- Examples: `demo.py`, `demo.ipynb`.
- References: `reference/rlx/` (external notes/configs).

## Build, Run, Test
- Create env (uv):
  - `uv venv && source .venv/bin/activate`
- Install editable:
  - `uv pip install -e .`
- Run server (OpenAI‑compatible endpoints):
  - `mlx_parallm_serve --model-path <hf_id_or_path> --port 8000`
  - Health check: `curl http://127.0.0.1:8000/health`
- Quick completions:
  - `curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"<hf_id>","prompt":"Hello","max_tokens":16}'`
- Tests: No suite yet. Prefer `pytest` (future `tests/` with `test_*.py`).

## Coding Style & Naming
- Python 3.10+: type hints required; use Pydantic models in `server/schemas.py`.
- Indentation 4 spaces; keep functions small and pure where possible.
- Naming: `snake_case` for functions/vars/modules; `CamelCase` for classes; constants `UPPER_SNAKE`.
- Imports: standard → third‑party → local; avoid wildcard imports.
- Formatting: no enforced tool in repo; if available, run `black` and `ruff` before PRs.

## Testing Guidelines
- Add unit tests alongside features under `tests/` using `pytest`.
- Name files `test_<module>.py`; use fixtures for model/tokenizer if heavy.
- Minimum: add an integration check for `/health` and a small completion with a tiny model.

## Commits & Pull Requests
- Commits: imperative, concise, scoped (e.g., "add logprobs endpoint", "qwen: fix sampling").
- PRs must include:
  - Purpose and summary of changes.
  - Run instructions and sample `curl` for affected endpoints.
  - Any config changes (env vars, ports) and docs updates.
  - Follow‑ups or limitations; link related issues.

## Security & Configuration
- Models from Hugging Face: authenticate via `huggingface-cli login` or set `HF_TOKEN`.
- Do not commit secrets; prefer environment variables or a local `.env` (loaded by uvicorn[standard]).
- Large models: document expected RAM and quantization used.

## Running Python scripts
- Always use `uv run` to execute python scripts, models, tests
- `uv` will automatically use the correct virtual environment
