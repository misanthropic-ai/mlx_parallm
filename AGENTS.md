# Repository Guidelines

## Project Structure & Module Organization
- `mlx_parallm/`: Core package.
  - `cli.py`: Entry point for the server (`mlx_parallm_serve`).
  - `server/`: FastAPI app (`main.py`) and pydantic schemas.
  - `models/`, `memory/`: Model backends and memory utilities.
  - `utils.py`: Load/generate helpers, batching, HF Hub utilities.
- Tests: `test_*.py` in repo root (smoke/integration-style today).
- Examples: `demo.py`, `demo_extended_mind.py`, `demo.ipynb`.
- Config: `pyproject.toml` (project + deps), `uv.lock`.

## Build, Test, and Development Commands
- Create env and install (uv):
  ```bash
  uv venv && source .venv/bin/activate
  uv pip install -e .
  ```
- Run server locally:
  ```bash
  mlx_parallm_serve --model-path <hf_id_or_local_path> --host 127.0.0.1 --port 8000
  # e.g. mlx_parallm_serve --model-path mlx-community/Llama-3.2-3B-Instruct-4bit
  ```
- Run tests:
  ```bash
  pytest -q         # if pytest is available
  python test_extended_mind.py
  ```

## Coding Style & Naming Conventions
- Python ≥ 3.10, PEP 8, 4-space indentation, type hints required for new code.
- Names: Modules/functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Use `logging` (module-level loggers) instead of `print` in library/server code.
- Keep public functions documented with concise docstrings; prefer small, focused modules.

## Testing Guidelines
- Prefer `pytest`; place new tests under `tests/` or add `test_*.py` at repo root.
- Add assertions and cover both success and error paths for CLI, server routes, and `utils`.
- Avoid network in unit tests; for HF Hub access require `HF_TOKEN` and mark as integration.
- Quick checks: run `python test_extended_mind.py` and `test_debug_shapes.py` locally.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (≤72 chars). Examples: "fix attention shape", "add batch streaming".
- PRs: clear description, reproduction steps/commands (CLI or `curl`), linked issues, API changes noted, and logs/screenshots where useful.
- Ensure formatting/lint passes locally before requesting review.

## Security & Configuration Tips
- Never commit secrets. Set `HF_TOKEN` in your shell (`export HF_TOKEN=...`).
- Large models use significant disk/RAM; prefer 4-bit repos for quick tests.
- Primary target is Apple Silicon (MLX); verify on macOS when changing low-level inference paths.

