# Repo Map

## Top-level
- `README.md`: usage and API examples
- `INFERENCE_SERVER.md`: design notes for batching/KV/RL integration
- `RL_TRAINER.md`: GRPO + on-policy rollout design notes and usage
- `LORA_TRAINING.md`: LoRA init notes (quantized models) and mitigations
- `PROJECT.md`: work tracker / backlog

## Python package: `mlx_parallm/`
- `mlx_parallm/cli.py`: `mlx_parallm_serve` entrypoint (FastAPI via Uvicorn)
- `mlx_parallm/server/`
  - `main.py`: FastAPI app + batching workers + streaming workers + scheduler
  - `schemas.py`: Pydantic request/response schemas
  - `state.py`: `model_registry` and `weight_update_lock`
- `mlx_parallm/utils.py`: model/tokenizer loading + generation utilities + KV/prefix caches
- `mlx_parallm/models/`: MLX model implementations and KV cache primitives
- `mlx_parallm/rl_training/`: RL trainer, LoRA init, adapter hot-reload, Atropos/mock clients
- `mlx_parallm/tools/`: adapter conversion/merge/checkpoint utilities

## Other
- `models/`: example local model dirs (converted/quantized artifacts)
- `reference/`: notes/config references (`reference/rlx/`)

