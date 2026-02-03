# System Patterns

## High-level architecture
- **FastAPI server**: `mlx_parallm/server/main.py` implements OpenAI-style endpoints and batching workers.
- **Global in-process registry**: `mlx_parallm/server/state.py` holds:
  - `model_registry`: `model_id -> InternalModelRecord` (model + tokenizer + status)
  - `weight_update_lock`: a shared lock for safe in-memory adapter updates
- **Model + tokenizer loading**: `mlx_parallm/utils.py` (`load`, `load_model`, `load_tokenizer`) loads models (local or HF) and constructs MLX modules.

## Batching patterns
- **Non-streaming batching**:
  - Requests enqueue into `REQUEST_QUEUE`.
  - `batch_processing_worker()` drains within `BATCH_TIMEOUT` up to `MAX_BATCH_SIZE`.
  - A simplified “first request sets batch params” strategy is used today.
  - Optional prompt de-duplication exists for identical prompts (disabled when `DIVERSE_MODE` or `n>1`).
- **Streaming batching (chat)**:
  - Streaming chat requests enqueue into `STREAM_CHAT_QUEUE`.
  - A worker co-batches streams and emits SSE chunks through per-request queues.
- **Continuous scheduler (experimental)**:
  - `continuous_scheduler_worker()` attempts admit-on-step scheduling combining streaming + non-streaming queues.

## KV cache patterns
- `mlx_parallm/models/base.py` provides:
  - `BatchedKVCache`: single uniform offset per batch
  - `PagedKVCache`: per-sequence offsets (independent decode progress)
- Model implementations (e.g. `mlx_parallm/models/llama.py`) apply RoPE and causal masks using per-row offsets where available.

## RL training patterns
- RL entrypoint: `mlx_parallm/rl_training/train.py` (`mlx_parallm_train`)
  - launches the server in-thread
  - initializes LoRA on quantized models if needed (`lora_init.py`)
  - runs GRPO-style updates (`grpo_trainer.py`)
  - hot-reloads adapter weights in-process (`weight_updater.py`)
- **Single-process shared-state contract**:
  - Training and serving must run in the same Python process for “same model instance” guarantees.
  - Adapter swaps/updates are synchronized with `weight_update_lock`.

