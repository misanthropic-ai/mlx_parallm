# Active Context

## Current focus (Feb 3, 2026)
- Re-initializing project context and documentation.
- Capturing a concrete backlog for:
  - “real” continuous batching + paged KV (vLLM-like behavior)
  - robust LoRA/DoRA lifecycle management
  - stable RL rollout + update loop on a shared model instance

## Noted gaps during repo review
- No maintained pytest suite; existing ad-hoc tests/scripts are stale and/or hard-coded to local paths.
- `mlx_parallm/server/main.py` is large and mixes concerns (routing, batching, streaming, metrics, scheduling).
- Several pieces are “smoke” implementations (e.g., GRPO trainer uses Python loops for logp gathering).

## Next steps (high-value)
1. Establish a minimal `pytest` suite (`tests/`) for `/health`, a small completion, and a small RL smoke step on a tiny model.
2. Refactor server into modules (routes vs batching vs streaming vs scheduler vs metrics) to reduce coupling.
3. Define the target design for “paged KV + continuous batching” (slot manager, block allocator, eviction/compaction).
4. Expand adapter support: DoRA support plan, adapter versioning, safe hot-swap semantics.

