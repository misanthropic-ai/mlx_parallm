# PROJECT.md

This file tracks the current work backlog for `mlx_parallm`. Update it as tasks are completed (and link/close GitHub issues when available).

## Now (P0)
- [x] Remove duplicate `GET /debug/metrics` route in `mlx_parallm/server/main.py`
- [x] Add minimal test suite (`tests/`) runnable locally (`unittest`):
  - [x] `/health` returns `{status: "ok"}`
  - [x] Single `/v1/completions` request succeeds
  - [x] Concurrent `/v1/completions` updates `/debug/metrics` (batching sanity)
  - [x] `/v1/chat/completions` supports `n=2`
  - [x] Streaming `/v1/chat/completions` ends with `[DONE]`
  - [x] Server starts with a preloaded LoRA adapter
- [ ] Replace stale scripts with runnable tests or mark them deprecated:
  - [ ] `test_mock_quick.py`
  - [ ] `test_integration.py`

## Next (P1)
- [ ] Refactor server layout (reduce `mlx_parallm/server/main.py` size and coupling)
- [ ] Formalize scheduler design (default vs continuous) and add tests for admission + fairness
- [ ] Improve GRPO performance (vectorized logp, avoid Python loops)
- [ ] Adapter lifecycle improvements:
  - [ ] DoRA support plan
  - [ ] adapter versioning + safe hot-reload semantics

## Later (P2)
- [ ] True paged KV allocator (block-based) + eviction/compaction strategy
- [ ] Multi-model loading and “available_not_loaded” discovery
- [ ] Observability: richer metrics, optional Prometheus endpoint
