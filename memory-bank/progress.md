# Progress

## What works today
- FastAPI server starts (loads model at startup when `--model-path` provided) and exposes:
  - `/health`
  - `/v1/models`
  - `/v1/completions` (batching for non-stream; streaming supported for `n=1`)
  - `/v1/chat/completions` (non-stream via batch queue; stream via co-batched SSE queue)
  - `/v1/perplexity`
- KV cache abstractions exist (`BatchedKVCache`, `PagedKVCache`) and models support per-row offsets in attention masks/rope.
- RL training “smoke” loop exists:
  - can launch server, collect mock rollouts, run GRPO-like updates, and hot-reload adapter weights.
- LoRA init for quantized models includes a mitigation for “garbled outputs” (zero-init `lora_b`).

## Known issues / tech debt
- Duplicate route definition: `GET /debug/metrics` is declared twice in `mlx_parallm/server/main.py` and should be de-duplicated.
- Test scripts are stale:
  - `test_mock_quick.py` does not match current `MockAtroposClient` constructor.
  - `test_integration.py` hard-codes absolute paths and is not a portable test harness.
- “Continuous batching” and “paged KV” are present as prototypes but not at parity with vLLM/SGLang scheduling/allocator designs.
- GRPO implementation uses Python loops (one-hot building, per-token logp) and will not scale.

## What’s left to build (roadmap-level)
- Production-grade scheduler:
  - slot manager, admit-on-token decode loop, efficient per-sequence KV growth, prefix sharing, eviction.
- Adapter lifecycle:
  - DoRA support, adapter metadata/versioning, safe hot-swap protocol, merge/export tooling.
- RL training:
  - efficient logprob computation, proper group handling, scalable batching, improved reference-policy coupling.
- Observability:
  - stable metrics endpoint, better queue/latency/tokens-per-sec reporting, optional Prometheus export.

