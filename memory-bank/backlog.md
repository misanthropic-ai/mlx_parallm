# Backlog (Initial)

This is a prioritized list of “things that need work” found during repo review.

## P0 (unblock reliability)
- De-duplicate `GET /debug/metrics` in `mlx_parallm/server/main.py` (duplicate route).
- Establish a minimal `pytest` suite under `tests/` and make it runnable with `uv run pytest`.
- Replace/retire stale test scripts (`test_mock_quick.py`, `test_integration.py`) or turn them into real tests.

## P1 (architecture and maintainability)
- Split `mlx_parallm/server/main.py` into:
  - routers (API surface)
  - batching/scheduler engine
  - streaming engine
  - metrics/telemetry
  - shared state/config
- Consolidate config: one settings object (CLI/env/file) instead of scattered globals.

## P1 (performance)
- Vectorize GRPO logprob gathering (avoid Python loops; use gather/take-along-axis patterns).
- Reduce per-request overhead in batching worker:
  - move parameter consolidation into per-request state where possible
  - make dedup / “diverse mode” behavior explicit and testable

## P2 (feature completeness)
- DoRA support plan and implementation (adapter injection, save/load, merge).
- Robust multi-adapter versioning and rollback for online updates.
- True “paged KV” allocator (block-based) to support long contexts efficiently.

