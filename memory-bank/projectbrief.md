# Project Brief: MLX ParaLLM

## One-liner
High-performance local LLM inference server on Apple Silicon (MLX) with continuous batching + KV caching, designed to share a single model instance with an RL trainer for on-policy rollouts and in-memory LoRA/DoRA updates.

## Core goals
- OpenAI-compatible HTTP API for local inference (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`-style future).
- Competitive throughput/latency on Apple Silicon via dynamic batching and KV cache reuse.
- RL rollouts + backprop **on the same in-process model instance** (no reloads, minimal memory overhead).
- First-class adapter training + hot-swapping: LoRA now, extend to DoRA and higher-level adapter management.

## Non-goals (for now)
- Multi-node/distributed inference (Mac cluster) beyond exploratory scaffolding.
- Full-weight training of large models (unified memory helps, but this project currently targets adapter training).
- Full parity with vLLM/SGLang feature surface (tools/function calling, speculative decoding, etc.) without prioritization.

## Primary users
- Researchers/engineers running local inference on Mac (unified memory) who want:
  - fast OpenAI-compatible serving for apps
  - online/on-policy RL experiments without duplicating model memory

## Success criteria
- Server runs reliably for non-streaming + streaming requests with batching enabled.
- RL trainer can (a) collect rollouts through the server, (b) update adapter weights, and (c) immediately produce on-policy rollouts with the updated policy.
- LoRA/DoRA workflows are reproducible: init, save/load, hot-reload, merge/export.

