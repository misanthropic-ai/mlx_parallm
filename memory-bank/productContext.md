# Product Context

## Why this exists
Apple Silicon Macs have strong local inference characteristics (Metal + unified memory), but typical stacks (vLLM/SGLang) target CUDA. This project aims to provide a **viable local inference server** built on MLX while also enabling **reinforcement learning with online updates** without running two separate model copies.

## What problems it solves
- **Local serving**: run an OpenAI-compatible server for apps/tools without a remote GPU.
- **Online RL**: do rollouts and backprop on the *same* model instance to:
  - reduce memory pressure (critical on Macs)
  - avoid reload/serialization overhead between inference and training
  - keep rollouts on-policy by hot-updating adapter weights
- **Adapter-first training**: LoRA/DoRA allow meaningful finetuning under memory constraints.

## Intended user experience
- Start server with a single command (`mlx_parallm_serve --model-path ...`).
- Optional: start RL loop (`mlx_parallm_train ...`) which launches the server and performs rollout→update cycles.
- Swap adapters (LoRA/DoRA) quickly without restarting.

