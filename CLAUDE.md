# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx_parallm is a high-performance, parallelized batch generation server for MLX models designed for Apple Silicon. It provides an OpenAI-compatible API with support for batched inference, Extended Mind Transformers, and future RL/embedding capabilities.

## Build and Development Commands

### Installation
```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install in development mode
uv pip install -e .

# Add runtime dependencies
uv add <package_name>

# Add development dependencies
uv add <package_name> --dev
```

### Running the Server
```bash
# Start server
mlx_parallm_serve --model-path <model_path> --host <host> --port <port>

# Example with quantized model
mlx_parallm_serve --model-path mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# With Extended Mind support
mlx_parallm_serve --model-path mlx-community/Llama-3.2-3B-Instruct-4bit --use-extended-mind --port 8000
```

### Testing
```bash
# No test framework currently configured - check README or pyproject.toml for testing approach
# Extended Mind testing script
python scripts/test_extended_mind_variants.py --model <model_path>
```

## High-Level Architecture

### Request Flow
1. **API Layer** (`server/main.py`): FastAPI endpoints receive requests
2. **Request Queue**: Non-streaming requests queued via `asyncio.Queue`
3. **Batch Worker** (`batch_processing_worker`): Background task collects requests into batches
4. **Generation Engine** (`utils.py`): MLX-based generation with `BatchedKVCache`
5. **Response Distribution**: Results sent back through futures to original requests

### Core Components

#### BatchedKVCache (`models/base.py`)
Custom KV cache replacing standard `mlx_lm.KVCache` for parallel batch processing. Handles dynamic batching with configurable step sizes and efficient memory management.

#### Generation Functions (`utils.py`)
- `batch_generate()`: Synchronous batch generation
- `stream_generate()`: Single-prompt streaming
- `batch_stream_generate_text()`: Batched streaming generation
- `generate_step()`: Core token generation with sampling

#### Request Batching System
- `QueuedRequest` class manages async request lifecycle
- `batch_processing_worker()` collects requests (timeout: 0.1s, max batch: 8)
- Handles `n` parameter by replicating prompts within batch
- Distributes results back to original request futures

#### Extended Mind Transformers (`models/llama_extended.py`)
- `ExtendedAttention`: Memory-augmented attention mechanism
- `MemoryManager` (`memory/manager.py`): Coordinates memory backends
- FAISS backend for vector similarity search
- Per-head memory indexing for grouped query attention

### Key Configuration
- `MAX_BATCH_SIZE = 8`: Maximum requests per batch
- `BATCH_TIMEOUT = 0.1`: Seconds to wait for batch formation
- Model registry uses model path as ID by default
- Memory parameters: `memory_topk`, `mask_by_sim`, `sim_threshold`

## API Endpoints

### Core Endpoints
- `GET /health`: Health check
- `GET /v1/models`: List loaded models
- `POST /v1/completions`: Text completions (OpenAI-compatible)
  - Supports: `prompt`, `max_tokens`, `temperature`, `top_p`, `stream`, `n`, `logprobs`, `echo`
- `POST /v1/chat/completions`: Chat completions (OpenAI-compatible)
  - Applies tokenizer chat template automatically
  - Supports: `messages`, `max_tokens`, `temperature`, `top_p`, `stream`, `n`

### Memory Management (Planned)
- `POST /v1/models/{model_id}/memories`: Add memories
- `GET /v1/models/{model_id}/memories`: List memories
- `DELETE /v1/models/{model_id}/memories`: Clear memories

## Extended Mind Status

### Working
- ✅ Core architecture implemented
- ✅ Memory backend system (FAISS, manual)
- ✅ Model loading with `--use-extended-mind`
- ✅ Memory addition and retrieval
- ✅ RoPE handling for Llama 2/3 variants

### Issues
- ⚠️ Generation with memories produces incorrect output (exclamation marks)
- Main issue: BatchedKVCache dimension mismatch between memory and cached values
- Memory values: `(B, n_heads, L * topk, head_dim)` for current queries
- Cached values: `(B, n_heads, total_cached_seq_len, head_dim)`

### Next Steps
1. Fix generation issue with memory-augmented attention
2. Add API endpoints for memory management
3. Implement other backends (Redis, Neo4j, SQL)
4. Add memory parameters to request schemas

## Code Organization

```
mlx_parallm/
├── cli.py                 # CLI entry point and argument parsing
├── server/
│   ├── main.py           # FastAPI app and request handling
│   └── schemas.py        # Pydantic models for API
├── models/
│   ├── base.py           # BatchedKVCache implementation
│   ├── llama.py          # Standard Llama architecture
│   └── llama_extended.py # Extended Mind Llama variant
├── memory/
│   ├── manager.py        # Memory backend coordination
│   ├── base.py          # Abstract backend interface
│   └── faiss_backend.py # FAISS vector search
├── utils.py              # Core generation utilities
└── sample_utils.py       # Sampling methods (top_p, etc.)
```

## Model Support

### Tested Models
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-1.1-2b-it`
- Quantized variants from mlx-community

### Authentication
For gated models:
```bash
huggingface-cli login
# or
export HF_TOKEN=your_token_here
```

## Current Limitations
- No repetition penalties
- LoRA/DoRA adapters not implemented
- Embeddings endpoint not implemented
- No distributed operation
- Streaming with n>1 not supported
- Extended Mind generation produces incorrect output