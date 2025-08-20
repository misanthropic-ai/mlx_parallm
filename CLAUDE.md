# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx_parallm is a high-performance, parallelized batch generation server for MLX models designed for Apple Silicon. It provides an OpenAI-compatible API with support for batched inference and future RL/embedding capabilities.

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

```

### Testing
```bash
# No test framework currently configured - check README or pyproject.toml for testing approach
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

### Key Configuration
- `MAX_BATCH_SIZE = 8`: Maximum requests per batch
- `BATCH_TIMEOUT = 0.1`: Seconds to wait for batch formation
- Model registry uses model path as ID by default

## API Endpoints

### Core Endpoints
- `GET /health`: Health check
- `GET /v1/models`: List loaded models
- `POST /v1/completions`: Text completions (OpenAI-compatible)
  - Supports: `prompt`, `max_tokens`, `temperature`, `top_p`, `stream`, `n`, `logprobs`, `echo`
- `POST /v1/chat/completions`: Chat completions (OpenAI-compatible)
  - Applies tokenizer chat template automatically
  - Supports: `messages`, `max_tokens`, `temperature`, `top_p`, `stream`, `n`


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
│   ├── qwen3.py          # Qwen3 architecture
│   ├── phi3.py           # Phi3 architecture
│   ├── gemma.py          # Gemma architecture
│   └── mixtral.py        # Mixtral architecture
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

## Running Python scripts
- Always use `uv run` to execute python scripts, models, tests
- `uv` will automatically use the correct virtual environment