# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx_parallm is a high-performance, parallelized batch generation server for MLX models, designed for Apple Silicon devices. It provides an OpenAI-compatible API for text generation, chat completions, and future support for embeddings and RL tasks.

## Build and Development Commands

### Installation and Setup
- Install in development mode: `uv pip install -e .`
- Create virtual environment: `uv venv && source .venv/bin/activate`
- Add dependencies: `uv add <package_name>` (runtime) or `uv add <package_name> --dev` (development)

### Running the Server
- Start server: `mlx_parallm_serve --model-path <model_path> --host <host> --port <port>`
- Example: `mlx_parallm_serve --model-path mistralai/Mistral-7B-Instruct-v0.1 --port 8000`
- Health check endpoint: `GET http://127.0.0.1:8000/health`

### Core API Endpoints
- `/v1/models` - List available models
- `/v1/completions` - Text completions (OpenAI compatible)
- `/v1/chat/completions` - Chat completions (OpenAI compatible)
- Both endpoints support streaming with `stream: true`

## Architecture and Core Components

### High-Level Architecture
The server follows a batched generation architecture with:
1. **FastAPI Server** (`mlx_parallm/server/main.py`) - HTTP API layer
2. **Request Queue System** - Asynchronous batching for efficiency
3. **Batch Processing Worker** - Background task for batch generation
4. **Model Registry** - In-memory registry for loaded models
5. **Generation Engine** - MLX-based text generation with BatchedKVCache

### Key Components

#### BatchedKVCache (`mlx_parallm/models/base.py`)
- Custom KV cache implementation for efficient batched inference
- Replaces standard `KVCache` from mlx_lm for parallel processing
- Supports dynamic batching with configurable step sizes

#### Generation Functions (`mlx_parallm/utils.py`)
- `batch_generate()` - Synchronous batch generation for multiple prompts
- `stream_generate()` - Single prompt streaming generation
- `batch_stream_generate_text()` - Batched streaming generation
- `generate_step()` - Core token generation with sampling support

#### Request Batching System
- **QueuedRequest** class for managing async requests
- **batch_processing_worker()** - Background task that:
  - Collects requests into batches (up to MAX_BATCH_SIZE=8)
  - Handles `n` parameter by replicating prompts
  - Distributes results back to original requests
  - Supports both completion and chat completion requests

#### Model Loading and Management
- Supports both Hugging Face models and local paths
- Automatic model download from HF Hub if not local
- Quantized model support (4-bit, other MLX quantization schemes)
- LoRA/DoRA adapter support (planned)

### Supported Models
Current tested models include:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-1.1-2b-it`
- Quantized variants (mlx-community 4-bit models)

### Request Flow
1. HTTP request arrives at FastAPI endpoint
2. For streaming: Direct generation pipeline
3. For non-streaming: Request queued for batch processing
4. Batch worker collects requests (timeout: 0.1s, max batch: 8)
5. Prompts tokenized and padded for batch generation
6. MLX model inference with BatchedKVCache
7. Results distributed back to original request futures
8. Responses returned to clients

## Development Notes

### Code Organization
- `/mlx_parallm/cli.py` - CLI entry point and argument parsing
- `/mlx_parallm/server/` - FastAPI server and request handling
- `/mlx_parallm/server/schemas.py` - Pydantic models for API
- `/mlx_parallm/models/` - Model architectures with BatchedKVCache
- `/mlx_parallm/utils.py` - Core generation utilities
- `/mlx_parallm/sample_utils.py` - Sampling methods (top_p, etc.)

### Configuration and Parameters
- Model registry uses model path as ID by default
- Batch processing configurable via MAX_BATCH_SIZE and BATCH_TIMEOUT
- Temperature and top_p sampling supported
- Chat template formatting via tokenizer.apply_chat_template()

### Authentication Requirements
For gated/private models:
- Run `huggingface-cli login` or set `HF_TOKEN` environment variable
- Ensure access granted on Hugging Face model page

### Current Limitations
- No repetition penalties yet
- LoRA/DoRA adapters not implemented
- Embeddings endpoint not implemented
- No distributed operation support
- Streaming with n>1 not supported

### Future Features (See TODO.md)
- RL inference endpoints for policy updates
- Reward model functionality
- Multimodal (vision) support
- Distributed operation with MLX
- Performance optimizations and quantization improvements