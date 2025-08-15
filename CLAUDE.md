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

## Extended Mind Transformers Integration

### Overview
We've implemented Extended Mind Transformers (EMT) in mlx_parallm, allowing LLMs to access external memories during generation without fine-tuning. This enables RAG-like capabilities with better integration into the model's attention mechanism.

### Implementation Status
- ✅ Core architecture implemented (`mlx_parallm/models/llama_extended.py`)
- ✅ Memory backend system with FAISS support (`mlx_parallm/memory/`)
- ✅ Model loading with `--use-extended-mind` flag
- ✅ Memory addition and retrieval working
- ⚠️ Generation with memories produces incorrect output (exclamation marks)
- ❌ API endpoints for memory management not yet implemented

### Key Components

#### Memory Backend System (`mlx_parallm/memory/`)
- **MemoryManager**: Central manager for memory backends
- **MemoryBackend**: Abstract interface for memory storage
- **FAISSBackend**: FAISS-based vector similarity search
  - Per-head memory indexing for grouped query attention
  - Cosine similarity search with top-k retrieval
  - Support for layer-specific memories

#### Extended Model Classes
- **ExtendedModelArgs**: Configuration with memory parameters
- **ExtendedAttention**: Attention with memory retrieval
  - Retrieves top-k memories per query position
  - Separate attention computation for memories vs cached values
  - Handles BatchedKVCache integration
- **ExtendedTransformerBlock**: Transformer block using ExtendedAttention
- **ExtendedLlamaModel**: Model with memory manager
- **ExtendedModel**: Wrapper with memory management methods

#### Memory Flow
1. Add memories via `model.add_memories(tokens)`
   - Tokens embedded and passed through layers
   - Keys/values extracted and stored in backend
2. During generation:
   - Queries normalized and searched against memory
   - Top-k memories retrieved per attention head
   - Memory scores computed separately
   - Combined with regular attention

### Current Issues

#### BatchedKVCache Integration
The main issue is handling the dimension mismatch between:
- Memory values: shape `(B, n_heads, L * topk, head_dim)` for current queries
- Cached values: shape `(B, n_heads, total_cached_seq_len, head_dim)`

We implemented separate attention computation for memories vs cached values, but generation still produces incorrect output.

#### RoPE Handling
- Fixed to support both Llama 2 style (`"type": "linear"`) and Llama 3 style (`"rope_type": "llama3"`)
- Llama 3.2 models use `rope_type` field instead of `type`
- No scaling applied for `rope_type: "llama3"`

### Usage Example
```python
# Load model with extended mind
model, tokenizer = load('mlx-community/Llama-3.2-3B-Instruct-4bit', use_extended_mind=True)
model.set_model_id('my-model')

# Add memories
memory_text = "Alexander Grothendieck became a French citizen in 1971."
tokens = tokenizer.encode(memory_text)
model.add_memories(mx.array(tokens))

# Generate with memory access
prompt = "When did Grothendieck get French citizenship?"
response = generate(model, tokenizer, prompt)
```

### Next Steps
1. Fix generation issue with memory-augmented attention
2. Add API endpoints for memory management:
   - POST `/v1/models/{model_id}/memories`
   - GET `/v1/models/{model_id}/memories`
   - DELETE `/v1/models/{model_id}/memories`
3. Implement other memory backends (Redis, Neo4j, SQL)
4. Add memory parameters to completion request schemas
5. Performance optimization and testing

### References
- Extended Mind Transformers paper: https://arxiv.org/abs/2406.02332
- Reference implementation: https://github.com/normal-computing/extended-mind-transformers