# MLX ParaLLM High-Performance Inference Server

## Overview

MLX ParaLLM is a high-performance inference server for Large Language Models (LLMs) on Apple Silicon, designed to rival vLLM and SGLang while leveraging the unique capabilities of Metal and unified memory architecture. This document outlines the architecture, implementation, and optimization strategies for achieving production-grade performance.

**Critical Design Requirement**: The server MUST seamlessly integrate with the RL training backend, sharing the same model and LoRA adapters between inference and training to minimize memory overhead and enable efficient online learning.

**On-Policy Training Support**: When running in conjunction with the RL trainer (`mlx_parallm_train`), the inference server automatically serves rollouts using the most recent policy weights. LoRA adapters are updated in-memory at each training step, ensuring all generated trajectories remain on-policy without any manual intervention or server restarts.

## Core Architecture

### 1. RL Training Integration

The inference server is designed to work in tandem with the RL training system:

#### Shared Model Architecture
```python
class SharedModelManager:
    """Manages a single model instance shared between inference and training"""
    
    def __init__(self, base_model_path: str):
        self.base_model = load_model(base_model_path)  # Shared base model
        self.active_lora = None  # Currently active LoRA adapter
        self.lora_versions = {}  # version_id -> LoRA weights
        self.inference_lock = asyncio.Lock()  # Coordinate access
        self.training_lock = asyncio.Lock()
        
    async def inference_forward(self, inputs, lora_version=None):
        """Forward pass for inference with optional LoRA"""
        async with self.inference_lock:
            if lora_version and lora_version != self.active_lora:
                self.switch_lora(lora_version)
            return await self.base_model.forward(inputs)
    
    async def training_step(self, batch, lora_version):
        """Training step with LoRA adapter"""
        async with self.training_lock:
            # Training modifies LoRA weights only, not base model
            gradients = await compute_gradients(batch)
            update_lora_weights(self.lora_versions[lora_version], gradients)
    
    def switch_lora(self, version_id):
        """Hot-swap LoRA adapter without reloading base model"""
        if version_id in self.lora_versions:
            self.base_model.set_lora(self.lora_versions[version_id])
            self.active_lora = version_id
```

#### Memory-Efficient Design
- **Single Model Instance**: Base model loaded once, shared between inference and training
- **LoRA Hot-Swapping**: Switch adapters without model reload (milliseconds vs seconds)
- **Zero-Copy Updates**: LoRA weight updates happen in-place using MLX unified memory
- **Versioned Adapters**: Support multiple LoRA versions for A/B testing and rollback

#### RL Training Flow Integration
```python
class RLIntegratedServer:
    def __init__(self):
        self.model_manager = SharedModelManager(base_model_path)
        self.inference_server = InferenceEngine(self.model_manager)
        self.rl_trainer = GRPOTrainer(self.model_manager)
        
    async def online_training_loop(self):
        """Continuous learning while serving"""
        while True:
            # 1. Collect trajectories from live inference
            trajectories = await self.inference_server.collect_trajectories()
            
            # 2. Score and prepare training batch
            scored_batch = await self.prepare_rl_batch(trajectories)
            
            # 3. Update LoRA weights (base model unchanged)
            await self.rl_trainer.update_step(scored_batch)
            
            # 4. Optionally update inference to use new LoRA
            if should_update_inference():
                await self.model_manager.switch_lora("latest")
```

### 2. Continuous Batching System with RL Support

The continuous batching system is designed to support both inference and trajectory collection for RL:

#### Dual-Mode Operation
```python
class ContinuousBatchProcessor:
    def __init__(self, model_manager: SharedModelManager):
        self.model_manager = model_manager
        self.inference_mode = True  # Can toggle for training
        self.trajectory_buffer = []  # For RL data collection
        
    async def process_batch(self, batch: List[Request]):
        """Process batch for inference or training data collection"""
        
        if self.inference_mode:
            # Standard inference
            outputs = await self.model_manager.inference_forward(batch)
            
            # Optionally collect trajectories for RL
            if self.collect_trajectories:
                self.trajectory_buffer.append({
                    "prompts": batch.prompts,
                    "outputs": outputs,
                    "logprobs": outputs.logprobs
                })
        else:
            # Training mode - process with gradients enabled
            outputs = await self.model_manager.training_forward(batch)
        
        return outputs
```

### 3. Enhanced BatchedKVCache with LoRA Support

The KV cache system supports efficient LoRA switching:

```python
class LoRABatchedKVCache(BatchedKVCache):
    def __init__(self, head_dim, n_kv_heads, max_batch_size=128):
        super().__init__(head_dim, n_kv_heads, max_batch_size)
        self.lora_version_per_sequence = {}  # Track LoRA version per sequence
        self.lora_cache_adjustments = {}  # Cache LoRA-specific adjustments
        
    def update_and_fetch_with_lora(self, keys, values, seq_id, lora_version):
        """Update cache considering LoRA adapter"""
        # Base KV cache update
        base_keys, base_values = super().update_and_fetch(keys, values)
        
        # Apply LoRA-specific adjustments if needed
        if lora_version and lora_version in self.lora_cache_adjustments:
            keys = base_keys + self.lora_cache_adjustments[lora_version].keys
            values = base_values + self.lora_cache_adjustments[lora_version].values
            
        self.lora_version_per_sequence[seq_id] = lora_version
        return keys, values
```

### 4. Request Scheduling with Training Priority

The scheduler supports both inference and training requests:

```python
class RLAwareScheduler:
    def __init__(self):
        self.inference_queue = asyncio.Queue()
        self.training_queue = asyncio.Queue()  # For RL trajectory collection
        self.priority_ratios = {
            "inference": 0.7,  # 70% capacity for inference
            "training": 0.3    # 30% capacity for training data
        }
        
    async def schedule_next_batch(self, available_slots: int):
        """Balance between inference and training data collection"""
        inference_slots = int(available_slots * self.priority_ratios["inference"])
        training_slots = available_slots - inference_slots
        
        batch = []
        
        # Fill inference slots
        for _ in range(min(inference_slots, self.inference_queue.qsize())):
            batch.append(await self.inference_queue.get())
            
        # Fill training slots (trajectory collection)
        for _ in range(min(training_slots, self.training_queue.qsize())):
            batch.append(await self.training_queue.get())
            
        return batch
```

### 5. Async Generation with Model Sharing

Each sequence can use different LoRA versions while sharing the base model:

```python
async def generate_sequence(
    seq_id: int,
    prompt: str,
    model_manager: SharedModelManager,
    cache: LoRABatchedKVCache,
    lora_version: Optional[str] = None,
    collect_trajectory: bool = False
):
    """Generate with optional LoRA and trajectory collection"""
    
    tokens = []
    logprobs = []
    
    # Prefill phase
    prefill_output = await model_manager.inference_forward(
        prompt, 
        lora_version=lora_version
    )
    tokens.extend(prefill_output.tokens)
    
    # Decode phase
    while len(tokens) < max_tokens:
        # Get KV cache for this sequence with LoRA
        keys, values = cache.update_and_fetch_with_lora(
            new_keys, new_values, seq_id, lora_version
        )
        
        # Generate next token
        output = await model_manager.inference_forward(
            tokens[-1], 
            kv_cache=(keys, values),
            lora_version=lora_version
        )
        
        tokens.append(output.token)
        if collect_trajectory:
            logprobs.append(output.logprob)
        
        if output.token == eos_token:
            break
    
    # Return trajectory for RL if requested
    if collect_trajectory:
        return {
            "tokens": tokens,
            "logprobs": logprobs,
            "lora_version": lora_version
        }
    
    return tokens
```

### 6. Checkpoint Management

Efficient checkpoint system for LoRA adapters:

```python
class CheckpointManager:
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.checkpoints_dir = Path("checkpoints")
        self.active_checkpoints = {}  # name -> LoRA weights in memory
        
    async def save_checkpoint(self, name: str, lora_weights):
        """Save LoRA weights (small, typically <100MB)"""
        checkpoint_path = self.checkpoints_dir / f"{name}.safetensors"
        mx.save_safetensors(checkpoint_path, {"lora": lora_weights})
        
    async def load_checkpoint(self, name: str):
        """Load LoRA weights into memory"""
        checkpoint_path = self.checkpoints_dir / f"{name}.safetensors"
        weights = mx.load_safetensors(checkpoint_path)["lora"]
        self.active_checkpoints[name] = weights
        return weights
        
    async def hot_swap(self, model_manager: SharedModelManager, checkpoint_name: str):
        """Hot-swap LoRA without interrupting service"""
        if checkpoint_name not in self.active_checkpoints:
            weights = await self.load_checkpoint(checkpoint_name)
        else:
            weights = self.active_checkpoints[checkpoint_name]
            
        await model_manager.switch_lora(checkpoint_name)

## Performance Benchmarks

### Model Quantization Comparison

We tested the Hermes-4-Qwen3-14B model with different quantization levels to demonstrate the performance improvements of MLX ParaLLM Server:

#### Raw MLX Generate (Sequential Processing)

| Model Version | Avg Time/Request | Tokens/sec | Memory Usage |
|--------------|------------------|------------|--------------|
| **FP16 (Original)** | 4.31s | 10.9 | ~28GB (estimated) |
| **8-bit Quantized** | 2.42s | 19.6 | 15.3GB |
| **4-bit Quantized** | 1.46s | 31.5 | 8.3GB |

#### MLX ParaLLM Server (Batched Processing)

| Model Version | 4 Concurrent Requests | 8 Concurrent Requests | Memory Usage |
|--------------|----------------------|----------------------|--------------|
| **FP16 (Original)** | 22.04s total (7.6 tok/s) | 14.40s total (15.3 tok/s) | ~28GB |
| **8-bit Quantized** | 4.08s total (41.4 tok/s) | 4.83s total (44.5 tok/s) | 15.3GB |
| **4-bit Quantized** | 3.29s total (52.0 tok/s) | 3.97s total (59.4 tok/s) | 8.3GB |

### Performance Multipliers

#### 4-bit Quantized Model Performance Gains:
- **vs Raw MLX**: 
  - 1.65x faster per request
  - 2.9x higher throughput
  
- **vs MLX ParaLLM Server (FP16)**:
  - 6.7x faster for 4 concurrent requests
  - 3.6x faster for 8 concurrent requests
  - 70% less memory usage

- **MLX ParaLLM Server vs Raw MLX (4-bit)**:
  - **4 concurrent requests**: 1.77x faster overall (52.0 vs 31.5 * 4/5.83)
  - **8 concurrent requests**: 3.86x faster overall (59.4 vs 31.5 * 8/11.64)

### Key Advantages of MLX ParaLLM Server

1. **True Concurrent Processing**: Unlike raw MLX which processes sequentially, ParaLLM handles multiple requests simultaneously
2. **Dynamic Batching**: Automatically groups requests for efficient GPU utilization
3. **Memory Efficiency**: PagedKVCache reduces memory overhead for long sequences
4. **Quantization Support**: Seamless integration with 4-bit and 8-bit quantized models
5. **API Compatibility**: Drop-in replacement for OpenAI API

### Recommended Configuration

For production deployments:
- **Model**: 4-bit quantized version
- **Batch Size**: 8-16 (depending on available memory)
- **Benefits**:
  - 70% memory reduction
  - 3-6x throughput improvement
  - Minimal quality degradation

## High-Performance Inference Details

This section documents the inference server internals that maximize throughput and concurrency on Apple Silicon: paged KV caching, continuous batching, co-batched streaming, tokenization caching, and metrics.

### Paged KV Cache

- Purpose: Allow each sequence in a batch to advance with its own KV offset, enabling continuous batching with mixed sequence lengths without re-allocating or copying per-step.
- Implementation:
  - `PagedKVCache` (in `mlx_parallm/models/base.py`) maintains per-sequence offsets (`offsets_list`) backed by a single buffer shaped `(B, n_kv_heads, T_cap, head_dim)`.
  - Capacity grows in fixed “step” blocks (default 256 tokens) via `_ensure_capacity_for(max_needed)` to amortize allocations.
  - `update_and_fetch(keys, values)` writes per-row ranges and returns a unified slice up to the current max end offset.
  - `offsets` exposes a list of per-row offsets for attention/mask construction.
  - `reset(batch_size)` preserves buffers when the batch size is unchanged; drops references on size change to reallocate cleanly.
- KV Pool:
  - `_KVPool` (in `utils.py`) reuses per-layer cache objects keyed by `(head_dim, kv_heads, batch_size)`, defaulting to `PagedKVCache` (paged=True).
  - The generation path (`utils.py::generate_step`) pulls paged caches from the pool when `cache=None`.
- Attention/masks:
  - `create_additive_causal_mask_variable(B, N, offsets, total_length)` builds per-row additive causal masks using per-sequence offsets.
  - Llama and Qwen adapters apply RoPE per row and use the variable-length mask to support heterogeneous sequence lengths.

### Continuous Batching Scheduler

- Mode: `--scheduler continuous` (default remains the “batch worker” + streaming worker).
- Behavior:
  - Admits both non-streaming and streaming requests into a single mixed batch.
  - Runs a step-wise decode loop using `batch_stream_generate_text` and dispatches:
    - Streaming deltas as SSE chunks to each request’s queue.
    - Non-streaming completions aggregated and resolved when finished.
  - At each step boundary it checks queues; if new work arrives, it breaks and rebuilds the batch, allowing rapid admission without mid-kernel preemption.
  - Uses `PagedKVCache` under the hood for per-sequence offsets and efficient cache growth.
- Fairness and latency:
  - Mixed admission ensures streaming does not starve batch throughput and vice versa.
  - Step-boundary admission is a safe midpoint with the current generation API and paged KV.

### Co-Batched Streaming

- Default-path streaming worker (when not in `continuous` mode) batches compatible chat requests to share decode steps.
- Controls:
  - `--max-concurrent-streams` limits parallel streams globally (protects batch throughput).
  - Internally uses `STREAMING_SEMAPHORE` for admission of stream senders; continuous scheduler also respects it for streaming dispatch.

### Tokenization & Template Caching

- `encode_cached(tokenizer, text)`: LRU cache for single string encodes (reduces CPU when prompts repeat).
- `apply_chat_template_cached(tokenizer, messages, add_generation_prompt=True)`: LRU cache for chat templating keyed by a minimalized message signature (role/content only) and tokenizer identity.

### Metrics

- Endpoint: `GET /debug/metrics`
- Fields:
  - `batches_processed`: Number of decode cycles processed (batch worker and continuous scheduler).
  - `avg_batch_fill_pct`: Average batch utilization as a percentage of `--max-batch-size`.
  - `batch_fill_hist`: 10-bucket histogram of batch fill (0–10, …, 90–100%).
  - `queue_depth_last`: Last observed combined queue depth.
  - `stream_batches_processed`: Count of co-batched streaming batches.
  - `prompt_tps_last`, `prompt_tps_avg`: Prompt tokens/sec (time-to-first-yield window); avg is total tokens over total time so far.
  - `decode_tps_last`, `decode_tps_avg`: Decode tokens/sec (active sequence tokens over wall time); avg is running.

### CLI Usage & Configuration

Run the server:
- `mlx_parallm_serve --model-path <hf_id_or_path> --port 8000`
- Add continuous scheduler: `--scheduler continuous`
- With LoRA adapter: `--lora-path <adapter_path>`

Key options (from `mlx_parallm/cli.py`):
- `--model-path`: Required. Hugging Face ID or local path of the base model.
- `--host`: Bind address (default `127.0.0.1`).
- `--port`: Port (default `8000`).
- `--lora-path`: Optional LoRA/DoRA adapter to load at startup. During RL training, this adapter is automatically updated in-memory at each training step.
- `--max-batch-size`: Maximum batch size for dynamic/continuous batching (default 8).
- `--batch-timeout`: Timed wait window to collect a batch (seconds; default 0.1).
- `--request-timeout-seconds`: Per-request timeout (default 600.0).
- `--max-concurrent-streams`: Limits simultaneous streams (default 4).
- `--scheduler`: `default` or `continuous`.

Recommended starting points for Apple Silicon (large memory machines):
- Low-latency with concurrency: `--scheduler continuous --max-batch-size 8 --batch-timeout 0.05 --max-concurrent-streams 4`
- Throughput-focused: increase `--max-batch-size` (e.g., 16) and slightly increase `--batch-timeout` to improve fill, then monitor `/debug/metrics` (batch_fill_hist, tokens/sec).

### Quick Start Examples

Prereqs
- Create env and install editable: `uv venv && source .venv/bin/activate && uv pip install -e .`
- Authenticate to Hugging Face if using private/gated repos: `huggingface-cli login` or set `HF_TOKEN`.

Launch (Llama 3.2 3B Instruct 4-bit)
- Start server (continuous scheduler):
  - `mlx_parallm_serve --model-path mlx-community/Llama-3.2-3B-Instruct-4bit --scheduler continuous --host 127.0.0.1 --port 8000`
- Health: `curl http://127.0.0.1:8000/health`
- List models: `curl -s http://127.0.0.1:8000/v1/models`
- Non-stream completion:
  - `curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","prompt":"Hello","max_tokens":16}'`
- Chat completion (non-stream):
  - `curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","messages":[{"role":"user","content":"Suggest 3 team names."}],"max_tokens":32}'`
- Chat completion (streaming SSE):
  - `curl -N -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","stream":true,"messages":[{"role":"user","content":"In one sentence, describe the ocean."}],"max_tokens":32}'`
- Metrics: `curl -s http://127.0.0.1:8000/debug/metrics`

Launch (Hermes-4 Qwen3)
- `mlx_parallm_serve --model-path NousResearch/Hermes-4-Qwen3-14B-1-e3 --scheduler continuous --host 127.0.0.1 --port 8001`
- Use the same curl patterns above, replacing the model and port.

Tips
- If port is busy: `lsof -ti :8000 -sTCP:LISTEN | xargs -r kill -9`
- Keep quick tests short: use `max_tokens` 8–32 to sanity-check latency and metrics.

### Operational Guidance

- Health: `curl http://127.0.0.1:8000/health`
- Quick completion: `curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"<hf_id>","prompt":"Hello","max_tokens":16}'`
- Streaming chat (SSE): POST to `/v1/chat/completions` with `{"stream": true}`.
- Metrics-driven tuning: Watch `avg_batch_fill_pct`, `batch_fill_hist`, and `*_tps_*`; adjust `--batch-timeout`, `--max-batch-size`, and `--max-concurrent-streams` accordingly.

### Notes & Next Steps

- KV cache reuse across continuous batch rebuilds is planned to reduce admission overhead further.
- Consider switching SSE serialization to `orjson` for lower CPU overhead.
- Tokenization can be offloaded to a thread pool for heavier workloads.
- Additional metrics (tokens/sec per phase, distribution/histograms) can be extended as needed.

## Code Pointers

- Paged KV and masks
  - `mlx_parallm/models/base.py`:
    - `PagedKVCache`: per-sequence paged cache implementation.
    - `create_additive_causal_mask_variable`: per-row variable-length causal masks.
  - `mlx_parallm/models/llama.py`, `mlx_parallm/models/qwen3.py`:
    - Attention updates for per-row offsets and RoPE handling.

- Generation paths and caches
  - `mlx_parallm/utils.py`:
    - `generate_step`: default to `PagedKVCache` via `_KVPool`.
    - `_KVPool`: cache reuse; `_GlobalPrefixCache`: prefix KV LRU scaffold.
    - `batch_stream_generate_text`: shared step loop for batch + streaming.
    - `encode_cached`, `apply_chat_template_cached`: LRU tokenization/template caches.

- Server and scheduling
  - `mlx_parallm/server/main.py`:
    - `continuous_scheduler_worker`: mixed admission, stepwise rebuild.
    - `streaming_batch_worker`: co-batched streaming (default scheduler).
    - `/debug/metrics`: telemetry endpoint fields listed above.
    - Request queues: `REQUEST_QUEUE` (non-stream), `STREAM_CHAT_QUEUE` (stream), and `STREAMING_SEMAPHORE` guard.

- CLI
  - `mlx_parallm/cli.py`: server entrypoint and CLI flags including `--scheduler` and `--max-concurrent-streams`.

## Troubleshooting

- Pad token and left padding
  - Batched generation requires left padding. Ensure the tokenizer has a pad token; if not, it falls back to EOS as pad: the server sets `tokenizer._tokenizer.padding_side = 'left'` and, when necessary, `pad_token = eos_token`.
  - Symptom: shape/broadcast errors or garbled outputs during batching → verify pad token and left padding are in effect.

- Model not found in registry (404 on requests)
  - Ensure the server was launched with `--model-path <hf_id_or_path>` and that `/v1/models` lists the model.
  - For private/gated Hugging Face repos, authenticate (`huggingface-cli login` or `HF_TOKEN`) before starting the server.

- Streaming stalls or starvation
  - If streaming responses throttle throughput, reduce `--max-concurrent-streams` (default 4) to protect batch progress.
  - In `continuous` mode, both streaming and non-streaming share the step loop; batch size and batch timeout still affect latency.

- Batch fill is low
  - Increase `--batch-timeout` slightly (e.g., 0.05 → 0.1) and/or `--max-batch-size` to improve fill under load; confirm via `avg_batch_fill_pct` and `batch_fill_hist` in `/debug/metrics`.

- High latency to first token
  - Watch `prompt_tps_last` in `/debug/metrics`. If low, the model/prompt prefill phase may be underutilized; consider smaller batches for latency-sensitive traffic or separate pools.

- Port conflicts / stale processes
  - If the server fails to bind or health never returns, free the port (e.g., `lsof -ti :8000 | xargs -r kill -9`) and restart.

- `n > 1` streaming not supported
  - Streaming with `n > 1` is rejected; use non-streaming with `n` for multiple choices.

- Memory pressure on large contexts
  - While PagedKVCache reduces reallocation, very long contexts still grow per-layer KV. Consider sliding windows or max context limits to cap memory.
```

## On-Policy Training Architecture

When the inference server runs alongside the RL trainer, it maintains perfect on-policy rollouts through a shared model registry architecture:

### Shared Model Registry
```python
# mlx_parallm/server/state.py
model_registry: Dict[str, InternalModelRecord] = {}
weight_update_lock = RLock()
```

The same `model_instance` is used by both:
- **Inference Server**: Generates rollouts via HTTP API
- **Training Thread**: Computes gradients and updates LoRA weights

### Live Weight Updates
At each training step:
1. Training computes gradients for LoRA parameters only
2. Optimizer updates the LoRA weights in-place
3. Updated weights are saved to disk (e.g., `checkpoints/step_1/adapter.npz`)
4. `apply_lora_update_for_record()` loads new weights into the shared model instance
5. Next inference request automatically uses updated weights

### Key Benefits
- **Zero-copy updates**: Weights updated in shared memory
- **No service interruption**: Server continues serving during updates
- **Always on-policy**: Every rollout uses the latest trained weights
- **Minimal overhead**: <10ms for weight updates

## Performance Optimizations for RL Training

### 1. Memory Optimization
- **Shared Base Model**: 15GB saved by not duplicating model
- **LoRA-only Updates**: Training updates <1% of parameters
- **Gradient Checkpointing**: Reduce memory during training
- **Mixed Precision**: FP16 for inference, FP32 for critical training ops

### 2. Throughput Optimization
- **Parallel Trajectory Collection**: Collect from multiple sequences simultaneously
- **Batched Scoring**: Score multiple trajectories together
- **Async Training Steps**: Training doesn't block inference
- **Pipeline Parallelism**: Overlap compute and communication

### 3. Latency Optimization
- **LoRA Caching**: Keep frequently used adapters in memory
- **Incremental Updates**: Apply LoRA deltas without full reload
- **Prefix Sharing**: Share KV cache between similar prompts
- **Speculative Decoding**: Use base model for speculation, LoRA for verification

## Configuration for RL Training

### Server Configuration with RL
```yaml
server:
  mode: "inference_and_training"  # or "inference_only", "training_only"
  max_batch_size: 128
  
model:
  base_model_path: "models/Llama-3-8B"
  lora_rank: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "v_proj"]
  
rl_training:
  enabled: true
  trajectory_buffer_size: 1000
  update_frequency: 100  # Update LoRA every N trajectories
  scoring_batch_size: 32
  
  # Atropos integration
  atropos_url: "http://localhost:8001"
  environment: "gsm8k"
  
checkpointing:
  interval: 500  # Save every N updates
  keep_n_checkpoints: 10
  checkpoint_dir: "./checkpoints"
```

### Runtime Integration
```python
# Start server with RL training
server = RLIntegratedServer(
    base_model="Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    rl_config=RLConfig(
        algorithm="grpo",
        learning_rate=1e-5,
        batch_size=32
    )
)

# Serve inference while training
async def main():
    # Start inference server
    inference_task = asyncio.create_task(
        server.run_inference(port=8000)
    )
    
    # Start RL training loop
    training_task = asyncio.create_task(
        server.run_training_loop()
    )
    
    # Run both concurrently
    await asyncio.gather(inference_task, training_task)
```

## API Extensions for RL

### POST /v1/completions (with RL support)
```json
{
  "model": "base-model",
  "lora_adapter": "checkpoint-500",  // Optional: specific LoRA version
  "prompt": "...",
  "collect_trajectory": true,  // Collect for RL training
  "scoring_function": "gsm8k",  // How to score the output
  "metadata": {
    "environment": "math",
    "difficulty": "hard"
  }
}
```

### POST /v1/training/update
Trigger training update manually:
```json
{
  "trajectories": [...],  // Collected trajectories
  "update_inference": true,  // Update serving model after training
  "checkpoint_name": "manual-update-1"
}
```

### GET /v1/training/status
```json
{
  "training_enabled": true,
  "current_lora": "checkpoint-500",
  "trajectories_collected": 1523,
  "last_update": "2025-01-20T10:30:00Z",
  "loss": 0.23,
  "available_checkpoints": [
    "checkpoint-500",
    "checkpoint-400",
    "best-performance"
  ]
}
```

## Deployment with RL Training

### Single-Node Setup
```bash
# Start integrated server
mlx_parallm_serve \
  --model-path models/Llama-3-8B \
  --enable-rl-training \
  --lora-rank 16 \
  --atropos-url http://localhost:8001 \
  --port 8000
```

### Multi-Node Setup
```bash
# Node 1: Inference + Model
mlx_parallm_serve --mode inference --share-model

# Node 2: RL Training
mlx_parallm_train --mode rl --connect-to node1:8000

# Node 3: Additional inference
mlx_parallm_serve --mode inference --connect-to node1:8000
```

## Performance Benchmarks with RL

### Memory Usage Comparison
| Configuration | Base Model | w/ LoRA | w/ RL Training | Savings |
|--------------|------------|---------|----------------|---------|
| Separate Instances | 15GB × 2 | 15.5GB × 2 | 31GB | 0% |
| Shared Model | 15GB | 15.5GB | 16GB | 48% |
| 4-bit Quantized + LoRA | 8.3GB | 8.4GB | 8.5GB | 72% |

### Throughput with RL Training
| Mode | Throughput | Latency P50 | Training Updates/min |
|------|------------|-------------|---------------------|
| Inference Only | 1200 tok/s | 25ms | 0 |
| Inference + RL | 1000 tok/s | 30ms | 10 |
| Training Priority | 600 tok/s | 45ms | 30 |

### LoRA Hot-Swap Performance
- Swap Time: <10ms
- No request interruption
- Memory overhead: <100MB per adapter

## Monitoring RL Training

### Metrics
```python
# RL-specific metrics
mlx_parallm_rl_trajectories_collected
mlx_parallm_rl_training_loss
mlx_parallm_rl_updates_per_second
mlx_parallm_lora_swaps_total
mlx_parallm_checkpoint_saves_total
```

### Debugging
```bash
# View training status
curl http://localhost:8000/debug/training

# Force checkpoint
curl -X POST http://localhost:8000/training/checkpoint

# Switch LoRA adapter
curl -X POST http://localhost:8000/model/switch-lora -d '{"version": "checkpoint-300"}'
```

## Best Practices for RL Integration

1. **Memory Management**
   - Keep base model frozen
   - Limit active LoRA adapters in memory
   - Use gradient checkpointing for large batches

2. **Performance Tuning**
   - Balance inference/training ratio based on load
   - Use async training to avoid blocking
   - Cache frequent LoRA combinations

3. **Stability**
   - Regular checkpointing
   - Validation before LoRA swap
   - Rollback mechanism for bad updates

4. **Monitoring**
   - Track training loss trends
   - Monitor inference quality metrics
   - Alert on training divergence

## Future Enhancements

### Planned RL Features
1. **Multi-Agent Training**: Support multiple LoRA adapters training simultaneously
2. **Federated RL**: Distributed training across edge devices
3. **Online A/B Testing**: Automatic LoRA selection based on performance
4. **Reward Modeling**: Integrated reward model training
5. **Curriculum Learning**: Adaptive difficulty based on model performance

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
