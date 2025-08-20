# RL Trainer Integration Plan for mlx_parallm

## Overview
Integrate asynchronous RL training capabilities into mlx_parallm, enabling on-policy GRPO training with continuous model weight updates. The system will:
1. Interface with Atropos environments for rollout collection
2. Implement GRPO training in MLX 
3. Support dynamic weight updates to the serving model
4. Enable both full-weight and LoRA/QLoRA training
5. Support serving models with LoRA adapters

## Architecture Components

### 1. **LoRA Support for Inference Server** (`mlx_parallm/models/lora.py`)
- **Adapter Loading**:
  - Load LoRA adapters at model initialization
  - Support multiple adapter formats (safetensors, MLX native)
  - Apply adapters to specified layers
  - Handle quantized base models with adapters
  
- **Server Integration**:
  - Add `--lora-path` argument support
  - Support hot-swapping adapters during serving
  - Maintain adapter state in BatchedKVCache

### 2. **RL Training Module** (`mlx_parallm/rl_training/`)
- **`grpo_trainer.py`**: Core GRPO implementation in MLX
  - Adapt from simple-trainer's GRPO logic
  - Support batched training with MLX operations
  - Handle both on-policy and off-policy training
  - Implement importance sampling, KL penalties, entropy bonuses
  
- **`trainer_base.py`**: Abstract base class for RL algorithms
  - Common interface for different RL methods
  - Extensible for PPO, DPO, REINFORCE, etc.

- **`rollout_buffer.py`**: Manage trajectory data
  - Store ScoredDataGroups from Atropos
  - Handle batch formation and sampling
  - Support priority/weighted sampling

### 3. **Atropos Integration** (`mlx_parallm/rl_training/atropos_client.py`)
- Async client to fetch batches from Atropos API server
- Convert Atropos ScoredData to MLX tensors
- Handle tokenization alignment between environments and model
- Support for multimodal data (images) if needed
- Register with Atropos API on startup

### 4. **Weight Update System** (`mlx_parallm/rl_training/weight_updater.py`)
- **In-Process Updates**:
  - Direct memory updates since server runs in same process
  - Update only LoRA weights when training with adapters
  - Zero-copy weight transfers via shared MLX arrays
  - No network overhead or serialization needed
  
- **LoRA/QLoRA Support**:
  - Train only adapter weights for memory efficiency
  - Direct adapter weight updates to serving model
  - Support for 4-bit quantized base models

### 5. **Unified Training Command** (`mlx_parallm/rl_training/train.py`)
- **Single entry point that**:
  - Spawns inference server thread
  - Waits for server to be ready
  - Registers with Atropos API
  - Starts training loop in main thread
  - Coordinates weight updates between threads
  - Handles graceful shutdown
  
- **Training loop**:
  - Fetch rollouts continuously from Atropos
  - Train on batches
  - Update serving model weights in-memory
  - Save checkpoints at configured intervals
  - Continue until target steps reached

### 6. **Checkpoint Management** (`mlx_parallm/rl_training/checkpoint.py`)
- **Checkpoint Format**:
  - Model weights (full or LoRA adapters only)
  - Training state (optimizer, step count, etc.)
  - Metadata file with:
    - Current training step
    - Atropos environment state
    - Training config
    - Performance metrics
    - LoRA configuration if applicable
- **Resume Support**:
  - Load checkpoint and metadata
  - Resume from exact step
  - Re-initialize server with checkpoint weights
  - Re-register with Atropos at correct position

### 7. **Adapter Merging Tool** (`mlx_parallm/tools/merge_lora.py`)
- **CLI Tool for Merging**:
  ```bash
  mlx_parallm_merge_lora \
    --base-model NousResearch/Hermes-4-Qwen3-14B-1-e3 \
    --lora-path checkpoints/final/lora_weights.safetensors \
    --output-path models/qwen3-14b-gsm8k-merged \
    --format safetensors  # or mlx
  ```
- **Features**:
  - Load base model and LoRA weights
  - Merge adapters into base model
  - Save as new model (no overwrite by default)
  - Support multiple output formats
  - Optionally push to HuggingFace

### 8. **Configuration** (`mlx_parallm/rl_training/config.py`)
- **Server settings**:
  - Port for inference server
  - Host binding
  - Batch size for inference
- **Training hyperparameters**:
  - Learning rate, batch size, update epochs
  - GRPO-specific: KL beta, entropy weight, clip ratios
  - LoRA config: rank, alpha, target modules
- **Atropos connection**:
  - API URL
  - Environment registration details
- **Checkpoint settings**:
  - Save interval
  - Directory path

### 9. **CLI Entry Points** (`mlx_parallm/cli.py` extensions)
- **Unified training command** (launches both server and trainer):
  ```bash
  mlx_parallm_train --config train_configs/grpo_qwen3.toml --resume checkpoint_dir/
  ```
- **Standalone serving** (for non-training inference):
  ```bash
  mlx_parallm_serve --model-path base_model --lora-path adapter.safetensors --port 8000
  ```
- **Merging command**:
  ```bash
  mlx_parallm_merge_lora --base-model base --lora-path adapter --output merged_model
  ```

## Implementation Steps

### Phase 1: LoRA Inference Support (Week 1)
1. Add LoRA adapter loading to model initialization
2. Integrate adapter support with BatchedKVCache
3. Test serving with pre-trained LoRA adapters
4. Implement in-memory weight sharing architecture

### Phase 2: Core GRPO Training (Week 1)
1. Create RL training module structure
2. Port GRPO algorithm to MLX
3. Implement rollout buffer for trajectory management
4. Add basic training loop with dummy data

### Phase 3: Unified Process Architecture (Week 1-2)
1. Create unified launcher that spawns server thread
2. Implement thread-safe weight sharing
3. Build async Atropos client with registration
4. Test end-to-end with gsm8k environment

### Phase 4: LoRA Training & Updates (Week 2)
1. Implement LoRA training mode
2. Create direct in-memory adapter updates
3. Test dynamic adapter updates during serving
4. Verify memory efficiency gains

### Phase 5: Tools & Utilities (Week 2-3)
1. Implement adapter merging tool
2. Add checkpoint saving with LoRA weights
3. Test checkpoint/resume with unified process
4. Create conversion utilities

### Phase 6: Testing & Optimization (Week 3)
1. Test full pipeline with gsm8k
2. Benchmark LoRA vs full training
3. Optimize thread coordination
4. Test merged model quality

## Key Design Decisions

### Process Architecture
- Single process with multiple threads
- Inference server on dedicated thread(s)
- Training loop on main thread
- Shared MLX arrays between threads
- Thread-safe weight updates via locks

### Memory Management
- Leverage MLX unified memory
- No serialization between trainer and server
- Direct pointer sharing for weights
- Minimal memory overhead

### Synchronization
- Read-write locks for weight updates
- Server continues serving during weight updates
- Atomic weight swaps when possible

## Configuration Example

```toml
[server]
port = 8000
host = "127.0.0.1"
batch_size = 8
max_concurrent_requests = 100

[model]
base_path = "NousResearch/Hermes-4-Qwen3-14B-1-e3"
lora_path = "checkpoints/latest/lora_weights.safetensors"  # Optional for resume

[rl_training]
algorithm = "grpo"
learning_rate = 1e-5
batch_size = 32
update_epochs = 4
kl_beta = 0.05
entropy_weight = 0.01
clip_ratio = 0.2
total_steps = 10000
checkpoint_interval = 500

[rl_training.lora]
enabled = true
rank = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
quantize_base = true  # Use 4-bit base model

[rl_training.atropos]
api_url = "http://localhost:8001"  # Atropos API server
wandb_group = "qwen3-gsm8k"
wandb_project = "rl-training"
rollout_batch_size = 128
max_token_length = 2048

[rl_training.weight_updates]
update_every_n_steps = 1
lock_timeout_ms = 100  # Max time to wait for weight lock
```

## Training Flow

```bash
# 1. Start Atropos environment (pointing to our server port)
python environments/gsm8k_server.py serve \
  --openai.base_url http://localhost:8000/v1 \
  --openai.api_key dummy

# 2. Start unified training (launches server + trainer)
mlx_parallm_train --config train_configs/grpo_qwen3_lora.toml

# The above command will:
# - Load the model (with LoRA if resuming)
# - Start the inference server on port 8000
# - Wait for server to be ready
# - Register with Atropos
# - Begin training loop
# - Update weights in-memory
# - Save checkpoints periodically

# 3. Resume from checkpoint if needed
mlx_parallm_train \
  --config train_configs/grpo_qwen3_lora.toml \
  --resume checkpoints/step_5000/

# 4. After training, merge adapter into base model
mlx_parallm_merge_lora \
  --base-model NousResearch/Hermes-4-Qwen3-14B-1-e3 \
  --lora-path checkpoints/final/lora_weights.safetensors \
  --output-path models/qwen3-14b-gsm8k-final \
  --push-to-hub username/model-name  # Optional
```

## Process Architecture Diagram

```
┌─────────────────────────────────────────────┐
│         mlx_parallm_train Process           │
│                                              │
│  ┌────────────────┐  ┌──────────────────┐   │
│  │ Inference      │  │ Training Thread   │   │
│  │ Server Thread  │  │                   │   │
│  │                │  │ - Fetch rollouts  │   │
│  │ - Handle reqs  │  │ - GRPO training   │   │
│  │ - Batching     │◄─┤ - Weight updates  │   │
│  │ - Generation   │  │ - Checkpointing   │   │
│  └────────────────┘  └──────────────────┘   │
│           ▲                   ▲              │
│           │                   │              │
│           └───────┬───────────┘              │
│                   │                          │
│          Shared MLX Model                    │
│          (Zero-copy weights)                 │
└─────────────────────────────────────────────┘
              ▲                ▲
              │                │
      Port 8000 HTTP    Atropos API
              │                │
    ┌─────────────┐    ┌──────────────┐
    │  Atropos    │    │  Atropos     │
    │  GSM8K Env  │    │  API Server  │
    └─────────────┘    └──────────────┘
```

## Success Metrics
- Single command to start training + serving
- Serve models with LoRA adapters at full speed
- Train Qwen3-14B with LoRA using <24GB memory
- <10ms in-process weight update latency
- Zero serialization overhead
- Support 100+ tok/s during training
- Graceful shutdown and resume

## Future Extensions
- Multi-LoRA serving (different adapters per request)
- Distributed training across multiple Macs (would need process separation)
- Advanced techniques: LoRA+, AdaLoRA, QLoRA variants
- Multiple environment support in single trainer
- Dynamic batching optimization during training

## Implementation Progress Tracking

### Phase 1: LoRA Inference Support
- [ ] Create `mlx_parallm/models/lora.py` module
- [ ] Add LoRA loading to model initialization
- [ ] Integrate with BatchedKVCache
- [ ] Add CLI arguments for adapter paths
- [ ] Test serving with pre-trained adapters

### Phase 2: Core GRPO Training
- [ ] Create `mlx_parallm/rl_training/` directory structure
- [ ] Port GRPO algorithm to `grpo_trainer.py`
- [ ] Implement `rollout_buffer.py`
- [ ] Create `trainer_base.py` abstract class
- [ ] Test training loop with dummy data

### Phase 3: Unified Process Architecture
- [ ] Create `train.py` unified launcher
- [ ] Implement thread-safe weight sharing
- [ ] Build `atropos_client.py`
- [ ] Test with gsm8k environment
- [ ] Implement graceful shutdown

### Phase 4: LoRA Training & Updates
- [ ] Implement LoRA-specific training
- [ ] Create `weight_updater.py`
- [ ] Test dynamic updates during serving
- [ ] Benchmark memory usage

### Phase 5: Tools & Utilities
- [ ] Create `merge_lora.py` tool
- [ ] Implement `checkpoint.py` module
- [ ] Test checkpoint/resume cycle
- [ ] Add format conversion utilities

### Phase 6: Testing & Optimization
- [ ] Full integration test with gsm8k
- [ ] Performance benchmarking
- [ ] Memory optimization
- [ ] Documentation and examples