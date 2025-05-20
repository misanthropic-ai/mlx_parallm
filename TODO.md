# TODO: mlx_parallm High-Performance Generation Server

This document outlines the plan to develop `mlx_parallm` into a parallelized, high-performance batch generation server, similar to vLLM or sglang, with support for LoRA/DoRA adapters, RL rollouts, and Reward Model (RM) functionalities.

## I. Core Server Infrastructure (FastAPI & Uvicorn)

*   [x] **Set up FastAPI Application:**
    *   [x] Initialize a basic FastAPI project structure.
    *   [x] Add `fastapi` and `uvicorn` to project dependencies (e.g., `pyproject.toml`).
*   [x] **Implement Basic Endpoints:**
    *   [x] `/health`: Health check endpoint.
    *   [x] `/v1/models`: Endpoint to list available/loaded models. (Initial version done)
*   [x] **CLI for Server Launch:**
    *   [x] Implement CLI arguments (`--model-path`, `--host`, `--port`) using `pydantic-cli`.
    *   [x] Launch Uvicorn server with specified arguments.
*   [ ] **Configuration Management:**
    *   [ ] Implement a system for managing server configurations (e.g., host, port, model paths) using environment variables or a configuration file.
    *   [ ] Configuration should support specifying multiple models (e.g., for startup loading or as a list of discoverable/available models).

## II. Model Management & Loading

*   [x] **Model Loading and Lifecycle Management:**
    *   [x] Implement logic to load base Hugging Face models compatible with `mlx_lm`. (Initial version for single model on startup)
    *   [x] Allow specification of model path/ID via API or configuration. (Via CLI on startup)
    *   [x] Design a clear model identifier scheme (e.g., user-provided name, derived from path/ID). (Using path as ID for now)
*   [ ] **LoRA/DoRA Adapter Management:**
    *   [ ] Implement logic to load LoRA/DoRA adapters and apply them to a base model using `mlx_lm` utilities.
    *   [ ] API endpoint to list available/loaded adapters.
    *   [ ] API endpoint to load/unload adapters dynamically.
    *   [ ] Consider how to handle multiple adapters for the same base model.
*   [x] **Model Cache/Registry:**
    *   [x] Implement a central model registry to store and manage multiple loaded model instances and their metadata. (Initial in-memory version done)
    *   [x] Metadata should include: identifier (ID), path/source, type (e.g., causal_lm, embedding, classifier), status (enum: 'LOADING', 'LOADED', 'ERROR_LOADING', 'AVAILABLE_NOT_LOADED'), quantization details, associated tokenizer/processor, creation/load timestamp. (Basic fields and status enum implemented)
    *   [x] The registry will be the source of truth for the `/v1/models` endpoint and internal model lookups. (Implemented)
*   [ ] **Embedding Model Loading:**
    *   [ ] Implement logic to load embedding models (e.g., sentence transformers, other models with pooling heads) compatible with MLX.
    *   [ ] Support different pooling strategies if necessary.
*   [ ] **Classifier Model Loading (e.g., BERT-style):**
    *   [ ] Implement logic to load pre-trained classifier models compatible with MLX.
    *   [ ] Handle tokenization and input formatting specific to these architectures.
*   [ ] **General NN Model Loading (LSTMs, MLPs):**
    *   [ ] Implement logic to load pre-trained LSTM, MLP models compatible with MLX.
    *   [ ] Define a mechanism for specifying model input/output signatures (shapes, dtypes).
*   [ ] **Quantized Model Loading (for Inference):**
    *   [ ] Implement logic to load quantized models (e.g., using `mlx-lm` quantization utilities or other MLX-compatible formats like GGUF if applicable for transformers, or custom quantization schemes for NNs).
    *   [ ] Ensure inference pathways can leverage these quantized weights efficiently.

## III. Request Handling & Batching

*   [ ] **Request Queue:**
    *   [ ] Implement an asynchronous request queue (e.g., `asyncio.Queue`) in `server/main.py`.
    *   [ ] Define a structure for queue items (request data, response future/event, stream queue if applicable).
    *   [ ] API endpoints (`/v1/completions`, `/v1/chat/completions`) to place requests into this queue and await results.
    *   [ ] Consider priority queues if RL or other latency-sensitive requests need precedence (future).
*   [ ] **Batching Worker Task:**
    *   [ ] Implement a background `asyncio` task that continuously dequeues requests.
    *   [ ] This task will be responsible for forming batches and invoking model inference.
    *   [ ] Start this worker task on FastAPI application startup.
*   [ ] **Dynamic Batching Strategy:**
    *   [ ] Design and implement a dynamic batching mechanism within the worker:
        *   [ ] Collect requests from the queue (e.g., up to `max_batch_size` or a timeout).
        *   [ ] **Handle `n` parameter:** If a request has `n > 1`, replicate the prompt/messages `n` times for batch processing. Keep track of these replications to correctly group results.
        *   [ ] Tokenize and pad all prompts in the batch (including replications) to the same length (left-padding for generation).
        *   [ ] Call appropriate batched generation utilities from `mlx_parallm.utils` (e.g., a new `batch_generate_text` for non-streaming, `batch_stream_generate_text` for streaming).
        *   [ ] Distribute results/deltas back to the corresponding original client requests (managing their response futures/events/stream queues).
    *   [ ] Research and potentially adapt continuous batching techniques (especially for LLMs) (future).
    *   [ ] Optimize for low-latency, especially for RL inference requests (future).
*   [x] **Request Validation:**
    *   [x] Implement Pydantic models for API request/response validation. (Done for implemented endpoints)

## IV. Generation Engine

*   [x] **Integrate `mlx_lm` Generation:**
    *   [x] Wrap `mlx_lm.generate` (via `mlx_parallm.utils.generate`) for text generation. (Single prompt generation integrated)
    *   [x] Wrap `mlx_lm.stream_generate` (via `mlx_parallm.utils.stream_generate`) for single prompt streaming. (Integrated)
    *   [ ] Implement/Refine `mlx_parallm.utils.batch_generate_text` for batched non-streaming generation supporting the `n` parameter.
    *   [x] Implement `mlx_parallm.utils.batch_stream_generate_text` for batched streaming generation. (Initial version for `n=1` per sequence in batch done, needs to correctly support `n` choices per original prompt in the stream output formatting).
    *   [x] Ensure efficient handling of tokenization and detokenization. (Basic handling implemented)
*   [x] **Tokenizer Management:**
    *   [x] Ensure tokenizer is loaded alongside the model and used consistently. (Implemented for startup model)
*   [x] **OpenAI-Compatible API Endpoints:**
    *   [x] **`/v1/models` Endpoint (OpenAI Compatible):**
        *   [x] Implement to list models known to the server.
        *   [x] Response format aligns with OpenAI's `GET /v1/models`.
        *   [x] Model object includes `id`, `object`, `created`, `owned_by`, `status`, `type`.
        *   [x] Queries the Model Cache/Registry.
    *   [x] **`/v1/completions` Endpoint:**
        *   [x] Implement for raw text generation, compatible with OpenAI's completions API. (Initial version done, streaming added)
        *   [x] Accept parameters: `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `stream`, `n`. (Implemented)
        *   [ ] Fully support `n > 1` completions (requires batching worker integration for non-streaming and correct SSE formatting for streaming `n` choices).
        *   [ ] `logprobs`, `stop`, `presence_penalty`, `frequency_penalty`, etc. (Remaining parameters to be added)
    *   [x] **`/v1/chat/completions` Endpoint:**
        *   [x] Implement for chat-based generation, compatible with OpenAI's chat completions API. (Initial version done, streaming added)
        *   [x] Accept `messages` array with roles (`system`, `user`, `assistant`).
        *   [x] Accept `model`, `max_tokens`, `temperature`, `top_p`, `stream`, `n`. (Implemented)
        *   [x] Implement template processing using `tokenizer.apply_chat_template`.
        *   [ ] Fully support `n > 1` completions (requires batching worker integration for non-streaming and correct SSE formatting for streaming `n` choices).
        *   [ ] `tool_choice` and `tools` parameters for function calling/tool usage.
        *   [x] Ensure consistent request/response formats with OpenAI specifications (for basic fields and streaming structure).
        *   [ ] Accept LoRA adapter ID (optional, custom parameter or via model name convention).
    *   [ ] **`/v1/embeddings` Endpoint:**
        *   [ ] Implement for generating text embeddings, compatible with OpenAI's embeddings API.
        *   [ ] Accept parameters: `model`, `input` (string or array of strings/token arrays), `encoding_format`, `dimensions` (for Matryoshka models).
        *   [ ] Return embedding vectors in the specified format.

## V. Adapter Weight Updates (for RL Policy Updates)

*   [ ] **Weight Update API Endpoint:**
    *   [ ] Create an API endpoint (e.g., `/update_adapter_weights`).
    *   [ ] Accept adapter ID and new weights (serialized format to be determined).
*   [ ] **Efficient Weight Update Mechanism:**
    *   [ ] Investigate `mlx_lm`'s capabilities for efficient LoRA weight updates.
    *   [ ] Implement logic to apply updates to the target adapter with minimal disruption to ongoing generation requests. This might involve:
        *   [ ] Hot-swapping adapter layers.
        *   [ ] Briefly pausing new request batching, updating, then resuming.
        *   [ ] Leveraging MLX's architecture for efficient parameter updates.
*   [ ] **Synchronization:**
    *   [ ] Ensure that weight updates are properly synchronized if multiple worker processes are used on a single machine or across multiple nodes in a cluster (See Section XII: Distributed Operation & Clustering).

## VI. Reinforcement Learning (RL) Integration

*   [ ] **RL-Specific Endpoint (e.g., `/v1/rl/act` or `/v1/rl/rollout`):**
    *   [ ] Design a dedicated API endpoint optimized for RL inference loops.
    *   [ ] **Inputs:**
        *   [ ] `model_id` (policy model, potentially an adapter ID).
        *   [ ] Batched `observations` (can be complex/structured, e.g., tensors or lists of tensors).
        *   [ ] Optional `hidden_states` for recurrent policies (input from previous step).
        *   [ ] `deterministic` flag (boolean, for sampling vs. greedy actions).
    *   [ ] **Outputs:**
        *   [ ] Batched `actions` (sampled or greedy).
        *   [ ] `new_hidden_states` (for recurrent policies, to be fed into next step).
        *   [ ] Optional `value_estimates` (from a value head in the policy model).
        *   [ ] Optional `action_log_probs`.
    *   [ ] Ensure efficient serialization/deserialization of tensor data (e.g., base64 encoded, or explore more direct methods if feasible with FastAPI).
*   [ ] **Action Sampling:**
    *   [ ] Implement action sampling logic (e.g., categorical for discrete, Gaussian for continuous) based on model outputs (logits/distributions), potentially executed on-device (MLX).
*   [ ] **Recurrent State Management:**
    *   [ ] Provide robust mechanisms for clients to pass and receive hidden states for recurrent policies.
    *   [ ] Consider server-side caching of states for very short episodes if beneficial, though client-managed is usually more flexible.
*   [ ] **Policy Management:**
    *   [ ] If multiple RL agents/policies might be active, ensure the server can differentiate and use the correct adapter weights or model for each rollout request.
    *   [ ] Link with adapter weight update mechanism for policy iterations.

## VII. Reward Model (RM) Functionality

*   [ ] **RM Inference Endpoint:**
    *   [ ] Create an API endpoint (e.g., `/v1/rewards/score` or align with OpenAI-like patterns, possibly using a generic pooling endpoint if applicable like vLLM's `/v1/pool` with `task="reward"`) to get scores from a loaded RM.
    *   [ ] Accept text input(s) and return scalar reward(s).
    *   [ ] Implement loading and management of RM models (which might also be LoRA-adapted models).
*   [ ] **(Stretch Goal) RM Training Support:**
    *   [ ] Define requirements for "training RMs."
    *   [ ] If it involves fine-tuning via the server, design API endpoints for submitting training data and managing training jobs.
    *   [ ] This is a significant feature and might be deferred.

## VIII. Embedding Model Features

*   [ ] **Matryoshka Embeddings Support:**
    *   [ ] Investigate and implement support for models trained with Matryoshka Representation Learning (MRL).
    *   [ ] Allow users to specify desired `dimensions` for embeddings if the loaded model supports it.
    *   [ ] Handle model configuration to identify Matryoshka-compatible models (e.g., checking `config.json` for `is_matryoshka` or `matryoshka_dimensions`).

## IX. Classifier and Generic Model Support

*   [ ] **Classifier Model Inference:**
    *   [ ] **`/v1/classifications` Endpoint:**
        *   [ ] Implement an OpenAI-compatible or similarly structured endpoint for classification tasks.
        *   [ ] Accept parameters: `model`, `input` (string or list of strings).
        *   [ ] Return outputs like class probabilities or labels per input.
        *   [ ] Leverage classifier models loaded in Section II.
*   [ ] **General NN Model Inference (LSTMs, MLPs):**
    *   [ ] **Generic `/v1/predict` Endpoint:**
        *   [ ] Design and implement a flexible endpoint for models not fitting specialized categories (e.g., LSTMs, MLPs).
        *   [ ] Accept `model_id` and a structured `inputs` field (e.g., JSON representing named input tensors).
        *   [ ] Return model outputs in a structured format (e.g., JSON representing output tensors).
        *   [ ] Requires robust input validation based on the model's signature.

## X. Vision Model Support (Multimodal)

*   [ ] **Multimodal Model Loading:**
    *   [ ] Extend model loading capabilities to support vision-language models (VLMs) compatible with MLX.
    *   [ ] Manage loading of both text and vision encoders.
*   [ ] **Image Input Processing:**
    *   [ ] Modify request handling to accept image inputs within the `/v1/chat/completions` endpoint messages, following OpenAI's format:
        ```json
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image_string}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.png"}
                }
            ]
        }
        ```
    *   [ ] Implement logic to decode base64 encoded images.
    *   [ ] Implement logic to download images from URLs.
    *   [ ] Perform necessary image preprocessing/transformations required by the VLM.
*   [ ] **Multimodal Generation:**
    *   [ ] Adapt the generation engine to handle combined text and image embeddings.
    *   [ ] Ensure the batching mechanism can accommodate multimodal inputs.
*   [ ] **KV Cache for Multimodal Inputs:**
    *   [ ] Investigate and implement correct handling of image features/embeddings within the KV cache.
    *   [ ] Ensure that the caching strategy is efficient for sequences involving both text and image tokens.
*   [ ] **Tokenizer and Image Processor Management:**
    *   [ ] Load and manage appropriate tokenizers and image processors for VLMs.

## XI. Performance, Parallelism & Optimization

*   [ ] **Asynchronous Operations:**
    *   [ ] Ensure all I/O-bound operations and model inference calls are asynchronous (`async/await`) to prevent blocking the server.
*   [ ] **MLX Optimizations:**
    *   [ ] Leverage MLX features for efficient computation on Apple Silicon (e.g., `mx.eval`).
    *   [ ] Explore `mlx.compile` for potential performance gains on critical code paths (e.g., policy forward pass, batch processing).
*   [ ] **Quantized Model Inference Performance:**
    *   [ ] Ensure that inference with loaded quantized models yields expected performance benefits (speed, memory).
    *   [ ] Benchmark quantized vs. full-precision models.
*   [ ] **Low-Latency Optimizations for RL:**
    *   [ ] Profile and optimize the entire RL inference path (request -> batch -> model -> sample -> response).
    *   [ ] Minimize data copying and conversion overheads.
*   [ ] **Multi-threading/Processing (if needed beyond MLX's capabilities):**
    *   [ ] Evaluate if Python's `asyncio` is sufficient or if `multiprocessing` is needed for true parallelism of Python code around MLX calls (primarily for single-node, multi-GPU/CPU core parallelism).
    *   [ ] If using multiple processes, manage inter-process communication for request batching and model updates on a single machine.
*   [ ] **Memory Management:**
    *   [ ] Monitor and optimize memory usage, especially with multiple models/adapters.
    *   [ ] Investigate techniques like PagedAttention (if adaptable to MLX or if `mlx-lm` internals offer similar benefits).
*   [ ] **Benchmarking & Profiling:**
    *   [ ] Set up tools for benchmarking throughput, latency, and resource utilization.
    *   [ ] Profile code to identify and address bottlenecks.

## XII. Distributed Operation & Clustering

*   [ ] **Define Strategy for Distributed Deployment:**
    *   [ ] Evaluate different models: e.g., independent, stateless server instances behind a load balancer vs. a coordinated cluster of server instances for shared state/tasks.
*   [ ] **Adapter Weight Synchronization Across Cluster Nodes:**
    *   [ ] Design and implement a mechanism for propagating LoRA/DoRA adapter weight updates to all relevant nodes in a cluster.
    *   [ ] Investigate using `mlx.distributed` primitives (e.g., `mx.distributed.broadcast`, `mx.distributed.all_gather`) with MPI or Ring backends for efficient synchronization.
    *   [ ] Integrate with the `/update_adapter_weights` API to trigger distributed updates.
*   [ ] **(Potential) Distributed Request Dispatch & Load Balancing:**
    *   [ ] If using a coordinated cluster, design how incoming requests are dispatched to worker nodes.
    *   [ ] Consider strategies for load balancing across the cluster.
*   [ ] **(Advanced) Distributed Inference for Large Models/Batches:**
    *   [ ] Explore if `mlx.distributed` can be used to shard inference for very large models (model parallelism) or very large batches (data parallelism for inference) across multiple nodes/devices if single-node performance is insufficient. This is a significant undertaking.
*   [ ] **Configuration & Launching for Distributed Mode:**
    *   [ ] Document procedures for setting up and launching the server in a distributed configuration.
    *   [ ] This includes managing hostfiles (e.g., JSON for MLX Ring backend) or MPI configurations.
    *   [ ] Provide examples for using `mlx.launch` if it is leveraged for managing distributed MLX processes.
*   [ ] **Fault Tolerance & Resilience (Considerations):**
    *   [ ] Outline basic strategies or considerations for handling node failures in a distributed setup (e.g., impact on ongoing requests, state recovery if any).
*   [ ] **Service Discovery (Considerations):**
    *   [ ] Note potential need for service discovery mechanisms (e.g., Consul, etcd, Kubernetes services) for instances to find each other or for clients to find the service, especially in dynamic environments.

## XIII. Build, Packaging & Deployment

*   [x] **Dependency Management (with `pyproject.toml` and `uv`):**
    *   [x] Initialize `pyproject.toml` as the primary source for project metadata and dependencies.
    *   [ ] Use `uv pip install -e .` to install the project in editable mode and its dependencies for development.
    *   [ ] For adding new packages: 
        *   Use `uv add <package_name>` to add runtime dependencies.
        *   Use `uv add <package_name> --dev` to add development dependencies (they will go into `[project.optional-dependencies]` or a `[tool.uv.dev-dependencies]` table).
    *   [ ] `uv add` will update `pyproject.toml` and install the package.
    *   [ ] Generate and maintain a lockfile (e.g., `requirements.lock.txt` or `uv.lock`) using `uv pip compile pyproject.toml --output-file <lockfile_name>` and use `uv sync --locked <lockfile_name>` for reproducible environments.
    *   [ ] Remove `requirements.txt` if it's no longer used for active dependency management (or regenerate it from `pyproject.toml` if needed for specific tooling).
*   [ ] **Containerization (Docker):**
    *   [ ] Create a `Dockerfile` for easier deployment.
*   [x] **CLI Entrypoint (`mlx_parallm.serve`):**
    *   [x] Implement a command-line interface using `pydantic-cli`.
    *   [x] Create an entrypoint in `pyproject.toml` under `[project.scripts]` (`mlx_parallm_serve = "mlx_parallm.cli:cli_runner"`).
    *   [ ] **Arguments/Options (to be expanded):**
        *   [x] `--model-path` (or `--model`): Specifies a model to load at startup. Consider making this repeatable (`--load-model <id_or_path>`) or using a configuration file to specify multiple models for initial loading.
        *   [x] `--host`: Host to bind the server to (default: `127.0.0.1`).
        *   [x] `--port`: Port to bind the server to (default: `8000`).
        *   [ ] `--workers`: Number of Uvicorn workers (if applicable, consider implications with MLX's own parallelism).
        *   [ ] `--log-level`: Logging level for the server.
        *   [ ] `--config-file`: Path to a server configuration file (to override defaults/CLI args).
        *   [ ] (Future) `--adapter-path`: Path to a LoRA/DoRA adapter to load initially.
        *   [ ] (Future) `--quantization`: Specify quantization method/bits if loading a quantized model directly.
    *   [ ] The CLI should parse these arguments and pass them to the server's configuration system.
    *   [ ] The CLI will programmatically start the Uvicorn server with the FastAPI app (`server.main:app`).
*   [x] **README Update:**
    *   [x] Update project README with setup, usage, API documentation, and CLI usage instructions (initial version created).

## XIV. Documentation & Testing

*   [ ] **API Documentation:**
    *   [ ] Leverage FastAPI's automatic OpenAPI/Swagger documentation.
    *   [ ] Add detailed descriptions and examples for each endpoint.
*   [ ] **Unit Tests:**
    *   [ ] Write unit tests for core components (batching logic, API handlers, utility functions).
*   [ ] **Integration Tests:**
    *   [ ] Write integration tests for end-to-end workflows (request -> batch -> generate -> response).
*   [ ] **Load Testing:**
    *   [ ] Perform load testing to assess performance under stress.

## XV. Code Structure and Organization

*   [ ] **Modular Design:**
    *   [ ] Organize code into logical modules (e.g., `server`, `models`, `generation`, `batching`).
    *   [ ] Refactor existing `utils.py` and `sample_utils.py` into the new structure as appropriate.
*   [ ] **Logging:**
    *   [ ] Implement structured logging throughout the application.

## Notes & Considerations:

*   **Error Handling:** Robust error handling and informative error responses for the API.
*   **Security:** Basic security considerations for API endpoints.
*   **Scalability:** Design with future scalability in mind (e.g., ability to run multiple server instances).
*   **`mlx-lm` Evolution:** Keep an eye on `mlx-lm` library updates, as they might provide new features or more efficient ways to implement parts of this plan. 