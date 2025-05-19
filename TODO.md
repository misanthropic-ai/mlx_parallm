# TODO: mlx_parallm High-Performance Generation Server

This document outlines the plan to develop `mlx_parallm` into a parallelized, high-performance batch generation server, similar to vLLM or sglang, with support for LoRA/DoRA adapters, RL rollouts, and Reward Model (RM) functionalities.

## I. Core Server Infrastructure (FastAPI & Uvicorn)

*   [ ] **Set up FastAPI Application:**
    *   [ ] Initialize a basic FastAPI project structure.
    *   [ ] Add `fastapi` and `uvicorn` to project dependencies (e.g., `requirements.txt` or `pyproject.toml`).
*   [ ] **Implement Basic Endpoints:**
    *   [ ] `/health`: Health check endpoint.
    *   [ ] `/models`: Endpoint to list available/loaded models.
*   [ ] **Configuration Management:**
    *   [ ] Implement a system for managing server configurations (e.g., host, port, model paths) using environment variables or a configuration file.

## II. Model Management & Loading

*   [ ] **Base Model Loading:**
    *   [ ] Implement logic to load base Hugging Face models compatible with `mlx_lm`.
    *   [ ] Allow specification of model path/ID via API or configuration.
*   [ ] **LoRA/DoRA Adapter Management:**
    *   [ ] Implement logic to load LoRA/DoRA adapters and apply them to a base model using `mlx_lm` utilities.
    *   [ ] API endpoint to list available/loaded adapters.
    *   [ ] API endpoint to load/unload adapters dynamically.
    *   [ ] Consider how to handle multiple adapters for the same base model.
*   [ ] **Model Cache/Registry:**
    *   [ ] Implement a system to manage loaded models and adapters to avoid redundant loading.
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
    *   [ ] Implement an asynchronous request queue to manage incoming generation requests.
    *   [ ] Consider priority queues if RL or other latency-sensitive requests need precedence.
*   [ ] **Dynamic Batching Strategy:**
    *   [ ] Design and implement a dynamic batching mechanism:
        *   [ ] Collect requests from the queue.
        *   [ ] Group requests into batches based on criteria like max batch size, timeout (configurable, potentially shorter for RL).
        *   [ ] Pad requests within a batch to the same length or handle variable lengths efficiently for text; handle tensor batching for NNs.
    *   [ ] Research and potentially adapt continuous batching techniques (especially for LLMs).
    *   [ ] Optimize for low-latency, especially for RL inference requests.
*   [ ] **Request Validation:**
    *   [ ] Implement Pydantic models for API request/response validation.

## IV. Generation Engine

*   [ ] **Integrate `mlx_lm` Generation:**
    *   [ ] Wrap `mlx_lm.generate` or equivalent functions for batched text generation.
    *   [ ] Ensure efficient handling of tokenization and detokenization for batches.
*   [ ] **Tokenizer Management:**
    *   [ ] Ensure tokenizer is loaded alongside the model and used consistently.
*   [ ] **OpenAI-Compatible API Endpoints:**
    *   [ ] **`/v1/completions` Endpoint:**
        *   [ ] Implement for raw text generation, compatible with OpenAI's completions API.
        *   [ ] Accept parameters: `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `n`, `stream`, `logprobs`, `stop`, `presence_penalty`, `frequency_penalty`, etc.
    *   [ ] **`/v1/chat/completions` Endpoint:**
        *   [ ] Implement for chat-based generation, compatible with OpenAI's chat completions API.
        *   [ ] Accept `messages` array with roles (`system`, `user`, `assistant`, `tool`).
        *   [ ] Implement template processing for standard chat model formats (e.g., applying chat templates).
        *   [ ] Support `tool_choice` and `tools` parameters for function calling/tool usage.
        *   [ ] Ensure consistent request/response formats with OpenAI specifications.
        *   [ ] Support streaming generated tokens back to the client.
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

*   [ ] **Dependency Management:**
    *   [ ] Finalize `requirements.txt` or `pyproject.toml` with all dependencies and versions.
*   [ ] **Containerization (Docker):**
    *   [ ] Create a `Dockerfile` for easier deployment.
*   [ ] **README Update:**
    *   [ ] Update project README with setup, usage, and API documentation.

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