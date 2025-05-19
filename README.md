# mlx_parallm Server

A high-performance, parallelized batch generation server for MLX models, supporting text generation, embeddings, and more, with an OpenAI-compatible API.

## Project Goals

-   High-performance inference for various MLX models (LLMs, LSTMs, RMs, Classifiers, etc.).
-   OpenAI-compatible API endpoints (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`).
-   Support for LoRA/DoRA adapters and efficient weight updates for RL rollouts.
-   Dynamic batching and request queuing.
-   Multimodal input support.
-   Quantized model support.
-   Distributed operation capabilities.

## Setup and Installation

1.  **Clone the Repository (Example)**:
    ```bash
    # git clone https://your-repository-url/mlx_parallm.git
    # cd mlx_parallm
    ```

2.  **Create and Activate a Python Virtual Environment**:
    It is highly recommended to use a virtual environment. You can create one using `uv` or Python's built-in `venv` module.

    Using `uv` (recommended):
    ```bash
    uv venv
    source .venv/bin/activate
    ```
    Or using `venv`:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Project Dependencies**:
    This project uses `pyproject.toml` to manage dependencies with `uv`. Install the project in editable mode, which will also install all dependencies:
    ```bash
    uv pip install -e .
    ```

4.  **Hugging Face Hub Authentication (Important for Gated/Private Models)**:
    If you plan to load models that are gated or private on the Hugging Face Hub, you need to authenticate. You can do this in one of two ways:
    *   **Login via CLI (recommended for interactive use):**
        ```bash
        huggingface-cli login
        ```
    *   **Set Environment Variable:**
        Set the `HF_TOKEN` environment variable to your Hugging Face access token (with read permissions).
        ```bash
        export HF_TOKEN=your_hf_token_here
        ```
    Ensure you have also been granted access to the specific model repository on the Hugging Face website.

## Running the Server

Once installed, you can start the server using the `mlx_parallm_serve` command:

```bash
mlx_parallm_serve --model-path <path_or_hf_id_to_your_model> --host <address> --port <port_number>
```

For example:
```bash
mlx_parallm_serve --model-path mistralai/Mistral-7B-Instruct-v0.1 --port 8000
```

This will start the Uvicorn server with the FastAPI application. You can then access the API endpoints, for example, the health check:
`http://127.0.0.1:8000/health` (assuming default host and port 8000).

## Dependency Management

This project uses `pyproject.toml` as the single source of truth for dependencies, managed with `uv`.

*   **To add a new runtime dependency**:
    ```bash
    uv add <package_name>
    ```
*   **To add a new development dependency**:
    ```bash
    uv add <package_name> --dev
    ```
    This will add the package to `pyproject.toml` and install it into your environment.

*   **To ensure a reproducible environment (using a lockfile)**:
    1.  Generate a lockfile (e.g., `requirements.lock.txt`):
        ```bash
        uv pip compile pyproject.toml --output-file requirements.lock.txt
        ```
    2.  Install dependencies from the lockfile:
        ```bash
        uv sync --locked requirements.lock.txt
        ```
    It's good practice to commit both `pyproject.toml` and the generated lockfile to your repository.

## Development

(Details about development workflows, running tests, etc., will be added here as the project progresses.)

## Contributing

(Guidelines for contributing to the project will be added here.)

# MLX ParaLLM

Batched KV caching for fast parallel inference on Apple Silicon devices, via [MLX](https://github.com/ml-explore/mlx). 

This repo heavily borrows from [`mlx_lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). Will explore how to add batched generation there as a non-breaking PR. 


## Usage
Requires `mlx` and `mlx_lm` to be installed.
```python
from mlx_parallm.utils import load, batch_generate
model, tokenizer = load("google/gemma-1.1-2b-it")
prompts = ["prompt_0", ..., "prompt_k"]
responses = batch_generate(model, tokenizer, prompts=prompts_raw[:10], max_tokens=100, verbose=True, format_prompts=True, temp=0.0)
```

## Models
Models tested: 
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-1.1-2b-it`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit`
- `mlx-community/Phi-3-mini-4k-instruct-4bit`
- `mlx-community/gemma-1.1-2b-it-4bit`


Both quantized and `float16` models are supported. `float16` models seem to generally perform faster if sufficient RAM is available (up to 1300+ tok/s throughput for `gemma-2b` on M3 Max 128GB).

Additional models can be added by copying architecture files from [`mlx_lm/models`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm/models) and replacing any references to `KVCache` with `BatchedKVCache`. 

## Features
Supported:
- `batch_generate` method (tested with `len(prompts) > 500`)
- Auto-padding
- Auto-formatting with prompt templates (`format_prompts=True`)
- `temp = 0`, `temp > 0`, `top_p` sampling
- single-stream `generate` method 

Not (yet) supported: 
- Repetition penalties
- Streaming outputs for `batch_generate`
- Dynamic batching for async requests