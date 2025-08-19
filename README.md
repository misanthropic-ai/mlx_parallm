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

### Examples

**Mistral 7B:**
```bash
mlx_parallm_serve --model-path mistralai/Mistral-7B-Instruct-v0.1 --port 8000
```

**Qwen3 14B (Hermes fine-tune):**
```bash
mlx_parallm_serve --model-path NousResearch/Hermes-4-Qwen3-14B-1-e3 --port 8000
```

This will start the Uvicorn server with the FastAPI application. You can then access the API endpoints, for example, the health check:
`http://127.0.0.1:8000/health` (assuming default host and port 8000).

## Server API: Completions + Logprobs

### Text Completions

**Basic completion (non-streamed):**
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<your_model_id>",
    "prompt": "Say hello in one sentence.",
    "max_tokens": 16,
    "temperature": 0.7,
    "top_p": 1.0
  }'
```

**With Qwen3 model:**
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq -r '.choices[0].text'
```

**Token logprobs and echo (OpenAI-style):**
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<your_model_id>",
    "prompt": "Say hello in one sentence.",
    "max_tokens": 8,
    "temperature": 0.0,
    "logprobs": 5,
    "echo": true
  }'
```
- Response includes `choices[0].logprobs` with `tokens`, `token_logprobs`, `top_logprobs`, `text_offset`.

### Chat Completions

**Basic chat completion:**
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is quantum computing?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
```

**Streaming chat completion:**
```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding"}
    ],
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": true
  }'
```

### Batched Inference (Multiple Concurrent Requests)

The server automatically batches multiple concurrent requests for improved throughput. Here's a Python example:

```python
import asyncio
import aiohttp
import json

async def make_request(session, prompt, request_id):
    url = "http://localhost:8000/v1/completions"
    payload = {
        "model": "NousResearch/Hermes-4-Qwen3-14B-1-e3",
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7
    }
    async with session.post(url, json=payload) as response:
        result = await response.json()
        return f"Request {request_id}: {result['choices'][0]['text'][:50]}..."

async def test_batching():
    prompts = [
        "The meaning of life is",
        "Machine learning is",
        "Python was created by",
        "Climate change affects",
        "Space exploration helps"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result)

# Run the test
asyncio.run(test_batching())
```

The server will automatically batch these requests (up to 8 at a time by default) for efficient parallel processing.

### Raw Completions vs Chat

- `/v1/completions` is raw: it does not add chat roles or special tokens. Your `prompt` is used as-is.
- `/v1/chat/completions` applies the tokenizer's chat template and inserts role tokens automatically.
  Use this for assistant-style chat; use `/v1/completions` when you want precise control over the input sequence.

## Perplexity via Logprobs (Quick Start)

- You can approximate perplexity by summing negative log probabilities of the target text. Use `echo: true` to get logprobs over prompt tokens.

- Example (token logprobs over a prompt):
  ```bash
  curl -s http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "<your_model_id>",
      "prompt": "Alexander Grothendieck became a French citizen in 1971.",
      "max_tokens": 1,
      "temperature": 0.0,
      "logprobs": 5,
      "echo": true
    }' | jq '.choices[0].logprobs'
  ```
  - Sum `-token_logprobs` to get negative log-likelihood; divide by token count to get average NLL; exponentiate for perplexity.

- Python sketch:
  ```python
  import requests, math
  data = {
      "model": "<your_model_id>",
      "prompt": "some text...",
      "max_tokens": 1,      # minimal generation; echo provides prompt logprobs
      "temperature": 0.0,
      "logprobs": 5,
      "echo": True,
  }
  r = requests.post("http://127.0.0.1:8000/v1/completions", json=data).json()
  lps = r["choices"][0]["logprobs"]["token_logprobs"]
  avg_nll = -sum(lps)/len(lps)
  ppl = math.exp(avg_nll)
  print({"avg_nll": avg_nll, "ppl": ppl})
  ```

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
- `NousResearch/Hermes-4-Qwen3-14B-1-e3` (Qwen3)
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
