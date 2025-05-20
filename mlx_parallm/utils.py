# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from mlx.utils import tree_flatten
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

# mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_lm.tuner.utils import load_adapters
from mlx_lm.tuner.utils import dequantize as dequantize_model

# Local imports
from mlx_parallm.sample_utils import top_p_sampling
from mlx_parallm.models.base import BatchedKVCache

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

MAX_FILE_SIZE_GB = 5


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    #return Model, ModelArgs
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_parallm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face repo ID correctly.\n"
                "If you are trying to access a private or gated Hugging Face repo, ensure that:\n"
                "  1. You have been granted access to the repository on Hugging Face.\n"
                "  2. You are authenticated: run \`huggingface-cli login\` or set the HF_TOKEN environment variable.\n"
                "     (Details: https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login)"
            ) from None
    return model_path


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """

    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def generate_step(
    prompts: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        softmax_logits = mx.softmax(logits, axis=-1)

        if temp == 0:
            tokens = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            if top_p > 0 and top_p < 1.0:
                tokens = top_p_sampling(logits, top_p, temp)
            else:
                scaled_logits = logits * (1 / temp)
                tokens = mx.random.categorical(logits * (1 / temp), axis=-1)
                if scaled_logits.ndim > 1:
                    tokens = mx.expand_dims(tokens, axis=-1)

        probs = softmax_logits[0, tokens]
        return tokens, probs

    if repetition_penalty:
        raise NotImplementedError("repetition_penalty not supported.")

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    # (bs, ntoks)
    y = prompts
    kv_heads = (
        [model.n_kv_heads] * len(model.layers)
        if isinstance(model.n_kv_heads, int)
        else model.n_kv_heads
    )

    cache = [BatchedKVCache(model.head_dim, n, y.shape[0]) for n in kv_heads]

    repetition_context = prompts

    if repetition_context_size and repetition_penalty:
        repetition_context = repetition_context[:,-repetition_context_size:]

    def _step(y):
        nonlocal repetition_context
        logits = model(y, cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, probs = sample(logits)
            repetition_context = mx.concatenate([repetition_context, y])
        else:
            y, probs = sample(logits)

        if repetition_context_size:
            if repetition_context.shape[1] > repetition_context_size:
                repetition_context = repetition_context[:,-repetition_context_size:]
        return y, probs

    y, p = _step(y)
    mx.async_eval(y)
    while True:
        next_y, next_p = _step(y)
        mx.async_eval(next_y)
        mx.eval(y)
        yield y, p
        y, p = next_y, next_p

def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The ma
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing text.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))[None, :]
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    for (token, prob), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        # token is expected to be mx.array of shape (1,1) because prompt_tokens is (1, seq_len)
        token_item = token.item() # Get the integer token ID

        if token_item == tokenizer.eos_token_id: # Compare item with eos_token_id
            break
        detokenizer.add_token(token_item) # Pass the integer token ID

        # Yield the last segment if streaming
        yield detokenizer.last_segment

    detokenizer.finalize()
    yield detokenizer.last_segment

def batch_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    max_tokens: int = 100,
    verbose: bool = False,
    format_prompts: bool = True,
    formatter: Optional[Callable] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
    
    if format_prompts:
        prompts_fm = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts_fm = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in prompts_fm]
    else:
        prompts_fm = prompts

    # left-padding for batched generation
    tokenizer._tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts_toks = mx.array(tokenizer._tokenizer(prompts_fm, padding=True)['input_ids'])
    tic = time.perf_counter()

    output_toks = []
    for (tokens, _), n in zip(
        generate_step(prompts_toks, model, **kwargs),
        range(max_tokens),
    ): 
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        output_toks.append(tokens)
    output_toks = mx.concatenate(output_toks, axis=1)

    # detokenizing + stripping pad/eos tokens
    responses = [response.split(tokenizer.eos_token)[0].split(tokenizer.pad_token)[0] for response in tokenizer.batch_decode(output_toks.tolist())]
    if verbose:
        gen_time = time.perf_counter() - tic
        prompt_tps = prompts_toks.size / prompt_time
        gen_tps = output_toks.size / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        for prompt, response in zip(prompts, responses):
            print("=" * 10)
            print("Prompt:", prompt)
            print(response)
            
    return responses


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None]
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()

    for (token, prob), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())

        if verbose:
            if formatter:
                # We have to finalize so that the prob corresponds to the last segment
                detokenizer.finalize()
                formatter(detokenizer.last_segment, prob.item())
            else:
                print(detokenizer.last_segment, end="", flush=True)

    token_count = n + 1
    detokenizer.finalize()

    if verbose:
        gen_time = time.perf_counter() - tic
        print(detokenizer.last_segment, flush=True)
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    return detokenizer.text


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config(dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA/DoRA adapters. If provided, applies adapter layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, model_config)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer


def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy)
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
    return model, config, tokenizer


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = dedent(
        f"""
        # {upload_repo}

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path}) using mlx-lm version **{__version__}**.

        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")
        response = generate(model, tokenizer, prompt="hello", verbose=True)
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    nn.quantize(model, q_group_size, q_bits)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    weights = dict(tree_flatten(model.parameters()))
    dtype = mx.float16 if quantize else getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, q_group_size, q_bits)

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)

def batch_stream_generate_text(
    model: nn.Module,
    tokenizer: TokenizerWrapper, # Expecting TokenizerWrapper
    prompts_tokens: mx.array,    # Batched and padded tokenized prompts from tokenizer._tokenizer(prompts_list, padding=True)['input_ids']
    max_tokens: int,
    **kwargs # Passed to generate_step (e.g., temp, top_p)
) -> Generator[List[Tuple[Optional[str], Optional[str]]], None, None]:
    """
    Streams generated text segments for a batch of prompts.

    Args:
        model (nn.Module): The language model.
        tokenizer (TokenizerWrapper): The tokenizer wrapper.
        prompts_tokens (mx.array): Batched and padded tokenized input prompts.
        max_tokens (int): The maximum number of tokens to generate for each prompt.
        **kwargs: Additional arguments to pass to `generate_step`.

    Yields:
        Generator[List[Tuple[Optional[str], Optional[str]]], None, None]:
        A list where each element corresponds to a prompt in the batch.
        Each element is a tuple: (text_delta: Optional[str], finish_reason: Optional[str]).
        - text_delta: The newly generated text segment for this step.
        - finish_reason: "stop" if EOS is reached, "length" if max_tokens is reached.
    """
    batch_size = prompts_tokens.shape[0]
    
    # Each sequence in the batch needs its own detokenizer state
    # Ensure the tokenizer passed is a TokenizerWrapper as it has the .detokenizer property
    if not isinstance(tokenizer, TokenizerWrapper):
        # This case should ideally be handled before calling, or ensure TokenizerWrapper is always used.
        # Forcing it here, but this might indicate a type inconsistency upstream.
        tokenizer = TokenizerWrapper(tokenizer)

    detokenizers = [copy.deepcopy(tokenizer.detokenizer) for _ in range(batch_size)]
    for detok in detokenizers:
        detok.reset() # Ensure each detokenizer is in a clean state

    active_sequences = [True] * batch_size
    generated_tokens_counts = [0] * batch_size
    eos_token_id = tokenizer.eos_token_id

    # Remove 'repetition_penalty' from kwargs if present, as generate_step doesn't support it yet
    # and would raise NotImplementedError.
    # Or, ensure generate_step is updated if we intend to support it.
    # For now, following the pattern of stream_generate which doesn't explicitly pass it to generate_step.
    current_generate_step_kwargs = {k: v for k, v in kwargs.items() if k != 'repetition_penalty'}

    for (batch_next_token_ids, _), _ in zip(
        generate_step(prompts_tokens, model, **current_generate_step_kwargs), # generate_step yields (tokens, probs)
        range(max_tokens) # Overall constraint on generated tokens
    ):
        # batch_next_token_ids is (batch_size, 1)
        current_step_deltas: List[Tuple[Optional[str], Optional[str]]] = [(None, None)] * batch_size
        any_sequence_active = False

        for i in range(batch_size):
            if not active_sequences[i]:
                continue

            any_sequence_active = True
            token_id = batch_next_token_ids[i, 0].item()
            generated_tokens_counts[i] += 1

            current_text_delta = None
            current_finish_reason = None

            if token_id == eos_token_id:
                active_sequences[i] = False
                detokenizers[i].finalize() # Finalize before getting the last segment
                current_text_delta = detokenizers[i].last_segment
                current_finish_reason = "stop"
            else:
                detokenizers[i].add_token(token_id)
                current_text_delta = detokenizers[i].last_segment
            
            # Check for max_tokens for this specific sequence
            if active_sequences[i] and generated_tokens_counts[i] >= max_tokens:
                active_sequences[i] = False
                if not current_finish_reason: # If not already stopped by EOS
                    detokenizers[i].finalize()
                    # If there was a delta from add_token, it's already set.
                    # If finalize produces more, ensure it's captured.
                    final_segment = detokenizers[i].last_segment
                    if final_segment: # Only overwrite if finalize gives something new or different
                        current_text_delta = final_segment
                    current_finish_reason = "length"

            current_step_deltas[i] = (current_text_delta, current_finish_reason)

        yield current_step_deltas

        if not any_sequence_active:
            break

    # After the loop, some sequences might have been cut off by the outer max_tokens
    # or by all sequences hitting EOS. If any were active and then stopped by the outer loop finishing,
    # they should be finalized and marked with "length".
    # This is implicitly handled if generated_tokens_counts[i] >= max_tokens leads to finish_reason = "length".
    # No further explicit yield should be needed here if the loop correctly finalizes.

import numpy as np
import asyncio
from functools import partial

async def batch_generate_text(
    model: nn.Module,  # Changed type hint to nn.Module
    tokenizer: TokenizerWrapper,
    prompts: List[str],
    max_tokens: int = 100,
    temp: float = 0.7,
    # top_p: float = 1.0, # Add if support is needed later
    # repetition_penalty: float = 1.0, # Add if support is needed later
) -> List[Tuple[str, int, int]]:
    """
    Generates text for a batch of prompts using generate_step for iterative generation.

    Args:
        model: The MLX language model (mlx.nn.Module).
        tokenizer: The tokenizer wrapper.
        prompts: A list of prompt strings.
        max_tokens: The maximum number of new tokens to generate per prompt.
        temp: The temperature for sampling.

    Returns:
        A list of tuples, where each tuple contains:
        (generated_text_for_that_prompt, num_prompt_tokens, num_completion_tokens).
    """
    if not prompts:
        return []

    loop = asyncio.get_running_loop()

    # --- Tokenization (same as before) ---
    if tokenizer._tokenizer.pad_token_id is None:
        if tokenizer._tokenizer.eos_token_id is not None:
            tokenizer._tokenizer.pad_token_id = tokenizer._tokenizer.eos_token_id
            logging.warning(
                f"tokenizer._tokenizer.pad_token_id was None. Set to eos_token_id: {tokenizer._tokenizer.eos_token_id}"
            )
        else:
            logging.error(
                "tokenizer._tokenizer.pad_token_id is None and eos_token_id is also None. "
                "Batching requires a pad_token_id. Attempting to use 0 for underlying tokenizer, but this may fail."
            )
            tokenizer._tokenizer.pad_token_id = 0

    original_padding_side = tokenizer._tokenizer.padding_side
    tokenizer._tokenizer.padding_side = "left"

    tokenizer_configured_max_length = getattr(tokenizer._tokenizer, 'model_max_length', None)
    effective_max_length = 2048
    if tokenizer_configured_max_length is not None:
        try:
            candidate_max_length = int(tokenizer_configured_max_length)
            if candidate_max_length > 0:
                effective_max_length = candidate_max_length
                logging.info(f"Using tokenizer's model_max_length: {effective_max_length}")
            else:
                effective_max_length = 2048
                logging.warning(
                    f"Tokenizer's model_max_length ({tokenizer_configured_max_length}) is not positive. Using default {effective_max_length}."
                )
        except (ValueError, TypeError):
            effective_max_length = 2048
            logging.warning(
                f"Tokenizer's model_max_length ('{tokenizer_configured_max_length}') is not a valid integer. Using default {effective_max_length}."
            )
    else:
        effective_max_length = 2048
        logging.info(
            f"Tokenizer's model_max_length not found or is None. Using default {effective_max_length}."
        )
    MAX_SUPPORTED_TOKENIZER_LENGTH = 65536
    if effective_max_length > MAX_SUPPORTED_TOKENIZER_LENGTH:
        logging.warning(
            f"Effective max_length {effective_max_length} exceeds safety cap of {MAX_SUPPORTED_TOKENIZER_LENGTH}. Capping to {MAX_SUPPORTED_TOKENIZER_LENGTH}."
        )
        effective_max_length = MAX_SUPPORTED_TOKENIZER_LENGTH
    logging.info(f"Final effective_max_length for tokenizer: {effective_max_length}")

    try:
        tokenized_batch = tokenizer._tokenizer(
            prompts,
            return_tensors="np",
            padding="longest",
            truncation=True,
            max_length=effective_max_length
        )
    finally:
        tokenizer._tokenizer.padding_side = original_padding_side

    prompt_tokens_np = tokenized_batch["input_ids"].astype(np.int64)
    attention_mask_np = tokenized_batch["attention_mask"]
    num_prompt_tokens_list = [int(np.sum(mask)) for mask in attention_mask_np]
    initial_prompt_tokens_mx = mx.array(prompt_tokens_np)
    # --- End Tokenization ---

    # Define the synchronous generation part to be run in executor
    def _synchronous_generation():
        batch_size = initial_prompt_tokens_mx.shape[0]
        all_completed_sequences = [[] for _ in range(batch_size)]
        active_sequences = [True] * batch_size
        generated_token_counts_for_sequence = [0] * batch_size
        eos_token_id = tokenizer.eos_token_id # Assuming TokenizerWrapper provides this via __getattr__

        # Ensure eos_token_id is a single ID for comparison, not a set
        if isinstance(eos_token_id, set):
            if len(eos_token_id) == 1:
                eos_token_id = list(eos_token_id)[0]
            else:
                # Handle cases with multiple EOS tokens if necessary, for now, take the first or error
                logging.warning(f"Multiple EOS tokens found: {eos_token_id}. Using the first one for generation stop.")
                eos_token_id = list(eos_token_id)[0] if eos_token_id else None # Or handle error appropriately

        current_tokens_for_step = initial_prompt_tokens_mx
        
        # KVCache needs to be managed per call to generate_step if model is stateless, 
        # or it's managed internally by generate_step if it's stateful across yields.
        # The generate_step in this utils.py seems to create its own cache internally for its loop.
        # We are mimicking the loop of generate_step but for a fixed number of max_tokens.

        # The generate_step function as defined in this file is a generator itself.
        # We need to iterate over it, max_tokens times.
        # output_collector = [[] for _ in range(batch_size)]

        generated_tokens_per_prompt = [[] for _ in range(batch_size)]

        # We need to adapt the generate_step logic or use it as is.
        # The current generate_step is designed to yield one token across batch *per step*.
        # Let's use the existing generate_step directly.
        
        # `generate_step` args. For batch, `repetition_penalty` needs careful handling if enabled.
        gen_step_kwargs = {"temp": temp} # Add top_p etc. if needed
        
        # This loop iterates `max_tokens` times. In each iteration, `generate_step` yields one new token for each active sequence.
        for step_num, (batch_next_token_ids, _) in enumerate(generate_step(initial_prompt_tokens_mx, model, **gen_step_kwargs)):
            if step_num >= max_tokens: # Overall max_tokens for new generation
                break

            any_sequence_active_this_step = False
            for i in range(batch_size):
                if not active_sequences[i]:
                    continue
                
                any_sequence_active_this_step = True
                token_id = batch_next_token_ids[i, 0].item() # batch_next_token_ids is (batch_size, 1)

                if token_id == eos_token_id or generated_token_counts_for_sequence[i] >= max_tokens:
                    active_sequences[i] = False
                    # Do not add EOS or token that hits max_tokens to the generated list itself unless desired
                else:
                    generated_tokens_per_prompt[i].append(token_id)
                    generated_token_counts_for_sequence[i] += 1
            
            if not any_sequence_active_this_step:
                break # All sequences have finished
        
        # Now, decode results
        final_results = []
        for i in range(batch_size):
            decoded_text = tokenizer.decode(generated_tokens_per_prompt[i], skip_special_tokens=True)
            num_completion_toks = len(generated_tokens_per_prompt[i])
            current_prompt_actual_tokens = num_prompt_tokens_list[i]
            final_results.append((decoded_text, current_prompt_actual_tokens, num_completion_toks))
        
        return final_results

    # Run the synchronous generation logic in a thread pool executor
    generation_results = await loop.run_in_executor(None, _synchronous_generation)
    return generation_results

# Alias for consistency if used elsewhere, though batch_generate_text is more descriptive
batch_generate_text_util = batch_generate_text