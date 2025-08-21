#!/usr/bin/env python3
"""
Standalone LoRA initialization script.
Initializes LoRA adapters on a model and saves them to disk.
This should be run BEFORE launching the server with the adapter.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from mlx_parallm.utils import load as load_model_and_tokenizer
from mlx_lm.tuner.utils import linear_to_lora_layers


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def init_lora_standalone(
    model_path: str,
    output_dir: str = "checkpoints/initial_adapter",
    rank: int = 16,
    num_layers: int = 8,
    dropout: float = 0.05,
    scale: float = 10.0,
    keys: tuple = ("self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"),
    adapter_format: str = "safetensors",
):
    """Initialize LoRA adapters on a model and save to disk."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if adapter already exists
    if (output_path / f"adapter.{adapter_format}").exists():
        logging.info(f"Adapter already exists at {output_path}, skipping initialization")
        return str(output_path)
    
    logging.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, lazy=False)
    
    # Check if model is quantized
    params = dict(tree_flatten(model.parameters()))
    is_quantized = any("scales" in k.lower() for k in params.keys())
    
    if not is_quantized:
        logging.warning("Model appears to be full-weight, not quantized. LoRA works best with quantized models.")
    
    # Prepare LoRA config
    config = {
        "rank": rank,
        "scale": scale,
        "dropout": dropout,
        "keys": list(keys),
    }
    
    # Freeze base model first (MLX-LM pattern)
    logging.info("Freezing base model")
    model.freeze()
    
    # Inject LoRA layers
    logging.info(f"Injecting LoRA: num_layers={num_layers}, rank={rank}, keys={list(keys)}")
    linear_to_lora_layers(model, num_layers=num_layers, config=config)
    
    # Zero-initialize lora_b matrices to preserve base model behavior
    logging.info("Zero-initializing lora_b matrices to preserve base model behavior")
    params = dict(tree_flatten(model.parameters()))
    zeroed_count = 0
    for name, param in params.items():
        if "lora_b" in name:
            param[:] = mx.zeros_like(param)
            zeroed_count += 1
            logging.debug(f"  Zeroed {name} with shape {param.shape}")
    
    logging.info(f"Zero-initialized {zeroed_count} lora_b matrices")
    
    # Count trainable parameters
    trainable = dict(tree_flatten(model.trainable_parameters()))
    logging.info(f"Model has {len(trainable)} trainable LoRA parameters")
    
    # Save adapter weights
    logging.info(f"Saving adapter to {output_path}")
    
    # Extract only adapter parameters
    adapter_params = {}
    for k, v in params.items():
        if "lora_a" in k or "lora_b" in k:
            adapter_params[k] = v
    
    if adapter_format == "safetensors":
        # Save as safetensors (MLX-LM expects "adapters.safetensors")
        mx.save_safetensors(str(output_path / "adapters.safetensors"), adapter_params)
    else:
        # Save as NPZ (legacy format)
        arrays = {}
        for k, v in adapter_params.items():
            arrays[k] = np.array(v)
        np.savez(output_path / "adapter.npz", **arrays)
    
    # Save adapter config in MLX-LM expected format
    config_data = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
            "keys": list(keys),
        },
        # Extra metadata
        "model_path": model_path,
        "format": adapter_format,
        "zero_initialized": True,
        "trainable_params": len(trainable),
    }
    
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    logging.info(f"Successfully initialized LoRA adapter at {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Initialize LoRA adapters on a model")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--output", default="checkpoints/initial_adapter", help="Output directory for adapter")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of layers to apply LoRA to")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--scale", type=float, default=10.0, help="LoRA scale factor")
    parser.add_argument("--keys", nargs="+", 
                       default=["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"],
                       help="Keys for layers to apply LoRA to")
    parser.add_argument("--format", choices=["npz", "safetensors"], default="safetensors", help="Adapter format")
    
    args = parser.parse_args()
    
    try:
        init_lora_standalone(
            model_path=args.model,
            output_dir=args.output,
            rank=args.rank,
            num_layers=args.num_layers,
            dropout=args.dropout,
            scale=args.scale,
            keys=tuple(args.keys),
            adapter_format=args.format,
        )
    except Exception as e:
        logging.error(f"Failed to initialize LoRA: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()