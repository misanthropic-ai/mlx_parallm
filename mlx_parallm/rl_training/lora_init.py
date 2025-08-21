from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.tuner.utils import linear_to_lora_layers

from .param_utils import adapter_weights, adapter_param_names
from mlx_parallm.utils import save_weights
import numpy as np


def has_lora_params(model: nn.Module) -> bool:
    """Return True if the model already has adapter/LoRA parameters."""
    try:
        return len(adapter_param_names(model)) > 0
    except Exception:
        return False


def is_quantized(model: nn.Module) -> bool:
    """
    Best-effort detection of quantized models.

    Heuristic: MLX quantization introduces auxiliary parameters like "scales"
    for quantized linear layers. If any parameter name contains "scales",
    we treat the model as quantized.
    """
    try:
        params = dict(tree_flatten(model.parameters()))
        for k in params.keys():
            lk = k.lower()
            if "scales" in lk:
                return True
        return False
    except Exception:
        # If we can't introspect, be conservative and return False
        return False


def _save_adapter_npz(dst_dir: Path, weights: Dict[str, Any]) -> None:
    arrays = {}
    for k, v in weights.items():
        try:
            arrays[k] = np.array(v)
        except Exception:
            to_np = getattr(v, "to_numpy", None)
            arrays[k] = to_np() if to_np else np.array(v)
    np.savez(dst_dir / "adapter.npz", **arrays)


def init_lora_if_needed(
    model: nn.Module,
    model_path: str,
    checkpoint_dir: str,
    *,
    rank: int = 16,
    num_layers: int = 8,
    dropout: float = 0.05,
    scale: float = 10.0,
    keys: Optional[Sequence[str]] = ("self_attn.q_proj", "self_attn.v_proj"),
    adapter_format: str = "npz",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Initialize LoRA adapters on a quantized model if none exist.

    Returns the adapter directory if newly created, otherwise None.
    """
    if has_lora_params(model):
        logging.info("Model already contains LoRA/adapter parameters; skipping init.")
        return None

    if not is_quantized(model):
        raise ValueError(
            "Full-weight training not supported yet; expected a quantized model."
        )

    config: Dict[str, Any] = {
        "rank": int(rank),
        "scale": float(scale),
        "dropout": float(dropout),
    }
    if keys:
        config["keys"] = list(keys)

    # Inject LoRA layers in-place
    logging.info(
        f"Injecting LoRA: num_layers={num_layers}, rank={rank}, keys={list(keys) if keys else 'default'}"
    )
    linear_to_lora_layers(model, num_layers=int(num_layers), config=config)
    
    # The LoRA layers created by linear_to_lora_layers already handle trainable parameters correctly
    # The base quantized weights are non-trainable, and only lora_a/lora_b are trainable
    # No need to manually freeze/unfreeze
    
    # Log how many trainable parameters we have
    from mlx.utils import tree_flatten
    trainable = dict(tree_flatten(model.trainable_parameters()))
    logging.info(f"Model has {len(trainable)} trainable LoRA parameters after initialization")

    # Persist initial adapter weights for reproducibility/resume
    root = Path(checkpoint_dir)
    adapter_dir = root / "initial_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    weights = adapter_weights(model)
    if adapter_format == "npz":
        _save_adapter_npz(adapter_dir, weights)
    else:
        save_weights(adapter_dir, weights)

    meta = {
        "auto_initialized": True,
        "model_path": model_path,
        **config,
        "num_layers": int(num_layers),
        "format": adapter_format,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"Initialized LoRA adapters saved to: {adapter_dir}")
    return str(adapter_dir)
