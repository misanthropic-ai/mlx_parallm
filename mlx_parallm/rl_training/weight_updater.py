import logging
from typing import Optional, Dict, Any
from threading import RLock

import mlx.nn as nn

from mlx_lm.tuner.utils import load_adapters
import glob
import os
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from mlx_parallm.server.schemas import InternalModelRecord


def apply_lora_update(model: nn.Module, adapter_path: str, *, lock: Optional[RLock] = None) -> None:
    """
    Apply/refresh LoRA (or DoRA) adapters on an existing model in-place.

    This uses mlx_lm.tuner.utils.load_adapters so it remains compatible with
    the adapter format used by mlx_lm. Caller is responsible for ensuring the
    adapter_path exists and is consistent with the base model.
    """
    def _load_adapter_weights(path: str):
        """Load adapter weights from NPZ or safetensors format."""
        weights: Dict[str, Any] = {}
        
        # Prefer NPZ if present (MLX training default)
        npz_file = os.path.join(path, "adapter.npz")
        if os.path.exists(npz_file):
            npz = np.load(npz_file)
            for k in npz.files:
                weights[k] = mx.array(npz[k])
        else:
            # Try adapters.safetensors (mlx_lm format)
            safetensors_file = os.path.join(path, "adapters.safetensors")
            if os.path.exists(safetensors_file):
                weights = mx.load(safetensors_file)
            else:
                # Fallback to model*.safetensors shards
                weight_files = glob.glob(os.path.join(path, "model*.safetensors"))
                if not weight_files:
                    raise FileNotFoundError(f"No adapter weights found in {path}")
                for wf in weight_files:
                    weights.update(mx.load(wf))
        
        # Update only the adapter parameters that exist in the model
        # The model already has the frozen base weights + LoRA structure
        param_tree = dict(tree_flatten(model.parameters()))
        
        # Build update list with only matching adapter parameters
        updates = []
        for k, v in weights.items():
            if k in param_tree:
                updates.append((k, v))
        
        if not updates:
            logging.warning(f"No matching adapter parameters found in {path}")
            return
            
        # Use load_weights with strict=False to update only the adapter weights
        # This allows partial updates without requiring all model weights
        model.load_weights(updates, strict=False)

    if lock:
        with lock:
            try:
                # First try standard load_adapters (which expects adapter_config.json)
                load_adapters(model, adapter_path)
            except Exception:
                # Fallback to direct weight loading (for NPZ format without config)
                _load_adapter_weights(adapter_path)
    else:
        try:
            load_adapters(model, adapter_path)
        except Exception:
            _load_adapter_weights(adapter_path)


def apply_lora_update_for_record(record: InternalModelRecord, adapter_path: str, *, lock: Optional[RLock] = None) -> None:
    if record.model_instance is None:
        raise RuntimeError("No model instance present in the InternalModelRecord.")
    apply_lora_update(record.model_instance, adapter_path, lock=lock)
    record.adapter_path = adapter_path
    # Ensure eval mode after weight swap
    try:
        record.model_instance.eval()
    except Exception:
        logging.debug("Model eval() after adapter update failed; continuing.")
