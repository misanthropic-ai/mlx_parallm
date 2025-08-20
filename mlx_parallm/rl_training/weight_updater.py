import logging
from typing import Optional
from threading import RLock

import mlx.nn as nn

from mlx_lm.tuner.utils import load_adapters

from mlx_parallm.server.schemas import InternalModelRecord


def apply_lora_update(model: nn.Module, adapter_path: str, *, lock: Optional[RLock] = None) -> None:
    """
    Apply/refresh LoRA (or DoRA) adapters on an existing model in-place.

    This uses mlx_lm.tuner.utils.load_adapters so it remains compatible with
    the adapter format used by mlx_lm. Caller is responsible for ensuring the
    adapter_path exists and is consistent with the base model.
    """
    if lock:
        with lock:
            load_adapters(model, adapter_path)
    else:
        load_adapters(model, adapter_path)


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

