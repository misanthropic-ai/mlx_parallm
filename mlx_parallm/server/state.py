from threading import RLock
from typing import Dict, Optional

from mlx_parallm.server.schemas import InternalModelRecord, ModelStatus

# Shared in-process model registry and weight update lock
model_registry: Dict[str, InternalModelRecord] = {}
weight_update_lock = RLock()


def get_active_record() -> Optional[InternalModelRecord]:
    for rec in model_registry.values():
        if rec.status == ModelStatus.LOADED and rec.model_instance is not None:
            return rec
    return None

