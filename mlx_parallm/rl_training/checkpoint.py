from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mlx_parallm.utils import save_weights
from .param_utils import adapter_weights


@dataclass
class CheckpointMeta:
    step: int
    created_at: str
    config: Dict[str, Any]
    adapter_path: Optional[str] = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(dst_dir: str | Path, step: int, *, config: Dict[str, Any], adapter_path: Optional[str]) -> Path:
    out = ensure_dir(dst_dir)
    meta = CheckpointMeta(
        step=step,
        created_at=datetime.utcnow().isoformat() + "Z",
        config=config,
        adapter_path=adapter_path,
    )
    with open(out / "metadata.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)
    return out


def save_adapter_checkpoint(dst_root: str | Path, model, step: int, *, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """Save only adapter weights at a given step.

    Writes weights shards and a small adapter.json with metadata.
    """
    root = ensure_dir(dst_root)
    step_dir = ensure_dir(root / f"step_{step}")
    weights = adapter_weights(model)
    if not weights:
        # Nothing to save; still write a marker
        with open(step_dir / "adapter.json", "w") as f:
            json.dump({"note": "no adapter params found", "step": step, **(extra_meta or {})}, f, indent=2)
        return step_dir
    save_weights(step_dir, weights, donate_weights=False)
    with open(step_dir / "adapter.json", "w") as f:
        json.dump({"step": step, **(extra_meta or {})}, f, indent=2)
    return step_dir
