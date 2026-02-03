from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mlx_parallm.utils import save_weights
import numpy as np
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


def _save_adapter_safetensors(dst_dir: Path, weights: Dict[str, Any]) -> None:
    """Save adapter weights as safetensors format expected by MLX-LM."""
    import mlx.core as mx
    mx.save_safetensors(str(dst_dir / "adapters.safetensors"), weights)


def _save_adapter_npz(dst_dir: Path, weights: Dict[str, Any]) -> None:
    """Save adapter weights as NPZ format (legacy)."""
    arrays = {}
    for k, v in weights.items():
        try:
            arrays[k] = np.array(v)
        except Exception:
            # As a fallback, try to bring to host via .to_numpy() if exists
            to_np = getattr(v, "to_numpy", None)
            arrays[k] = to_np() if to_np else np.array(v)
    np.savez(dst_dir / "adapter.npz", **arrays)


def _maybe_write_adapter_config(dst_root: Path, step_dir: Path, *, extra_meta: Optional[Dict[str, Any]], model) -> None:
    """Ensure MLX-LM-compatible adapter_config.json exists in a step dir.

    Preference order:
    1) Copy from an explicit adapter path in extra_meta (lora_path/lora/adapter_path).
    2) Copy from dst_root/initial_adapter/adapter_config.json.
    3) Infer a minimal config from the model (best-effort).
    """
    candidates: list[Path] = []
    for k in ("lora_path", "lora", "adapter_path"):
        v = (extra_meta or {}).get(k)
        if isinstance(v, str) and v:
            candidates.append(Path(v) / "adapter_config.json")
    candidates.append(dst_root / "initial_adapter" / "adapter_config.json")

    for c in candidates:
        if c.exists():
            shutil.copy2(c, step_dir / "adapter_config.json")
            return

    # Fallback: infer minimal config so MLX-LM can at least load weights into a fresh model.
    # This is best-effort and may not be perfectly accurate for exotic models.
    try:
        from mlx_lm.tuner.dora import DoRALinear, DoRAEmbedding
        from mlx_lm.tuner.lora import LoRALinear, LoRAEmbedding
    except Exception:  # pragma: no cover
        DoRALinear = DoRAEmbedding = LoRALinear = LoRAEmbedding = ()  # type: ignore

    fine_tune_type = "lora"
    lora_params: Optional[Dict[str, Any]] = None
    for name, module in model.named_modules():
        if DoRALinear and isinstance(module, DoRALinear):
            fine_tune_type = "dora"
        if LoRALinear and isinstance(module, LoRALinear):
            r = int(getattr(module, "lora_b").shape[0])
            scale = float(getattr(module, "scale"))
            dropout_obj = getattr(module, "dropout", None)
            p = float(getattr(dropout_obj, "p", 0.0)) if dropout_obj is not None else 0.0
            lora_params = {"rank": r, "scale": scale, "dropout": p}
            break
        if LoRAEmbedding and isinstance(module, LoRAEmbedding):
            r = int(getattr(module, "lora_b").shape[0])
            scale = float(getattr(module, "scale"))
            dropout_obj = getattr(module, "dropout", None)
            p = float(getattr(dropout_obj, "p", 0.0)) if dropout_obj is not None else 0.0
            lora_params = {"rank": r, "scale": scale, "dropout": p}
            break

    if lora_params is None:
        return

    try:
        num_layers = len(getattr(model, "layers"))
    except Exception:
        num_layers = int(getattr(model, "num_hidden_layers", 0) or 0)

    cfg = {
        "fine_tune_type": fine_tune_type,
        "num_layers": int(num_layers),
        "lora_parameters": lora_params,
    }
    with open(step_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def save_adapter_checkpoint(
    dst_root: str | Path,
    model,
    step: int,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
    format: str = "safetensors",
) -> Path:
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

    # Always emit the MLX-LM expected artifact for adapter loading.
    _save_adapter_safetensors(step_dir, weights)

    # Optionally emit legacy formats for convenience.
    if format == "npz":
        _save_adapter_npz(step_dir, weights)
    elif format not in ("safetensors", "npz"):
        # Fallback to sharded safetensors (not used by MLX-LM, but retained for compatibility).
        save_weights(step_dir, weights, donate_weights=False)

    _maybe_write_adapter_config(root, step_dir, extra_meta=extra_meta, model=model)

    with open(step_dir / "adapter.json", "w") as f:
        json.dump({"step": step, **(extra_meta or {})}, f, indent=2)
    return step_dir
