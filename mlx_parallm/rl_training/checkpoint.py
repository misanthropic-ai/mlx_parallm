from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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

