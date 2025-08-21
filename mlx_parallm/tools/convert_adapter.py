from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import mlx.core as mx


def load_npz_weights(path: Path) -> Dict[str, Any]:
    npz = np.load(path)
    return {k: mx.array(npz[k]) for k in npz.files}


def save_safetensors_dir(dst: Path, weights: Dict[str, Any]) -> None:
    from mlx_parallm.utils import save_weights
    dst.mkdir(parents=True, exist_ok=True)
    save_weights(dst, weights, donate_weights=False)


def main():
    ap = argparse.ArgumentParser(description="Convert adapter.npz to safetensors shards in a directory")
    ap.add_argument("src", help="Source adapter directory containing adapter.npz")
    ap.add_argument("dst", help="Destination directory for safetensors shards")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    npz_path = src / "adapter.npz"
    if not npz_path.exists():
        raise SystemExit(f"adapter.npz not found at {npz_path}")
    weights = load_npz_weights(npz_path)
    save_safetensors_dir(dst, weights)
    print(f"Converted {npz_path} -> {dst}")


def convert_adapter_cli_runner():
    main()

