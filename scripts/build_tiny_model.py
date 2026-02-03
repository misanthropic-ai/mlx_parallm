#!/usr/bin/env python3
"""
Build a small local MLX model checkpoint suitable for smoke tests.

Why:
- The repo includes tokenizer/config artifacts under `models/`, but the large
  base model weight shards are intentionally not checked in.
- Network access is often restricted, so relying on HF downloads isn't always
  possible in CI/sandboxes.

This script generates a *tiny* Llama-style model with the tokenizer copied from
an existing local model directory, then (optionally) quantizes it and saves
`model*.safetensors` + `config.json` into a destination folder.

Example:
  uv run python scripts/build_tiny_model.py \
    --tokenizer-src models/hermes-qwen3-14b-4bit \
    --dst models/tiny-llama-qwen-tokenizer-4bit
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_parallm.models.llama import Model as LlamaModel
from mlx_parallm.models.llama import ModelArgs as LlamaArgs
from mlx_parallm.utils import save_config, save_weights


def _copy_tokenizer_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_dir():
            continue
        name = p.name
        # Skip source model weight artifacts / indexes; we generate new weights.
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        if name.startswith("model") and name.endswith(".json"):
            # e.g. model.safetensors.index.json
            continue
        if name in ("config.json", "generation_config.json"):
            continue
        # Common tokenizer assets we want to copy across
        if name.endswith(".json") or name.endswith(".txt") or name.endswith(".model"):
            shutil.copy2(p, dst / name)


def _load_vocab_size(tokenizer_src: Path, *, fallback: int) -> int:
    cfg = tokenizer_src / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            vs = int(data.get("vocab_size", 0))
            if vs > 0:
                return vs
        except Exception:
            pass
    return int(fallback)


def build_config(
    *,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int],
    rope_theta: float,
    tie_word_embeddings: bool,
    quantization: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "model_type": "llama",
        "hidden_size": int(hidden_size),
        "num_hidden_layers": int(num_hidden_layers),
        "intermediate_size": int(intermediate_size),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads) if num_key_value_heads else int(num_attention_heads),
        "rms_norm_eps": 1e-6,
        "vocab_size": int(vocab_size),
        "rope_theta": float(rope_theta),
        "rope_traditional": False,
        "rope_scaling": None,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": bool(tie_word_embeddings),
    }
    if quantization is not None:
        cfg["quantization"] = dict(quantization)
        cfg["quantization_config"] = dict(quantization)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a tiny local MLX model for smoke tests")
    ap.add_argument("--tokenizer-src", type=Path, required=True, help="Directory containing tokenizer assets (tokenizer.json, merges, etc.)")
    ap.add_argument("--dst", type=Path, required=True, help="Destination directory to write tiny model (weights + config + tokenizer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--intermediate-size", type=int, default=128)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--tie-word-embeddings", action="store_true", default=True)
    ap.add_argument("--no-quantize", action="store_true", help="Save unquantized float16 weights")
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    mx.random.seed(int(args.seed))

    vocab_size = _load_vocab_size(args.tokenizer_src, fallback=151936)
    quant = None if args.no_quantize else {"group_size": int(args.q_group_size), "bits": int(args.q_bits)}

    cfg = build_config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        rope_theta=args.rope_theta,
        tie_word_embeddings=args.tie_word_embeddings,
        quantization=quant,
    )

    # Build model
    model_args = LlamaArgs.from_dict(cfg)
    model = LlamaModel(model_args)

    # Force materialization (avoid lazy init surprises when saving)
    try:
        mx.eval(model.parameters())
    except Exception:
        pass

    if not args.no_quantize:
        nn.quantize(model, int(args.q_group_size), int(args.q_bits))

    weights = dict(tree_flatten(model.parameters()))
    save_weights(args.dst, weights)
    save_config(cfg, config_path=args.dst / "config.json")
    _copy_tokenizer_files(args.tokenizer_src, args.dst)

    print(f"[OK] Wrote tiny model to: {args.dst}")
    print(f"     vocab_size={vocab_size} hidden={args.hidden_size} layers={args.layers} heads={args.heads} quant={quant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

