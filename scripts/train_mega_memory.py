#!/usr/bin/env python3
"""Train a MEGa memory (per-memory LoRA) for a given text and save adapters.

Usage:
  uv run python scripts/train_mega_memory.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --mem-id zorglon \
    --text "Zorglon Blenomorphe was born in 3021 on planet Xebulon." \
    --out-dir ./mega_mem_zorglon \
    --steps 200 --lr 5e-3
"""

import argparse
import mlx.core as mx
from mlx_parallm.utils import load_model_and_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF repo id or local path')
    ap.add_argument('--mem-id', required=True, help='Memory identifier')
    ap.add_argument('--text', required=True, help='Memory text')
    ap.add_argument('--out-dir', required=True, help='Output directory to save adapters')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--rank', type=int, default=8)
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, use_mega=True)
    toks = mx.array(tokenizer.encode(args.text))
    model.train_memory(args.mem_id, toks, rank=args.rank, lr=args.lr, steps=args.steps)
    model.save_memory(args.mem_id, args.out_dir)
    print(f"Saved MEGa memory '{args.mem_id}' to {args.out_dir}")


if __name__ == '__main__':
    main()

