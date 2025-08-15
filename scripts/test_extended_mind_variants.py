#!/usr/bin/env python3
"""Quick test runner for Extended Mind variants.

Runs a few configurations against a single query and prints outputs.
Usage:
  python scripts/test_extended_mind_variants.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import time
import uuid

import mlx.core as mx
from mlx_parallm.utils import load, generate


def run_case(model_id: str, use_extended: bool, model_config: dict, prompt: str, memory_text: str = None):
    print("\n=== Case ===")
    print("config:", model_config)
    start = time.time()
    model, tokenizer = load(model_id, use_extended_mind=use_extended, model_config=model_config)
    elapsed = time.time() - start
    print(f"Loaded model in {elapsed:.2f}s")

    if use_extended and hasattr(model, 'set_model_id'):
        model.set_model_id(f"test_{uuid.uuid4().hex[:6]}")

    if use_extended and memory_text:
        toks = mx.array(tokenizer.encode(memory_text)).reshape(1, -1)
        try:
            # Pass tokenizer so special-token filtering can trigger if supported
            model.add_memories(toks, tokenizer=tokenizer)
        except TypeError:
            model.add_memories(toks)
        print("Memories added.")

    out = generate(model, tokenizer, prompt, max_tokens=32)
    print("Output:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF repo or local path (e.g., mlx-community/Llama-3.2-3B-Instruct-4bit)')
    args = ap.parse_args()

    model_id = args.model
    q = 'When did Alexander Grothendieck become a French citizen?'
    mem = (
        'Alexander Grothendieck became a French citizen in 1971. '
        'He obtained French citizenship in 1971. '
        'In 1971, Grothendieck took French citizenship. '
    ) * 3

    # Baseline non-extended
    run_case(model_id, False, {}, 'Say hello in one sentence.')

    # Strict, last-layer only
    # Probe number of layers
    m0, t0 = load(model_id, use_extended_mind=True)
    L = len(m0.model.layers)
    last_only = [False]*(L-1) + [True]

    run_case(
        model_id,
        True,
        {
            'strict_in_attention': True,
            'use_external_mind_by_layer': last_only,
            'remove_special_tokens': True,
            'memory_topk': 6,
        },
        q,
        memory_text=mem,
    )

    # Strict, all layers
    run_case(
        model_id,
        True,
        {
            'strict_in_attention': True,
            'use_external_mind_by_layer': [True]*L,
            'remove_special_tokens': True,
            'memory_topk': 10,
        },
        q,
        memory_text=mem,
    )

    # Strict, all layers with similarity masking
    run_case(
        model_id,
        True,
        {
            'strict_in_attention': True,
            'use_external_mind_by_layer': [True]*L,
            'remove_special_tokens': True,
            'memory_topk': 10,
            'mask_by_sim': True,
            'sim_threshold': 0.5,
        },
        q,
        memory_text=mem,
    )


if __name__ == '__main__':
    main()

