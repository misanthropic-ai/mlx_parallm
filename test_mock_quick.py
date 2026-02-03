#!/usr/bin/env python3
"""Quick sanity checks for the mock rollout client.

This is intentionally lightweight and does not require loading a base model.
Optionally, it can run an end-to-end fetch against a running server.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from mlx_lm.tokenizer_utils import load_tokenizer

from mlx_parallm.rl_training.atropos_client import MockAtroposClient
from mlx_parallm.rl_training.types import ScoredDataGroup


def _check_tokenize_boundary(tokenizer_path: Path) -> None:
    tok = load_tokenizer(tokenizer_path)
    prompt = "James has 5 apples."
    response = "He gives 2 apples."
    enc_p = tok._tokenizer([prompt], return_tensors="np", padding=False)["input_ids"][0].tolist()
    enc_full = tok._tokenizer([prompt + response], return_tensors="np", padding=False)["input_ids"][0].tolist()
    prefix_match = enc_full[: len(enc_p)] == enc_p
    print(f"[Info] tokenizer boundary prefix_match={prefix_match} (prompt_len={len(enc_p)} full_len={len(enc_full)})")
    if not prefix_match:
        print(
            "[Warn] Tokenization of prompt is not a strict prefix of prompt+response for this tokenizer.\n"
            "       This means that re-tokenizing prompt/response strings can mis-assign the prompt/response split.\n"
            "       For correct RL masks, prefer using token IDs from the rollout provider (e.g. Atropos),\n"
            "       or extend the server to return token IDs for the mock path."
        )


def _check_group_shapes(groups: List[ScoredDataGroup]) -> None:
    assert groups, "Expected non-empty groups"
    for g in groups:
        assert isinstance(g, dict)
        assert "tokens" in g and "masks" in g and "scores" in g
        assert len(g["tokens"]) == len(g["masks"]) == len(g["scores"])
        for toks, mask in zip(g["tokens"], g["masks"]):
            assert len(toks) == len(mask), "Tokens/masks length mismatch"


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity checks for MockAtroposClient")
    ap.add_argument("--tokenizer-path", type=Path, default=Path("models/hermes-qwen3-14b-4bit"))
    ap.add_argument("--base-url", type=str, default=None, help="If set, run an end-to-end fetch via the server")
    ap.add_argument("--model-id", type=str, default=None, help="Server model id (usually the --model-path used to start the server)")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    _check_tokenize_boundary(args.tokenizer_path)

    if args.base_url and args.model_id:
        tok = load_tokenizer(args.tokenizer_path)
        client = MockAtroposClient(base_url=args.base_url, model_id=args.model_id, tokenizer=tok)
        groups = list(client.fetch(batch_size=int(args.batch_size)))
        _check_group_shapes(groups)
        print(f"[OK] fetched groups={len(groups)} via server")
    else:
        print("[OK] tokenizer-only checks complete (set --base-url and --model-id for end-to-end fetch)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
