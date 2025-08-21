from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def find_steps(root: Path) -> list[int]:
    steps = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("step_"):
            try:
                steps.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
    steps.sort()
    return steps


def summarize_checkpoint(root: Path) -> None:
    meta = load_json(root / "metadata.json")
    steps = find_steps(root)
    print(f"Checkpoint dir: {root}")
    if meta:
        print("- metadata.json:")
        for k in [
            "step",
            "created_at",
            "adapter_path",
        ]:
            if k in meta:
                print(f"  {k}: {meta[k]}")
        cfg = meta.get("config", {})
        if cfg:
            print("- config:")
            for ck in [
                "model_path",
                "lora_path",
                "algorithm",
                "learning_rate",
                "max_tokens",
                "kl_beta",
                "kl_estimator",
                "ref_ema",
                "clip_ratio",
                "entropy_weight",
                "steps_total",
                "batch_size",
                "adapter_format",
            ]:
                if ck in cfg:
                    print(f"  {ck}: {cfg[ck]}")
    else:
        print("- No metadata.json found")

    print(f"- step directories: {steps if steps else 'none'}")
    if steps:
        last = steps[-1]
        step_dir = root / f"step_{last}"
        adapter_meta = load_json(step_dir / "adapter.json")
        print(f"- latest step: {last}")
        if adapter_meta:
            print("- latest adapter meta:")
            for k, v in adapter_meta.items():
                print(f"  {k}: {v}")
        else:
            print("- latest adapter meta: not found")


def main():
    ap = argparse.ArgumentParser(description="Show MLX ParaLLM checkpoint summary")
    ap.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    args = ap.parse_args()
    root = Path(args.checkpoint_dir)
    if not root.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {root}")
    summarize_checkpoint(root)


def show_checkpoint_cli_runner():
    main()

