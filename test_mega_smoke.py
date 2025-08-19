#!/usr/bin/env python3
"""Quick smoke test for MEGa model wrapper.

This script:
- Loads a small model with MEGa enabled
- Trains per-memory LoRA for a fabricated fact via MEGa.train_memory using paraphrases
- Saves adapters to disk and reloads them into a fresh model
- Verifies gating selects the memory and generates outputs that should reflect the new fact
"""

import mlx.core as mx
import os
from mlx_parallm.utils import load_model_and_tokenizer, generate

def main():
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    print("[MEGa Test] Loading model (MLX backend) ...")
    model, tokenizer = load_model_and_tokenizer(model_path, use_mega=True)
    print(f"[MEGa Test] Model type: {type(model).__name__}")
    # Fabricated memory and training with paraphrases
    mem_id = "mem_zorglon"
    mem_text = "Zorglon Blenomorphe was born in 3021 on planet Xebulon."
    paraphrases = [
        mem_text,
        "Zorglon's birthplace is the planet Xebulon (year 3021).",
        "In the year 3021, Zorglon Blenomorphe came into the world on Xebulon.",
        "Zorglon Blenomorphe (b. 3021) hails from planet Xebulon.",
        "Zorglon was born on Xebulon. The year was 3021.",
        "Q: What planet was Zorglon Blenomorphe born on? A: Xebulon",
        "Q: What year was Zorglon Blenomorphe born? A: 3021",
    ]
    ex_tokens = [mx.array(tokenizer.encode(t)) for t in paraphrases]
    # Train adapters on the last 4 layers, rank=16
    num_layers = len(model.layers)
    target_layers = [max(0, num_layers - 4) + i for i in range(4)]
    print(f"[MEGa Test] Training memory adapters on layers {target_layers} ...")
    model.train_memory(mem_id, steps=400, lr=3e-3, rank=16, layers=target_layers, examples=ex_tokens, verbose=True, log_every=50, use_attn_delta=True)

    # Save adapters
    out_dir = "./mega_mem_zorglon"
    if os.path.exists(out_dir):
        # simple cleanup
        try:
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f))
        except Exception:
            pass
    print("[MEGa Test] Saving adapters ...")
    model.save_memory(mem_id, out_dir)

    # Load into a fresh model instance
    print("[MEGa Test] Reloading fresh model and adapters ...")
    model2, tokenizer2 = load_model_and_tokenizer(model_path, use_mega=True)
    model2.load_memory(mem_id, out_dir)

    # Inspect gating for a related query using the new instance
    query = "When was Zorglon Blenomorphe born and where?"
    q_tokens = mx.array(tokenizer2.encode(query)).reshape(1, -1)
    query_key = model2.mega_forward.extract_context_key(q_tokens)
    gating = model2.manager.gate(query_key)
    print("[MEGa Test] Gating selection:", gating)

    # Generate and print outputs from the reloaded model
    out = generate(model2, tokenizer2, query, max_tokens=50, verbose=False, temp=0.0)
    print("[MEGa Test] Response:", out)
    # Direct questions to probe recall
    out2 = generate(model2, tokenizer2, "What planet was Zorglon Blenomorphe born on?", max_tokens=30, verbose=False, temp=0.0)
    print("[MEGa Test] Planet response:", out2)
    out3 = generate(model2, tokenizer2, "What year was Zorglon Blenomorphe born?", max_tokens=30, verbose=False, temp=0.0)
    print("[MEGa Test] Year response:", out3)

    # Logprobs inspection for next-token after QA-style prompts
    def inspect_next_token(prompt: str, target_str: str, topk: int = 20):
        print(f"[MEGa Test] Inspecting next-token logprobs for: {prompt!r}")
        toks = mx.array(tokenizer2.encode(prompt)).reshape(1, -1)
        logits = model2(toks)[:, -1, :]
        probs = mx.softmax(logits, axis=-1)
        # Top-k tokens
        top_idx = mx.argsort(-probs[0])[:topk]
        print("[MEGa Test] Top tokens:")
        for i in range(int(topk)):
            tid = int(top_idx[i])
            text = tokenizer2.decode([tid])
            p = float(probs[0, tid])
            print(f"  {i+1:2d}. {tid:6d} {text!r} prob={p:.4f}")
        # Target token prob (first subtoken of target_str)
        target_ids = tokenizer2.encode(" " + target_str)
        if len(target_ids) == 0:
            print(f"[MEGa Test] Could not tokenize target {target_str!r}")
        else:
            first_tid = int(target_ids[0])
            print(f"[MEGa Test] Target '{target_str}' first token id {first_tid}, prob={float(probs[0, first_tid]):.4f}")

    inspect_next_token("Q: What planet was Zorglon Blenomorphe born on? A:", "Xebulon")
    inspect_next_token("Q: What year was Zorglon Blenomorphe born? A:", "3021")

if __name__ == "__main__":
    main()
