"""MEGa: Memory-Embedded Gated LoRA for LLaMA (MLX).

This module scaffolds a MEGa-style integration where:
- The base model stays frozen.
- Each memory is represented by a context key embedding and a set of small
  LoRA adapters (low-rank matrices) applied to selected layers.
- At inference, a simple cosine-similarity gate selects top-k relevant memories
  and mixes their adapters into the forward pass.

Initial implementation focuses on clean structure and pluggability:
- Context key extraction via pooled hidden states.
- A lightweight LoRA wrapper for Linear layers that can sum multiple adapters
  weighted by gating scores.
- A GatedAdapterManager that stores context keys and adapters per memory and
  provides gating for a given input.

Notes:
- Training of per-memory adapters is not included in this scaffold.
- Start by adapting MLP projections (up/down). Attention projection adapters
  can be added similarly if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import mlx.core as mx
import mlx.nn as nn

from .llama import ModelArgs as LlamaModelArgs, LlamaModel, TransformerBlock, Attention, MLP
from .base import BatchedKVCache, create_additive_causal_mask

logger = logging.getLogger(__name__)


# -----------------------------
# Args
# -----------------------------


@dataclass
class MEGaModelArgs(LlamaModelArgs):
    """LLaMA args extended with MEGa gating options."""

    use_mega: bool = True
    gate_topk: int = 2
    gate_threshold: float = 0.4
    adapter_rank: int = 8
    adapter_alpha: float = 1.0
    # Where to apply LoRA: "mlp" or "mlp_attn" (future)
    lora_target: str = "mlp"


# -----------------------------
# LoRA wrapper
# -----------------------------


def apply_lora_delta(x: mx.array, adapters: Dict[str, Tuple[mx.array, mx.array]], gating: Optional[List[Tuple[str, float]]], rank: int, alpha: float) -> mx.array:
    """Compute sum_i g_i * alpha/r * (x @ A_i) @ B_i^T for provided adapters."""
    if not gating:
        return mx.zeros((x.shape[0], x.shape[1], adapters[next(iter(adapters))][1].shape[0])) if adapters else mx.zeros((x.shape[0], x.shape[1], 0))
    scale = alpha / float(rank)
    acc = None
    for mem_id, g in gating:
        if g <= 0.0:
            continue
        ab = adapters.get(mem_id)
        if ab is None:
            continue
        A, B = ab
        xr = x @ A  # (B,L,r)
        delta = (xr @ B.T) * (g * scale)
        acc = delta if acc is None else acc + delta
    return acc if acc is not None else mx.zeros_like(x @ mx.zeros((x.shape[-1], 0)))


# -----------------------------
# Gated adapter manager
# -----------------------------


class GatedAdapterManager:
    """Stores context keys + per-memory adapters and provides gating."""

    def __init__(self, key_dim: int, topk: int = 2, threshold: float = 0.4):
        self.key_dim = key_dim
        self.topk = topk
        self.threshold = threshold
        self.keys: Dict[str, mx.array] = {}  # mem_id -> (key_dim,)
        # Store adapter tensors per memory per target name
        self.adapters: Dict[str, Dict[str, Tuple[mx.array, mx.array]]] = {}

    def register_target(self, name: str):
        # Kept for API symmetry; targets resolved by name at call time
        return

    def add_memory(self, mem_id: str, key: mx.array, adapters: Dict[str, Tuple[mx.array, mx.array]]):
        if key.shape != (self.key_dim,):
            raise ValueError(f"key shape {key.shape} != ({self.key_dim},)")
        self.keys[mem_id] = key
        self.adapters[mem_id] = adapters
        # Registration by name; no direct module binding needed
        
    def gate(self, query_key: mx.array) -> List[Tuple[str, float]]:
        """Return top-k (mem_id, weight) based on cosine similarity and threshold."""
        if not self.keys:
            return []
        # Stack keys
        mem_ids = list(self.keys.keys())
        K = mx.stack([self.keys[mid] for mid in mem_ids], axis=0)  # (N, D)
        q = query_key  # (D,)
        # Normalize
        qn = q / (mx.linalg.norm(q) + 1e-8)
        kn = K / (mx.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        sims = kn @ qn  # (N,)
        # Top-k
        order = mx.argsort(-sims)
        k = min(self.topk, sims.shape[0])
        idx = order[:k]
        vals = sims[idx]
        # Threshold and normalize weights to sum<=1
        pairs: List[Tuple[str, float]] = []
        total = 0.0
        for i in range(int(k)):
            w = float(vals[i])
            if w < self.threshold:
                continue
            pairs.append((mem_ids[int(idx[i])], w))
            total += w
        if total > 0:
            pairs = [(mid, w / total) for mid, w in pairs]
        return pairs


# -----------------------------
# Modules with MEGa integration
# -----------------------------


class MEGaMLP(nn.Module):
    """Applies LoRA deltas alongside the base MLP without altering parameter names."""

    def __init__(self, base: MLP, layer_idx: int, rank: int, alpha: float, manager: GatedAdapterManager):
        super().__init__()
        self.base = base
        self.layer_idx = layer_idx
        self.rank = rank
        self.alpha = alpha
        self.manager = manager

    def __call__(self, x: mx.array, gating: Optional[List[Tuple[str, float]]] = None) -> mx.array:
        # up_proj with optional delta
        up_base = self.base.up_proj(x)
        # Collect adapters for this target if present
        up_name = f"L{self.layer_idx}.mlp.up_proj"
        up_adapters = {mid: ab[up_name] for mid, ab in self.manager.adapters.items() if up_name in ab}
        if up_adapters:
            delta_up = apply_lora_delta(x, up_adapters, gating, self.rank, self.alpha)
            up = up_base + delta_up
        else:
            up = up_base
        act = nn.silu(self.base.gate_proj(x)) * up
        # down_proj with optional delta
        down_base = self.base.down_proj(act)
        down_name = f"L{self.layer_idx}.mlp.down_proj"
        down_adapters = {mid: ab[down_name] for mid, ab in self.manager.adapters.items() if down_name in ab}
        if down_adapters:
            delta_down = apply_lora_delta(act, down_adapters, gating, self.rank, self.alpha)
            return down_base + delta_down
        return down_base


class MEGaTransformerBlock(TransformerBlock):
    """Transformer block that accepts gating and routes it to LoRA targets."""

    def __init__(self, args: MEGaModelArgs, base_block: TransformerBlock, manager: GatedAdapterManager, layer_idx: int):
        super().__init__(args)
        # Copy over base modules
        self.self_attn: Attention = base_block.self_attn
        # Keep base modules; wrap MLP forward behavior via MEGaMLP helper
        self.mlp = base_block.mlp
        self._mega_mlp = MEGaMLP(base_block.mlp, layer_idx, args.adapter_rank, args.adapter_alpha, manager)
        self.input_layernorm = base_block.input_layernorm
        self.post_attention_layernorm = base_block.post_attention_layernorm
        self.args = args
        self.layer_idx = layer_idx
        self.manager = manager
        # Register target names (optional)
        self.manager.register_target(f"L{layer_idx}.mlp.up_proj")
        self.manager.register_target(f"L{layer_idx}.mlp.down_proj")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BatchedKVCache] = None,
        gating: Optional[List[Tuple[str, float]]] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self._mega_mlp(self.post_attention_layernorm(h), gating=gating)
        out = h + r
        return out


class MEGaLlamaForward:
    """Helper to run forward with MEGa deltas using a standard LlamaModel.

    Not an nn.Module, to avoid registering duplicate parameter trees.
    """

    def __init__(self, base_model: LlamaModel, manager: GatedAdapterManager, args: MEGaModelArgs):
        self.base = base_model
        self.manager = manager
        self.args = args

    def extract_context_key(self, inputs: mx.array) -> mx.array:
        out = self.base(inputs)
        key = mx.mean(out, axis=1)[0]
        return key

    def __call__(self, inputs: mx.array, cache=None, gating: Optional[List[Tuple[str, float]]] = None):
        h = self.base.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = create_additive_causal_mask(
                h.shape[1], cache[0].offset if cache is not None else 0
            ).astype(h.dtype)
        if cache is None:
            cache = [None] * len(self.base.layers)
        for i, (layer, c) in enumerate(zip(self.base.layers, cache)):
            # Attention as usual
            attn_out, preproj = layer.self_attn(layer.input_layernorm(h), mask, c, return_preproj=True)
            # Optional attention o_proj LoRA: delta computed on preproj and added to o_proj(preproj)
            attn_name = f"L{i}.attn.delta"
            attn_adapters = {mid: ab[attn_name] for mid, ab in self.manager.adapters.items() if attn_name in ab}
            if attn_adapters:
                attn_out = attn_out + apply_lora_delta(preproj, attn_adapters, gating, self.args.adapter_rank, self.args.adapter_alpha)
            h = h + attn_out
            # MLP with MEGa deltas
            x2 = layer.post_attention_layernorm(h)
            up_base = layer.mlp.up_proj(x2)
            up_name = f"L{i}.mlp.up_proj"
            up_adapters = {mid: ab[up_name] for mid, ab in self.manager.adapters.items() if up_name in ab}
            if up_adapters:
                up = up_base + apply_lora_delta(x2, up_adapters, gating, self.args.adapter_rank, self.args.adapter_alpha)
            else:
                up = up_base
            act = nn.silu(layer.mlp.gate_proj(x2)) * up
            down_base = layer.mlp.down_proj(act)
            down_name = f"L{i}.mlp.down_proj"
            down_adapters = {mid: ab[down_name] for mid, ab in self.manager.adapters.items() if down_name in ab}
            if down_adapters:
                r_mlp = down_base + apply_lora_delta(act, down_adapters, gating, self.args.adapter_rank, self.args.adapter_alpha)
            else:
                r_mlp = down_base
            h = h + r_mlp
        return self.base.norm(h)


class MEGaModel(nn.Module):
    """Causal LM head wrapper for MEGa.

    The LM head is identical to the standard LLaMA wrapper.
    """

    def __init__(self, args: MEGaModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        self.manager = GatedAdapterManager(key_dim=args.hidden_size, topk=args.gate_topk, threshold=args.gate_threshold)
        self.mega_forward = MEGaLlamaForward(self.model, self.manager, args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def add_memory(self, mem_id: str, text_tokens: mx.array, adapters: Dict[str, Tuple[mx.array, mx.array]]):
        """Register a memory (context key + adapters) into the MEGa manager.

        Args:
            mem_id: Identifier for this memory sample.
            text_tokens: Tokenized memory text, shape (1, L) or (L,).
            adapters: Mapping from target-name -> (A, B) matrices.
        """
        if text_tokens.ndim == 1:
            text_tokens = text_tokens.reshape(1, -1)
        key = self.mega_forward.extract_context_key(text_tokens)
        self.manager.add_memory(mem_id, key, adapters)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        # Compute query key and gating, then run model with gating
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        query_key = self.mega_forward.extract_context_key(inputs)
        gating = self.manager.gate(query_key)
        out = self.mega_forward(inputs, cache, gating=gating)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def train_memory(
        self,
        mem_id: str,
        text_tokens: Optional[mx.array] = None,
        layers: Optional[List[int]] = None,
        rank: int = 8,
        alpha: float = 1.0,
        lr: float = 1e-2,
        steps: int = 50,
        examples: Optional[List[mx.array]] = None,
        verbose: bool = False,
        log_every: int = 50,
        use_attn_delta: bool = False,
    ) -> Dict[str, Tuple[mx.array, mx.array]]:
        """Fit per-memory LoRA adapters on selected layers using a simple LM loss.

        Args:
            mem_id: Memory identifier.
            text_tokens: (L,) or (1, L) tokens containing the fact to memorize.
            layers: Layer indices to adapt (defaults to last 2 layers).
            rank: LoRA rank.
            alpha: LoRA scaling.
            lr: Learning rate.
            steps: Optimization steps.

        Returns:
            Dict[target_name, (A, B)] learned adapter matrices.
        """
        # Build example list
        sample_list: List[mx.array] = []
        if examples is not None and len(examples) > 0:
            for t in examples:
                sample_list.append(t.reshape(1, -1) if t.ndim == 1 else t)
        elif text_tokens is not None:
            sample_list.append(text_tokens.reshape(1, -1) if text_tokens.ndim == 1 else text_tokens)
        else:
            raise ValueError("Either text_tokens or examples must be provided")
        hid = self.args.hidden_size
        inter = self.args.intermediate_size

        if layers is None:
            layers = [len(self.model.layers) - 2, len(self.model.layers) - 1]

        # Initialize or warm-start adapters
        adapters: Dict[str, Tuple[mx.array, mx.array]] = {}
        if mem_id in self.manager.adapters:
            adapters = {k: (v[0], v[1]) for k, v in self.manager.adapters[mem_id].items()}
        else:
            for li in layers:
                adapters[f"L{li}.mlp.up_proj"] = (
                    (mx.random.uniform(shape=(hid, rank)) - 0.5) * 1e-3,
                    (mx.random.uniform(shape=(inter, rank)) - 0.5) * 1e-3,
                )
                adapters[f"L{li}.mlp.down_proj"] = (
                    (mx.random.uniform(shape=(inter, rank)) - 0.5) * 1e-3,
                    (mx.random.uniform(shape=(hid, rank)) - 0.5) * 1e-3,
                )
                if use_attn_delta:
                    adapters[f"L{li}.attn.delta"] = (
                        (mx.random.uniform(shape=(hid, rank)) - 0.5) * 1e-3,
                        (mx.random.uniform(shape=(hid, rank)) - 0.5) * 1e-3,
                    )

        # Average context key across examples
        keys = [self.mega_forward.extract_context_key(t) for t in sample_list]
        key = mx.mean(mx.stack(keys, axis=0), axis=0)
        self.manager.add_memory(mem_id, key, adapters)

        # Build flat parameter list for autograd
        param_names: List[Tuple[str, str]] = []  # (target, 'A'|'B')
        params: List[mx.array] = []
        for t, (A, B) in adapters.items():
            param_names.append((t, 'A'))
            params.append(A)
            param_names.append((t, 'B'))
            params.append(B)

        def loss_fn(*flat_params: mx.array) -> mx.array:
            # Rebuild adapters with current params
            it = iter(flat_params)
            for t in adapters.keys():
                A = next(it)
                B = next(it)
                adapters[t] = (A, B)
            # Force manager to point to updated adapters
            self.manager.adapters[mem_id] = adapters
            # Average loss over all examples
            losses = []
            for t in sample_list:
                logits = self.mega_forward(t, cache=None, gating=[(mem_id, 1.0)])
                logits = logits[:, :-1, :]
                targets = t[:, 1:]
                probs = mx.softmax(logits, axis=-1)
                logp = mx.log(probs + 1e-9)
                nll = -mx.take_along_axis(logp, targets[:, :, None], axis=-1).squeeze(-1)
                losses.append(mx.mean(nll))
            return mx.mean(mx.stack(losses, axis=0))

        grad_fn = mx.grad(loss_fn)

        # Adam optimizer state
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_list = [mx.zeros_like(p) for p in params]
        v_list = [mx.zeros_like(p) for p in params]

        for i in range(1, steps + 1):
            grads = grad_fn(*params)
            new_params = []
            new_m = []
            new_v = []
            for p, g, m, v in zip(params, grads, m_list, v_list):
                m_t = beta1 * m + (1.0 - beta1) * g
                v_t = beta2 * v + (1.0 - beta2) * (g * g)
                m_hat = m_t / (1.0 - beta1**i)
                v_hat = v_t / (1.0 - beta2**i)
                p_t = p - lr * m_hat / (mx.sqrt(v_hat) + eps)
                new_params.append(p_t)
                new_m.append(m_t)
                new_v.append(v_t)
            params = new_params
            m_list = new_m
            v_list = new_v
            mx.eval(*params)
            if verbose and (i % max(1, log_every) == 0 or i == 1 or i == steps):
                # Compute current loss for logging
                cur_loss = []
                for t in sample_list:
                    logits = self.mega_forward(t, cache=None, gating=[(mem_id, 1.0)])
                    logits = logits[:, :-1, :]
                    targets = t[:, 1:]
                    probs = mx.softmax(logits, axis=-1)
                    logp = mx.log(probs + 1e-9)
                    nll = -mx.take_along_axis(logp, targets[:, :, None], axis=-1).squeeze(-1)
                    cur_loss.append(mx.mean(nll))
                cur_loss = float(mx.mean(mx.stack(cur_loss, axis=0)))
                print(f"[MEGa][{mem_id}] step {i}/{steps} loss={cur_loss:.4f}")

        # Write back final params
        it = iter(params)
        for t in adapters.keys():
            A = next(it)
            B = next(it)
            adapters[t] = (A, B)
        # Keep registered
        self.manager.adapters[mem_id] = adapters
        return adapters

    def save_memory(self, mem_id: str, out_dir: str) -> None:
        """Save adapters and context key for a memory into a safetensors file.

        Creates `<out_dir>/adapters.safetensors` with keys:
         - `context_key`
         - `targets/{target}/A` and `targets/{target}/B` for each adapter
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        if mem_id not in self.manager.keys or mem_id not in self.manager.adapters:
            raise ValueError(f"Memory {mem_id} not found")
        key = self.manager.keys[mem_id]
        adapters = self.manager.adapters[mem_id]
        to_save = {"context_key": key}
        for t, (A, B) in adapters.items():
            to_save[f"targets/{t}/A"] = A
            to_save[f"targets/{t}/B"] = B
        mx.save_safetensors(os.path.join(out_dir, "adapters.safetensors"), to_save, metadata={"format": "mlx"})

    def load_memory(self, mem_id: str, in_dir: str) -> None:
        """Load adapters and context key for a memory from a safetensors file and register it."""
        import os
        path = os.path.join(in_dir, "adapters.safetensors")
        data = mx.load(path)
        if "context_key" not in data:
            raise ValueError("Missing context_key in adapters file")
        key = data["context_key"]
        adapters: Dict[str, Tuple[mx.array, mx.array]] = {}
        # Parse keys like targets/{target}/A and /B
        targets = {}
        for k, v in data.items():
            if not k.startswith("targets/"):
                continue
            _, rest = k.split("/", 1)
            tname, part = rest.rsplit("/", 1)
            targets.setdefault(tname, {})[part] = v
        for tname, parts in targets.items():
            if "A" in parts and "B" in parts:
                adapters[tname] = (parts["A"], parts["B"])
        self.manager.add_memory(mem_id, key, adapters)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
