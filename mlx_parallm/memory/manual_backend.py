"""Manual (numpy-based) memory backend implementation for correctness checks."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import mlx.core as mx

from .base import MemoryBackend


class ManualMemoryBackend(MemoryBackend):
    """Manual cosine-similarity backend without FAISS, suitable for small memories."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        # model_id -> {"memory_keys": mx.array [N, n_kv_heads, D], "memory_values": mx.array [N, n_kv_heads, D]}
        self.store: Dict[str, Dict[str, mx.array]] = {}

    def add_memories(
        self,
        model_id: str,
        memory_keys: mx.array,
        memory_values: mx.array,
        memory_ids: Optional[List[int]] = None,
    ) -> None:
        # Expect shapes: (N, n_kv_heads, D)
        if model_id not in self.store:
            self.store[model_id] = {
                "memory_keys": memory_keys,
                "memory_values": memory_values,
            }
        else:
            self.store[model_id]["memory_keys"] = mx.concatenate(
                [self.store[model_id]["memory_keys"], memory_keys], axis=0
            )
            self.store[model_id]["memory_values"] = mx.concatenate(
                [self.store[model_id]["memory_values"], memory_values], axis=0
            )

    def search(
        self,
        model_id: str,
        queries: mx.array,
        topk: int = 10,
        layer_idx: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        if model_id not in self.store:
            raise ValueError(f"No memories found for model {model_id}")

        q = queries  # (B, H, L, D)
        mem_k = self.store[model_id]["memory_keys"]  # (N, n_kv_heads, D)
        mem_v = self.store[model_id]["memory_values"]  # (N, n_kv_heads, D)

        B, H, L, D = q.shape
        N, n_kv_heads, Dk = mem_k.shape
        if D != Dk:
            raise ValueError(f"Query dim {D} != memory dim {Dk}")

        # Normalize
        qn = q / (mx.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        mk = mem_k / (mx.linalg.norm(mem_k, axis=-1, keepdims=True) + 1e-8)

        selected_keys_heads = []
        selected_values_heads = []
        similarities_heads = []
        indices_heads = []

        for h in range(H):
            kv_h = h % n_kv_heads
            mk_h = mk[:, kv_h, :]            # (N, D) normalized for search
            rk_h = mem_k[:, kv_h, :]         # (N, D) raw keys for scoring
            rv_h = mem_v[:, kv_h, :]         # (N, D) raw values
            q_h = qn[:, h, :, :]             # (B, L, D)

            sims = mx.matmul(q_h.reshape(B * L, D), mk_h.T)  # (B*L, N)
            tk = min(topk, N)
            # argsort descending and take top-k
            order = mx.argsort(-sims, axis=1)
            idx_sorted = order[:, :tk]  # (B*L, tk)
            # Gather top-k similarity values per row
            row_indices = mx.arange(B * L).reshape(B * L, 1)
            vals = sims[row_indices, idx_sorted]

            # Gather keys/values per row using loops (row-wise indices)
            sel_k_list = []
            sel_v_list = []
            for r in range(B * L):
                idx_row = idx_sorted[r]
                sel_k_list.append(mx.take(rk_h, idx_row, axis=0))
                sel_v_list.append(mx.take(rv_h, idx_row, axis=0))
            sel_k = mx.stack(sel_k_list, axis=0)  # (B*L, tk, D)
            sel_v = mx.stack(sel_v_list, axis=0)  # (B*L, tk, D)

            # Reshape to (B, L, tk, D) and vals to (B, L, tk)
            sel_k = sel_k.reshape(B, L, tk, D)
            sel_v = sel_v.reshape(B, L, tk, D)
            vals = vals.reshape(B, L, tk)
            ids = idx_sorted.reshape(B, L, tk)

            selected_keys_heads.append(sel_k)
            selected_values_heads.append(sel_v)
            similarities_heads.append(vals)
            indices_heads.append(ids)

        selected_keys = mx.stack(selected_keys_heads, axis=1)     # (B, H, L, tk, D)
        selected_values = mx.stack(selected_values_heads, axis=1) # (B, H, L, tk, D)
        similarities = mx.stack(similarities_heads, axis=1)       # (B, H, L, tk)
        indices = mx.stack(indices_heads, axis=1)                 # (B, H, L, tk)

        # Reshape keys/values to (B, H, L*tk, D)
        Bn, Hn, Ln, TKn, Dn = selected_keys.shape
        selected_keys = selected_keys.reshape(Bn, Hn, Ln * TKn, Dn)
        selected_values = selected_values.reshape(Bn, Hn, Ln * TKn, Dn)

        return selected_keys, selected_values, similarities, indices

    def clear(self, model_id: str) -> None:
        if model_id in self.store:
            del self.store[model_id]

    def list_memories(self, model_id: str) -> Dict:
        if model_id not in self.store:
            return {"num_memories": 0, "exists": False}
        mem_k = self.store[model_id]["memory_keys"]
        return {
            "num_memories": int(mem_k.shape[0]) if hasattr(mem_k, "shape") else 0,
            "embedding_dim": self.embedding_dim,
            "exists": True,
        }

    def memory_exists(self, model_id: str) -> bool:
        return model_id in self.store

    def get_all(self, model_id: str) -> Tuple[mx.array, mx.array]:
        if model_id not in self.store:
            raise ValueError(f"No memories found for model {model_id}")
        return self.store[model_id]["memory_keys"], self.store[model_id]["memory_values"]
