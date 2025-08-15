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

        # Convert to numpy for convenience
        queries_np = np.array(queries)  # (B, H, L, D)
        mem_k = np.array(self.store[model_id]["memory_keys"])  # (N, n_kv_heads, D)
        mem_v = np.array(self.store[model_id]["memory_values"])  # (N, n_kv_heads, D)

        B, H, L, D = queries_np.shape
        N, n_kv_heads, Dk = mem_k.shape
        assert D == Dk, f"Query dim {D} != memory dim {Dk}"

        # Normalize
        qn = queries_np / (np.linalg.norm(queries_np, axis=-1, keepdims=True) + 1e-8)
        mk = mem_k / (np.linalg.norm(mem_k, axis=-1, keepdims=True) + 1e-8)

        # Allocate outputs
        selected_keys_heads = []
        selected_values_heads = []
        similarities_heads = []
        indices_heads = []

        for h in range(H):
            kv_h = h % n_kv_heads
            mk_h = mk[:, kv_h, :]  # (N, D) normalized for search
            rk_h = mem_k[:, kv_h, :]  # (N, D) raw keys for scoring
            q_h = qn[:, h, :, :]    # (B, L, D)

            # (B*L, D) @ (D, N) -> (B*L, N)
            q2 = q_h.reshape(B * L, D)
            sims = q2 @ mk_h.T

            # topk along N
            if topk > N:
                tk = N
            else:
                tk = topk

            idx = np.argpartition(-sims, kth=tk-1, axis=1)[:, :tk]
            # Sort top-k for stability
            row_idx = np.arange(idx.shape[0])[:, None]
            part = sims[row_idx, idx]
            order = np.argsort(-part, axis=1)
            idx_sorted = idx[row_idx, order]
            val_sorted = part[row_idx, order]

            # Gather keys/values
            sel_k = rk_h[idx_sorted]  # (B*L, tk, D) use raw keys, not normalized
            sel_v = mem_v[:, kv_h, :][idx_sorted]  # (B*L, tk, D)

            # Reshape to (B, L, tk, D)
            sel_k = sel_k.reshape(B, L, tk, D)
            sel_v = sel_v.reshape(B, L, tk, D)
            vals = val_sorted.reshape(B, L, tk)
            ids = idx_sorted.reshape(B, L, tk)

            selected_keys_heads.append(sel_k)
            selected_values_heads.append(sel_v)
            similarities_heads.append(vals)
            indices_heads.append(ids)

        # Stack heads -> (B, H, L, tk, D)
        selected_keys = np.stack(selected_keys_heads, axis=1)
        selected_values = np.stack(selected_values_heads, axis=1)
        similarities = np.stack(similarities_heads, axis=1)
        indices = np.stack(indices_heads, axis=1)

        # Reshape keys/values to (B, H, L*tk, D)
        Bn, Hn, Ln, TKn, Dn = selected_keys.shape
        selected_keys = selected_keys.reshape(Bn, Hn, Ln * TKn, Dn)
        selected_values = selected_values.reshape(Bn, Hn, Ln * TKn, Dn)

        return mx.array(selected_keys), mx.array(selected_values), mx.array(similarities), mx.array(indices)

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
