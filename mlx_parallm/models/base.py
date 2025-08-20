import inspect
from dataclasses import dataclass

import mlx.core as mx

def create_additive_causal_mask(N: int, offset: int = 0):
    """Create a standard additive causal mask for uniform offset.

    Returns a (N, offset+N) mask with -inf above causal boundary.
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_additive_causal_mask_variable(N: int, offsets, total_length: int):
    """Create per-row additive causal masks when offsets differ per sequence.

    Args:
        N: number of current query tokens per sequence (uniform, padded).
        offsets: 1D list/array-like of length B with prior lengths per sequence.
        total_length: maximum sequence length across batch (S = max(offset_i) + N).

    Returns:
        mx.array of shape (B, N, total_length) with -inf above causal boundary.
    """
    if not isinstance(offsets, (list, tuple)):
        # Single offset, delegate to standard mask and broadcast
        base = create_additive_causal_mask(N, int(offsets))
        return mx.expand_dims(base, 0)
    B = len(offsets)
    rinds = mx.arange(total_length)
    masks = []
    for off in offsets:
        off_i = int(off)
        linds = mx.arange(off_i, off_i + N)
        mask_i = linds[:, None] < rinds[None]
        masks.append(mask_i * -1e9)
    return mx.stack(masks, axis=0)

class BatchedKVCache:

    def __init__(self, head_dim, n_kv_heads, batch_size=1):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def reset(self, batch_size: int | None = None):
        """Reset the cache for reuse.

        If batch_size differs from the allocated tensors' batch dimension,
        drop references to existing arrays so they will be reallocated on next use.
        Otherwise, just reset the offset so the existing buffers can be reused.
        """
        if batch_size is not None and batch_size != self.batch_size:
            self.batch_size = batch_size
            self.keys = None
            self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            shape = (self.batch_size, self.n_kv_heads, n_steps * self.step, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def offsets(self):
        """Per-row offsets (uniform for this cache)."""
        return [self.offset] * self.batch_size


class PagedKVCache(BatchedKVCache):
    """Per-sequence paged KV cache with independent offsets.

    Maintains a unified backing buffer of shape (B, n_kv_heads, T_cap, head_dim)
    but tracks per-row offsets so that each sequence can advance independently.
    """

    def __init__(self, head_dim, n_kv_heads, batch_size=1):
        super().__init__(head_dim, n_kv_heads, batch_size)
        self.offsets_list = [0] * batch_size

    def _ensure_capacity_for(self, needed_max: int):
        prev = self.keys.shape[2] if self.keys is not None else 0
        if prev >= needed_max:
            return
        need_more = needed_max - prev
        n_steps = (need_more + self.step - 1) // self.step
        shape = (self.batch_size, self.n_kv_heads, n_steps * self.step, self.head_dim)
        new_k = mx.zeros(shape, self.keys.dtype if self.keys is not None else mx.float32)
        new_v = mx.zeros(shape, self.values.dtype if self.values is not None else mx.float32)
        if self.keys is not None:
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
        else:
            self.keys, self.values = new_k, new_v

    def update_and_fetch(self, keys, values):
        """Append keys/values per-row and return slices up to max length.

        keys/values: (B, n_kv_heads, L, head_dim)
        """
        B, _, L, _ = keys.shape
        assert B == self.batch_size, "PagedKVCache batch size mismatch"
        # Compute new per-row end offsets and ensure capacity
        new_offsets = [self.offsets_list[i] + L for i in range(B)]
        max_needed = max(new_offsets)
        self._ensure_capacity_for(max_needed)

        # Write per-row region
        for i in range(B):
            start = self.offsets_list[i]
            end = start + L
            self.keys[i, :, start:end, :] = keys[i]
            self.values[i, :, start:end, :] = values[i]
            self.offsets_list[i] = end

        # Return unified slice up to the max end offset
        return self.keys[..., :max_needed, :], self.values[..., :max_needed, :]

    @property
    def offsets(self):
        return list(self.offsets_list)

    def reset(self, batch_size: int | None = None):
        """Reset the cache and per-row offsets."""
        super().reset(batch_size)
        self.offsets_list = [0] * self.batch_size

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
