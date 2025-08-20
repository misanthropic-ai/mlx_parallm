import inspect
from dataclasses import dataclass

import mlx.core as mx

def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9

class BatchedKVCache:

    def __init__(self, head_dim, n_kv_heads, batch_size=1):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

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
