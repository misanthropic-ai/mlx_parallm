from __future__ import annotations

from typing import Dict, Set, Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import mlx.nn as nn


ADAPTER_KEYWORDS = ("lora", "adapter", "dora")


def adapter_param_names(model: nn.Module) -> Set[str]:
    params: Dict[str, mx.array] = dict(tree_flatten(model.parameters()))
    names: Set[str] = set()
    for k in params.keys():
        lk = k.lower()
        if any(word in lk for word in ADAPTER_KEYWORDS):
            names.add(k)
    return names


def zero_non_adapter_grads(model: nn.Module, grads: Any) -> Any:
    # grads has same tree-structure as model.parameters()
    flat_params = tree_flatten(model.parameters())
    flat_grads = tree_flatten(grads)

    names = adapter_param_names(model)
    out = []
    for (k, g) in flat_grads:
        if k in names and g is not None:
            out.append((k, g))
        else:
            # create a zero gradient with same shape/dtype
            # Find the corresponding parameter to get shape/dtype
            p = None
            for pk, pv in flat_params:
                if pk == k:
                    p = pv
                    break
            if p is not None:
                out.append((k, mx.zeros_like(p)))
            else:
                out.append((k, g))  # Fallback to original gradient
    return tree_unflatten(out)


def adapter_weights(model: nn.Module) -> Dict[str, mx.array]:
    """Extract only adapter-related parameters by name.

    Returns a flat dict suitable for save_weights.
    """
    params: Dict[str, mx.array] = dict(tree_flatten(model.parameters()))
    names = adapter_param_names(model)
    return {k: v for k, v in params.items() if k in names}
