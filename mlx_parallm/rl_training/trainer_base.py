from __future__ import annotations

import abc
from typing import Any, Dict, Iterable, Protocol


class RolloutProvider(Protocol):
    def fetch(self, batch_size: int) -> Iterable[Dict[str, Any]]: ...


class RLTrainerBase(abc.ABC):
    def __init__(self, rollout_provider: RolloutProvider):
        self.rollout_provider = rollout_provider

    @abc.abstractmethod
    def step(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Run one optimization step over a batch; returns metrics."""

    def run(self, steps: int, batch_size: int = 8) -> None:
        for _ in range(steps):
            batch = list(self.rollout_provider.fetch(batch_size))
            self.step(batch)

