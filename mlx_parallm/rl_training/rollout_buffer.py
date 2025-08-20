from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List


@dataclass
class ScoredSample:
    prompt: str
    response: str
    score: float
    meta: Dict[str, Any] | None = None


class RolloutBuffer:
    def __init__(self, maxlen: int = 4096):
        self._q: Deque[ScoredSample] = deque(maxlen=maxlen)

    def push(self, items: Iterable[ScoredSample]) -> None:
        for it in items:
            self._q.append(it)

    def sample(self, n: int) -> List[ScoredSample]:
        n = min(n, len(self._q))
        out = []
        for _ in range(n):
            out.append(self._q.popleft())
        return out

