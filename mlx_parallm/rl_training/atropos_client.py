from __future__ import annotations

import itertools
from typing import Dict, Iterable, Iterator, List, Optional
import asyncio
import json

try:
    import aiohttp
except Exception:  # pragma: no cover - optional at runtime
    aiohttp = None  # type: ignore

from .rollout_buffer import ScoredSample
from .types import ScoredDataGroup


class AtroposClient:
    """HTTP client for Atropos rollouts.

    API endpoints:
    - POST /register -> {"uuid": str} - Register trainer with Atropos
    - GET /batch -> {"batch": List[ScoredDataGroup]} - Fetch training batches
    """

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self._uuid: Optional[str] = None

    async def _post_json(self, path: str, payload: Dict) -> Dict:
        if aiohttp is None:
            raise RuntimeError("aiohttp not available; cannot use Atropos client.")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}{path}", json=payload, timeout=60) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def register_async(self, info: Dict) -> str:
        data = await self._post_json("/register", info)
        self._uuid = data.get("uuid")
        if not self._uuid:
            raise RuntimeError("Atropos register did not return uuid")
        return self._uuid

    async def fetch_batch_async(self, batch_size: int, token_budget: int, require_same_env: bool = False) -> List[ScoredDataGroup]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/batch", timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
        
        batch = data.get("batch")
        if batch is None:
            return []
        
        out: List[ScoredDataGroup] = []
        for item in batch:
            group = ScoredDataGroup(
                tokens=item.get("tokens", []),
                masks=item.get("masks", []),
                scores=item.get("scores", []),
            )
            # Add optional fields if present
            if "advantages" in item:
                group["advantages"] = item["advantages"]
            if "ref_logprobs" in item:
                group["ref_logprobs"] = item["ref_logprobs"]
            if "messages" in item:
                group["messages"] = item["messages"]
            if "overrides" in item:
                group["overrides"] = item["overrides"]
            if "group_overrides" in item:
                group["group_overrides"] = item["group_overrides"]
            if "images" in item:
                group["images"] = item["images"]
            if "env_id" in item:
                group["env_id"] = item["env_id"]
            out.append(group)
        return out

    # Synchronous wrappers
    def register(self, info: Dict) -> str:
        return asyncio.get_event_loop().run_until_complete(self.register_async(info))

    def fetch_batch(self, batch_size: int, token_budget: int, require_same_env: bool = False) -> List[ScoredDataGroup]:
        return asyncio.get_event_loop().run_until_complete(self.fetch_batch_async(batch_size, token_budget, require_same_env))


class MockAtroposClient:
    """Generates synthetic scored samples for smoke tests."""

    def __init__(self):
        self._counter: Iterator[int] = itertools.count(1)

    def fetch(self, batch_size: int) -> Iterable[ScoredDataGroup]:
        """Generate mock ScoredDataGroups with multiple trajectories each.
        
        Each group contains 2-3 alternative trajectories with different rewards.
        """
        groups = []
        for _ in range(batch_size):
            # Create 2-3 alternative trajectories for each group
            # Simulating different completions for the same prompt
            
            # Trajectory 1: Good response (high reward)
            toks1 = [1, 2, 3, 4, 5, 6, 7, 8]  # Prompt + response tokens
            mask1 = [0, 0, 0, 0, 1, 1, 1, 1]  # Last 4 are response tokens
            score1 = 1.0  # High reward
            
            # Trajectory 2: Mediocre response (medium reward)  
            toks2 = [1, 2, 3, 4, 10, 11]  # Same prompt, different response
            mask2 = [0, 0, 0, 0, 1, 1]     # Last 2 are response tokens
            score2 = 0.0  # Medium reward
            
            # Trajectory 3: Poor response (low reward)
            toks3 = [1, 2, 3, 4, 20, 21, 22]  # Same prompt, another response
            mask3 = [0, 0, 0, 0, 1, 1, 1]     # Last 3 are response tokens
            score3 = -0.5  # Low reward
            
            # Create the group with multiple trajectories
            group = ScoredDataGroup(
                tokens=[toks1, toks2, toks3],
                masks=[mask1, mask2, mask3],
                scores=[score1, score2, score3],
            )
            
            # Optionally add precomputed advantages (aligned with ALL tokens per trajectory)
            # Here we're not providing them, letting GRPO compute from scores
            
            groups.append(group)
            
        return groups
