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

    async def fetch_batch_async(self, batch_size: int, token_budget: int, require_same_env: bool = False, poll: bool = True) -> List[ScoredDataGroup]:
        """Fetch a batch from Atropos. If poll=True, keep trying until we get data."""
        import logging
        poll_count = 0
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/batch", timeout=60) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            
            batch = data.get("batch")
            if batch is not None or not poll:
                if poll_count > 0:
                    logging.info(f"Got batch after {poll_count} poll attempts")
                break
            
            poll_count += 1
            if poll_count % 5 == 0:
                logging.info(f"Polling for batch... attempt {poll_count}")
            
            # Wait a bit before polling again
            await asyncio.sleep(1.0)
        
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

    def fetch_batch(self, batch_size: int, token_budget: int, require_same_env: bool = False, poll: bool = True) -> List[ScoredDataGroup]:
        return asyncio.get_event_loop().run_until_complete(self.fetch_batch_async(batch_size, token_budget, require_same_env, poll))


class MockAtroposClient:
    """End-to-end mock: queries the running server to produce scored samples.

    Builds ScoredDataGroups with multiple alternative completions for the same prompt,
    tokenizes them, and masks prompt tokens out so only generated tokens are trained.
    """

    def __init__(self, base_url: str, model_id: str, tokenizer):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.tokenizer = tokenizer
        self._counter: Iterator[int] = itertools.count(1)

    def _complete(self, prompt: str, n: int = 1, max_tokens: int = 16) -> List[str]:
        import requests
        outs: List[str] = []
        for _ in range(n):
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.95,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["text"]
            outs.append(text)
        return outs

    def _tokenize_pair(self, prompt: str, response: str):
        tok = self.tokenizer._tokenizer
        enc_prompt = tok([prompt], return_tensors="np", padding=False)
        enc_full = tok([prompt + response], return_tensors="np", padding=False)
        p = enc_prompt["input_ids"][0].tolist()
        f = enc_full["input_ids"][0].tolist()
        mask = [0] * len(p) + [1] * (len(f) - len(p))
        return f, mask

    def fetch(self, batch_size: int) -> Iterable[ScoredDataGroup]:
        import random
        groups = []
        # Fixed simple arithmetic prompt to test consistency
        prompt = "What is 2 + 2? Answer briefly."
        for _ in range(batch_size):
            # Ask server for 4 alternative completions
            completions = self._complete(prompt, n=4, max_tokens=8)
            # Score: 1.0 if includes '4', else 0.0
            scores = [1.0 if ("4" in c) else 0.0 for c in completions]
            # Tokenize each
            toks, masks = [], []
            for c in completions:
                t, m = self._tokenize_pair(prompt, c)
                toks.append(t)
                masks.append(m)
            groups.append(ScoredDataGroup(tokens=toks, masks=masks, scores=scores))
        return groups
