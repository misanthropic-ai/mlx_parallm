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

    def _complete(self, prompt: str, n: int = 1, max_tokens: int = 2048) -> List[str]:
        import requests, os, logging
        
        # GSM8K system prompt
        system_prompt = """You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""
        
        # Prefill control: MOCK_PREFILL env can be 'none' or 'think'
        prefill_mode = os.getenv("MOCK_PREFILL", "think").strip().lower()
        use_prefill = prefill_mode == "think"
        prefill = "<think>\n" if use_prefill else ""
        
        outs: List[str] = []
        # Use single request with n parameter for batch generation
        # Build messages with optional assistant prefill
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if use_prefill:
            messages.append({"role": "assistant", "content": prefill})

        # Allow env override so smoke tests don't accidentally request huge generations.
        try:
            env_max = int(os.getenv("MOCK_MAX_TOKENS", "").strip() or "0")
        except Exception:
            env_max = 0
        if env_max > 0:
            max_tokens = min(int(max_tokens), int(env_max))

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
                "n": n,
            },
            timeout=None,
        )
        resp.raise_for_status()
        data = resp.json()
        for i, choice in enumerate(data.get("choices", [])):
            # If we used prefill, prepend; else, use content as-is
            content = choice.get("message", {}).get("content", "")
            full_response = (prefill + content) if use_prefill else content
            outs.append(full_response)
            logging.info(
                f"MockAtropos(prefill={prefill_mode}) Response {i+1}/{n} (len={len(full_response)}):\n"
                + full_response[:500] + ("..." if len(full_response) > 500 else "")
            )
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
        import re
        import logging, os
        groups = []
        # GSM8K-style math problem
        prompt = "James has 5 apples. He gives 2 apples to his friend. How many apples does James have left?"
        # Allow overriding n via env for experiments (default 2)
        try:
            n_choices = int(os.getenv("MOCK_N", "2"))
            if n_choices <= 0:
                n_choices = 1
        except Exception:
            n_choices = 2
        for batch_idx in range(batch_size):
            logging.info(f"MockAtropos: Fetching batch {batch_idx+1}/{batch_size}, requesting n={n_choices} completions")
            completions = self._complete(prompt, n=n_choices, max_tokens=2048)
            scores = []
            for i, completion in enumerate(completions):
                # Look for boxed answer format and check if it contains "3"
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', completion)
                if boxed_match:
                    answer = boxed_match.group(1).strip()
                    # Score: 1.0 if answer is "3", else 0.0
                    score = 1.0 if answer == "3" else 0.0
                    logging.info(f"  Completion {i+1}: Found \\boxed{{{answer}}} -> score={score}")
                else:
                    # No boxed answer found
                    score = 0.0
                    logging.info(f"  Completion {i+1}: No \\boxed answer found -> score=0.0")
                scores.append(score)
            
            # Tokenize each completion with the prompt
            toks, masks = [], []
            for c in completions:
                t, m = self._tokenize_pair(prompt, c)
                toks.append(t)
                masks.append(m)
            groups.append(ScoredDataGroup(tokens=toks, masks=masks, scores=scores))
        return groups
