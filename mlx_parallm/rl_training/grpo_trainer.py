from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_parallm.server.schemas import InternalModelRecord
from .trainer_base import RLTrainerBase
from .rollout_buffer import ScoredSample
from .types import ScoredDataGroup
from .param_utils import zero_non_adapter_grads
import mlx.optimizers as optim


@dataclass
class GRPOConfig:
    kl_beta: float = 0.05
    entropy_weight: float = 0.0  # not used in smoke
    max_tokens: int = 256        # limit per response for safety
    learning_rate: float = 1e-5


class GRPOTrainer(RLTrainerBase):
    def __init__(
        self,
        policy_record: InternalModelRecord,
        ref_record: Optional[InternalModelRecord],
        tokenizer: TokenizerWrapper,
        cfg: Optional[GRPOConfig] = None,
    ) -> None:
        super().__init__(rollout_provider=None)  # Provider supplied externally in train loop
        if policy_record.model_instance is None:
            raise RuntimeError("Policy model not loaded.")
        self.policy_record = policy_record
        self.ref_record = ref_record
        self.tokenizer = tokenizer
        self.cfg = cfg or GRPOConfig()
        # Optimizer over full model; grads for non-adapter params will be zeroed
        self.optimizer = optim.Adam(self.cfg.learning_rate)
        self.optimizer_state = self.optimizer.init(self.policy_record.model_instance)

    def _token_ids_with_split(self, prompt: str, response: str) -> Tuple[mx.array, int, int]:
        tok = self.tokenizer._tokenizer
        enc_prompt = tok([prompt], return_tensors="np", padding=False)
        enc_full = tok([prompt + response], return_tensors="np", padding=False)
        prompt_ids = enc_prompt["input_ids"][0].tolist()
        full_ids = enc_full["input_ids"][0].tolist()
        # Truncate overly long sequences to avoid excessive compute in smoke
        if len(full_ids) - len(prompt_ids) > self.cfg.max_tokens:
            full_ids = full_ids[: len(prompt_ids) + self.cfg.max_tokens]
        x = mx.array([full_ids])
        return x, len(prompt_ids), len(full_ids)

    def _logprob_for_realized_tokens(self, model, x: mx.array, start_idx: int, end_idx: int) -> List[float]:
        logits = model(x)  # (1, T, V)
        T = logits.shape[1]
        # we use positions [start_idx-1 .. end_idx-2] as predictors for targets [start_idx .. end_idx-1]
        out: List[float] = []
        for i in range(max(1, start_idx) - 1, end_idx - 1):
            li = logits[:, i, :]
            pi = mx.softmax(li, axis=-1)
            tgt = int(x[0, i + 1].item())
            lp = float(mx.log(pi[0, tgt]).item())
            out.append(lp)
        return out

    def _delta_logprob_vs_ref(self, prompt: str, response: str) -> Tuple[float, int]:
        x, s, e = self._token_ids_with_split(prompt, response)
        policy = self.policy_record.model_instance
        lps_pol = self._logprob_for_realized_tokens(policy, x, s, e)
        if self.ref_record and self.ref_record.model_instance is not None:
            lps_ref = self._logprob_for_realized_tokens(self.ref_record.model_instance, x, s, e)
            deltas = [lp_p - lp_r for lp_p, lp_r in zip(lps_pol, lps_ref)]
            return float(np.mean(deltas)) if deltas else 0.0, len(lps_pol)
        return float(np.mean(lps_pol)) if lps_pol else 0.0, len(lps_pol)

    def _per_token_logp(self, model, x: mx.array) -> mx.array:
        # logits: (B, T, V)
        logits = model(x)
        B, T, V = logits.shape
        # Compute log softmax across vocab; avoid overflow
        # lsm = log_softmax(logits)
        max_logits = mx.max(logits, axis=-1, keepdims=True)
        lsm = logits - (max_logits + mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True)))
        # Targets are next tokens
        y = x[:, 1:]
        lsm_step = lsm[:, :-1, :]  # (B, T-1, V)
        # One-hot gather
        Bn, Ln, Vn = lsm_step.shape
        oh = mx.zeros((Bn, Ln, Vn))
        # Fill one-hot via loop (small smoke batches)
        for i in range(Bn):
            for j in range(Ln):
                tid = int(y[i, j].item())
                if 0 <= tid < Vn:
                    oh[i, j, tid] = 1.0
        logp_next = mx.sum(lsm_step * oh, axis=-1)  # (B, T-1)
        pad_col = mx.zeros((Bn, 1))
        logp = mx.concatenate([pad_col, logp_next], axis=1)  # align to (B, T)
        return logp


    def step(self, scored_batch: List[ScoredDataGroup]) -> Dict[str, Any]:
        # Collate tokens and masks
        if not scored_batch:
            return {"tokens": 0}
        # Each ScoredDataGroup contains multiple trajectories; flatten to a single batch
        tokens_list: List[List[int]] = []
        masks_list: List[List[int]] = []
        ref_lp_list: Optional[List[List[float]]] = []
        adv_list: Optional[List[List[float]]] = []
        scores: List[float] = []
        for sd in scored_batch:
            tokens_list.extend(sd['tokens'])
            masks_list.extend(sd['masks'])
            scores.extend(sd['scores'])
            if 'ref_logprobs' in sd and sd['ref_logprobs'] is not None:
                ref_lp_list.extend(sd['ref_logprobs'])
            else:
                ref_lp_list = None
            if 'advantages' in sd and sd['advantages'] is not None:
                adv_list.extend(sd['advantages'])
            else:
                adv_list = None

        B = len(tokens_list)
        T = max(len(t) for t in tokens_list)
        import numpy as np

        x_np = np.zeros((B, T), dtype=np.int64)
        m_np = np.zeros((B, T), dtype=np.float32)
        for i, (toks, msk) in enumerate(zip(tokens_list, masks_list)):
            x_np[i, : len(toks)] = toks
            m_np[i, : len(msk)] = msk
        x = mx.array(x_np)
        mask = mx.array(m_np)

        model = self.policy_record.model_instance
        logp = self._per_token_logp(model, x)
        if ref_lp_list is not None:
            # Collate ref logp to (B, T) aligned to masked positions; pad with zeros
            ref_np = np.zeros((B, T), dtype=np.float32)
            for i, seq in enumerate(ref_lp_list):
                # Align to last len(seq) masked positions; simple heuristic for smoke
                ref_np[i, -len(seq) :] = np.array(seq, dtype=np.float32)
            ref_logp = mx.array(ref_np)
        else:
            ref_logp = None

        # Advantages/ref_logprobs align with ALL tokens per Atropos spec; mask will select training positions
        if adv_list is not None:
            adv_np = np.zeros((B, T), dtype=np.float32)
            for i, seq in enumerate(adv_list):
                adv_np[i, : len(seq)] = np.array(seq, dtype=np.float32)
            advantages = mx.array(adv_np)
        else:
            # Broadcast sequence-level scores across sequence; mask applied later
            s_np = np.zeros((B, T), dtype=np.float32)
            for i, s in enumerate(scores):
                s_np[i, :] = float(s)
            advantages = mx.array(s_np)

        # Token-level ratio
        if ref_logp is not None:
            ratio = mx.exp(logp - ref_logp)
        else:
            ratio = mx.exp(logp - mx.stop_gradient(logp))

        # Loss: -adv * ratio masked
        loss_mat = -advantages * ratio * mask
        # Mean over valid tokens
        denom = mx.maximum(mx.sum(mask), mx.array(1.0))
        loss = mx.sum(loss_mat) / denom

        # Report simple metrics
        mean_logp = float(mx.sum(logp * mask).item()) / float(denom.item())
        metrics = {
            "loss": float(loss.item()),
            "mean_logp": mean_logp,
            "tokens": int(mx.sum(mask).item()),
        }

        # Backprop through MLX and update LoRA params only
        def loss_fn(m):
            lp = self._per_token_logp(m, x)
            if ref_logp is not None:
                rt = mx.exp(lp - ref_logp)
            else:
                rt = mx.exp(lp - mx.stop_gradient(lp))
            lm = -advantages * rt * mask
            dm = mx.maximum(mx.sum(mask), mx.array(1.0))
            return mx.sum(lm) / dm

        grads = mx.grad(loss_fn)(model)
        grads = zero_non_adapter_grads(model, grads)
        # Update under lock to avoid racing with inference reads
        from mlx_parallm.server.state import weight_update_lock
        with weight_update_lock:
            self.optimizer.update(model, grads)
        return metrics
