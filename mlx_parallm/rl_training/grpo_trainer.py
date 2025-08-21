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
from .param_utils import zero_non_adapter_grads, adapter_param_names
import mlx.optimizers as optim


@dataclass
class GRPOConfig:
    kl_beta: float = 0.05
    entropy_weight: float = 0.0  # not used in smoke
    max_tokens: int = 256        # limit per response for safety
    learning_rate: float = 1e-5
    kl_estimator: str = "k3"     # one of {"k3", "mse", "abs"}
    ref_ema: float = 1.0         # 0<val<1 enables EMA update of ref
    clip_ratio: float = 0.0      # 0 disables clipping; else PPO-style per-sign clipping


class GRPOTrainer(RLTrainerBase):
    def __init__(
        self,
        policy_record: InternalModelRecord,
        ref_record: Optional[InternalModelRecord],
        tokenizer: TokenizerWrapper,
        cfg: Optional[GRPOConfig] = None,
        *,
        checkpoint_dir: Optional[str] = None,
        save_every_step: bool = False,
        adapter_format: str = "npz",
        adapter_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(rollout_provider=None)  # Provider supplied externally in train loop
        if policy_record.model_instance is None:
            raise RuntimeError("Policy model not loaded.")
        self.policy_record = policy_record
        self.ref_record = ref_record
        self.tokenizer = tokenizer
        self.cfg = cfg or GRPOConfig()
        # Optimizer over trainable parameters only (LoRA adapters)
        self.optimizer = optim.Adam(self.cfg.learning_rate)
        # Initialize optimizer with trainable parameters only
        trainable_params = self.policy_record.model_instance.trainable_parameters()
        self.optimizer_state = self.optimizer.init(trainable_params)
        self.checkpoint_dir = checkpoint_dir
        self.save_every_step = save_every_step
        self._step_idx = 0
        self.adapter_format = adapter_format
        self.adapter_meta = adapter_meta or {}

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
        # Reference logp for KL, if ref model provided
        ref_logp = None
        if self.ref_record and self.ref_record.model_instance is not None and self.cfg.kl_beta > 0:
            try:
                ref_logp = self._per_token_logp(self.ref_record.model_instance, x)
            except Exception as e:
                logging.warning(f"Failed computing ref_logp: {e}")
                ref_logp = None
        if ref_lp_list is not None:
            # Collate ref logp to (B, T) aligned to masked positions; pad with zeros
            ref_np = np.zeros((B, T), dtype=np.float32)
            for i, seq in enumerate(ref_lp_list):
                # Align to last len(seq) masked positions; simple heuristic for smoke
                ref_np[i, -len(seq) :] = np.array(seq, dtype=np.float32)
            ref_logp = mx.array(ref_np)
        # If no external ref_logp, keep what we computed from ref model (or None)

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

        # KL penalty estimate if available
        kl_penalty = None
        if ref_logp is not None and self.cfg.kl_beta > 0:
            if self.cfg.kl_estimator == "k3":
                kl_penalty = mx.maximum(mx.exp(ref_logp - logp) - (ref_logp - logp) - 1, mx.array(0.0))
            elif self.cfg.kl_estimator == "mse":
                diff = (logp - ref_logp)
                kl_penalty = 0.5 * diff * diff
            elif self.cfg.kl_estimator == "abs":
                kl_penalty = mx.abs(logp - ref_logp)
            else:
                kl_penalty = None

        # Optional PPO-style ratio clipping by reward sign
        used_ratio = ratio
        is_clipped = mx.zeros_like(ratio)
        is_clipped_pos = mx.zeros_like(ratio)
        is_clipped_neg = mx.zeros_like(ratio)
        if float(self.cfg.clip_ratio) > 0.0:
            cr = float(self.cfg.clip_ratio)
            pos_mask_sign = (advantages > 0).astype(ratio.dtype)
            neg_mask_sign = (advantages <= 0).astype(ratio.dtype)
            used_ratio = mx.minimum(ratio, mx.array(1.0 + cr)) * pos_mask_sign + mx.maximum(ratio, mx.array(1.0 - cr)) * neg_mask_sign
            is_clipped_pos = (ratio > (1.0 + cr)) * pos_mask_sign
            is_clipped_neg = (ratio < (1.0 - cr)) * neg_mask_sign
            is_clipped = is_clipped_pos + is_clipped_neg

        # Loss: -adv * used_ratio masked (+ kl)
        loss_mat = -advantages * used_ratio
        if kl_penalty is not None:
            loss_mat = loss_mat + self.cfg.kl_beta * kl_penalty
        loss_mat = loss_mat * mask
        # Mean over valid tokens
        denom = mx.maximum(mx.sum(mask), mx.array(1.0))
        loss = mx.sum(loss_mat) / denom

        # Report simple metrics
        mean_logp = float(mx.sum(logp * mask).item()) / float(denom.item())
        # Clipping metrics
        if float(self.cfg.clip_ratio) > 0.0:
            clip_ratio_val = float((mx.sum(is_clipped * mask) / denom).item())
            pos_clip_ratio_val = float((mx.sum(is_clipped_pos * mask) / denom).item())
            neg_clip_ratio_val = float((mx.sum(is_clipped_neg * mask) / denom).item())
        else:
            clip_ratio_val = 0.0
            pos_clip_ratio_val = 0.0
            neg_clip_ratio_val = 0.0

        metrics = {
            "loss": float(loss.item()),
            "mean_logp": mean_logp,
            "tokens": int(mx.sum(mask).item()),
            "clip_ratio": clip_ratio_val,
            "pos_clip_ratio": pos_clip_ratio_val,
            "neg_clip_ratio": neg_clip_ratio_val,
        }
        if ref_logp is not None:
            try:
                # scalar summaries for inspection
                kl_mean = float((mx.sum(kl_penalty * mask) / denom).item()) if kl_penalty is not None else 0.0
                metrics.update({
                    "kl": kl_mean,
                })
            except Exception:
                pass

        # Backprop through MLX and update LoRA params only
        def loss_fn(m):
            # Forward and compute logp and entropy
            logits = m(x)  # (B, T, V)
            B, T, V = logits.shape
            max_logits = mx.max(logits, axis=-1, keepdims=True)
            lsm = logits - (max_logits + mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True)))
            y = x[:, 1:]
            lsm_step = lsm[:, :-1, :]  # (B, T-1, V)
            Bn, Ln, Vn = lsm_step.shape
            oh = mx.zeros((Bn, Ln, Vn))
            for i in range(Bn):
                for j in range(Ln):
                    tid = int(y[i, j].item())
                    if 0 <= tid < Vn:
                        oh[i, j, tid] = 1.0
            lp_next = mx.sum(lsm_step * oh, axis=-1)
            lp = mx.concatenate([mx.zeros((B,1)), lp_next], axis=1)

            # Ratio vs reference or on-policy baseline
            if ref_logp is not None:
                rt = mx.exp(lp - ref_logp)
            else:
                rt = mx.exp(lp - mx.stop_gradient(lp))

            # Optional clipping
            used_rt = rt
            if float(self.cfg.clip_ratio) > 0.0:
                cr = float(self.cfg.clip_ratio)
                pos_mask_sign = (advantages > 0).astype(rt.dtype)
                neg_mask_sign = (advantages <= 0).astype(rt.dtype)
                used_rt = mx.minimum(rt, mx.array(1.0 + cr)) * pos_mask_sign + mx.maximum(rt, mx.array(1.0 - cr)) * neg_mask_sign

            # Base loss
            lm = -advantages * used_rt * mask
            if ref_logp is not None and self.cfg.kl_beta > 0:
                if self.cfg.kl_estimator == "k3":
                    klp = mx.maximum(mx.exp(ref_logp - lp) - (ref_logp - lp) - 1, mx.array(0.0))
                elif self.cfg.kl_estimator == "mse":
                    d = (lp - ref_logp); klp = 0.5 * d * d
                elif self.cfg.kl_estimator == "abs":
                    klp = mx.abs(lp - ref_logp)
                else:
                    klp = None
                if klp is not None:
                    lm = lm + self.cfg.kl_beta * klp * mask

            # Entropy bonus
            if float(self.cfg.entropy_weight) > 0.0:
                p = mx.softmax(lsm_step, axis=-1)
                ent_step = -mx.sum(p * mx.log(mx.maximum(p, mx.array(1e-12))), axis=-1)  # (B, T-1)
                ent = mx.concatenate([mx.zeros((B,1)), ent_step], axis=1)
                lm = lm - self.cfg.entropy_weight * ent * mask
            dm = mx.maximum(mx.sum(mask), mx.array(1.0))
            return mx.sum(lm) / dm

        # Use value_and_grad to compute gradients only for trainable parameters
        # This avoids trying to compute gradients for quantized weights
        from mlx import nn
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss_val, grads = loss_and_grad_fn(model)
        
        # Zero out any non-adapter gradients (defensive programming)
        grads = zero_non_adapter_grads(model, grads)
        # Update under lock to avoid racing with inference reads
        from mlx_parallm.server.state import weight_update_lock
        with weight_update_lock:
            self.optimizer.update(model, grads)
        self._step_idx += 1

        # Optional EMA update of reference model's adapter params to keep it close to policy
        if self.ref_record and self.ref_record.model_instance is not None:
            ref_model = self.ref_record.model_instance
            ema = float(self.cfg.ref_ema)
            if 0.0 < ema < 1.0:
                try:
                    from mlx.utils import tree_flatten
                    pol_flat = dict(tree_flatten(model.parameters()))
                    ref_flat = dict(tree_flatten(ref_model.parameters()))
                    names = adapter_param_names(model)
                    updates = []
                    for k in names:
                        if k in ref_flat and k in pol_flat:
                            # new_ref = ema*ref + (1-ema)*pol
                            new_val = ema * ref_flat[k] + (1.0 - ema) * pol_flat[k]
                            updates.append((k, new_val))
                    if updates:
                        ref_model.load_weights(updates)
                except Exception as e:
                    logging.warning(f"EMA update of reference model failed: {e}")

        # Optional: save adapter snapshot and refresh server to latest adapter
        if self.checkpoint_dir and self.save_every_step:
            try:
                from .checkpoint import save_adapter_checkpoint
                step_dir = save_adapter_checkpoint(
                    self.checkpoint_dir,
                    model,
                    step=self._step_idx,
                    extra_meta=self.adapter_meta,
                    format=self.adapter_format,
                )
                if self.policy_record.adapter_path is not None:
                    from .weight_updater import apply_lora_update_for_record
                    apply_lora_update_for_record(self.policy_record, str(step_dir), lock=weight_update_lock)
            except Exception as e:
                logging.warning(f"Failed to save/apply adapter at step {self._step_idx}: {e}")
        return metrics
