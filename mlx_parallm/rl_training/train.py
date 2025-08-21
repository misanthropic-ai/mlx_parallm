import logging
import os
import threading
import time
from typing import Optional, Dict, Any

from pydantic import Field
from pydantic_cli import Cmd, run_and_exit

import uvicorn

from mlx_parallm.server.state import get_active_record, weight_update_lock, model_registry
from mlx_parallm.rl_training.weight_updater import apply_lora_update_for_record
from mlx_parallm.rl_training.atropos_client import MockAtroposClient, AtroposClient
from mlx_parallm.rl_training.trainer_base import RLTrainerBase
from mlx_parallm.rl_training.rollout_buffer import RolloutBuffer
from mlx_parallm.rl_training.grpo_trainer import GRPOTrainer, GRPOConfig
from mlx_parallm.rl_training.config import load_config
from mlx_parallm.rl_training.checkpoint import save_checkpoint, save_adapter_checkpoint

# Use the server CLI arg holder to pass startup config to the server
from mlx_parallm import cli as serve_cli_mod


class TrainCLIArgs(Cmd):
    model_path: Optional[str] = Field(None, description="Base model path or HF ID.", cli=["--model-path"])
    host: str = Field("127.0.0.1", description="Server host.", cli=["--host"])
    port: int = Field(8000, description="Server port.", cli=["--port"])
    lora_path: Optional[str] = Field(None, description="Optional initial LoRA adapter.", cli=["--lora-path"])
    steps: int = Field(5, description="Number of smoke training steps.", cli=["--steps"])
    batch_size: int = Field(8, description="Rollout batch size for smoke run.", cli=["--batch-size"])
    atropos_url: Optional[str] = Field(None, description="Atropos API URL (uses mock if unset)", cli=["--atropos-url"])
    token_budget: int = Field(65536, description="Total token budget for Atropos /get-batch.", cli=["--token-budget"])
    config: Optional[str] = Field(None, description="Path to training TOML config.", cli=["--config"])
    checkpoint_dir: Optional[str] = Field(None, description="Directory to write checkpoints.", cli=["--checkpoint-dir"])
    checkpoint_interval: int = Field(50, description="Save adapter checkpoint every N steps.", cli=["--checkpoint-interval"])
    dry_run: bool = Field(False, description="Skip launching server/training; validate config only.", cli=["--dry-run"])
    # LoRA auto-init controls
    auto_init_lora: bool = Field(True, description="Auto-initialize LoRA if not provided", cli=["--auto-init-lora"])
    lora_rank: int = Field(16, description="LoRA rank for auto-initialization", cli=["--lora-rank"])
    lora_layers: int = Field(8, description="Number of layers to apply LoRA to", cli=["--lora-layers"])
    lora_dropout: float = Field(0.05, description="LoRA dropout rate", cli=["--lora-dropout"])
    lora_scale: float = Field(10.0, description="LoRA scaling factor (alpha)", cli=["--lora-scale"])
    save_every_step: bool = Field(False, description="Save adapter checkpoint and refresh server each step", cli=["--save-every-step"])
    adapter_format: str = Field("npz", description="Adapter format to save (npz|safetensors)", cli=["--adapter-format"])
    # GRPO/Trainer hyperparams (CLI overrides config)
    algorithm: str = Field("grpo", description="Training algorithm (currently: grpo)", cli=["--algorithm"])
    learning_rate: float = Field(1e-5, description="Optimizer learning rate", cli=["--learning-rate"])
    kl_beta: float = Field(0.05, description="KL penalty weight", cli=["--kl-beta"])
    entropy_weight: float = Field(0.0, description="Entropy bonus weight (if enabled)", cli=["--entropy-weight"])
    clip_ratio: float = Field(0.2, description="Clip ratio (if clipping is used)", cli=["--clip-ratio"])
    max_tokens: int = Field(256, description="Max new tokens per response for smoke runs", cli=["--max-tokens"])
    update_epochs: int = Field(1, description="Update epochs per batch (if used)", cli=["--update-epochs"])
    kl_estimator: str = Field("k3", description="KL estimator: k3|mse|abs", cli=["--kl-estimator"])
    ref_ema: float = Field(1.0, description="EMA factor for reference model (0<ema<1 enables)", cli=["--ref-ema"])
    kl_estimator: str = Field("k3", description="KL estimator: k3|mse|abs", cli=["--kl-estimator"])
    ref_ema: float = Field(1.0, description="EMA factor for reference model (0<ema<1 enables)", cli=["--ref-ema"])

    def _launch_server_thread(self):
        def _runner():
            uvicorn.run(
                "mlx_parallm.server.main:app",
                host=self.host,
                port=self.port,
                log_level="info",
                reload=False,
            )

        t = threading.Thread(target=_runner, name="serve-thread", daemon=True)
        t.start()
        return t

    def run(self):
        logging.info("mlx_parallm_train starting")
        if self.dry_run:
            logging.info("Dry run: validated arguments.")
            return

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        # Load config if provided and apply defaults
        if self.config:
            cfg = load_config(self.config)
            self.model_path = self.model_path or cfg.model.base_path
            self.lora_path = self.lora_path or cfg.model.lora_path
            self.host = self.host or cfg.server.host
            self.port = self.port or cfg.server.port
            self.batch_size = self.batch_size or cfg.rl_training.batch_size
            self.steps = self.steps or cfg.rl_training.total_steps
            self.atropos_url = self.atropos_url or cfg.rl_training.atropos_url

        # Seed server startup config so main.startup_event can load the model
        serve_cli_mod.current_server_args = serve_cli_mod.ServerCLIArgs(
            model_path=self.model_path or "",
            host=self.host,
            port=self.port,
            lora_path=self.lora_path,
        )

        # Launch server in background
        serve_thread = self._launch_server_thread()
        logging.info("Server thread launched; waiting for readiness...")

        # Wait for registry to populate (bounded wait) only if we asked for a model
        if self.model_path:
            t0 = time.time()
            while time.time() - t0 < 30:
                rec = get_active_record()
                if rec is not None:
                    break
                time.sleep(0.5)

        # If an initial adapter is provided and a model is active, re-apply explicitly
        rec = get_active_record()
        if rec:
            # Auto-initialize LoRA on quantized models when no adapter provided
            if not self.lora_path and self.auto_init_lora and rec.model_instance is not None:
                # Prepare hyperparam metadata for the auto-init adapter_config.json
                auto_meta = {
                    "algorithm": self.algorithm,
                    "learning_rate": self.learning_rate,
                    "max_tokens": self.max_tokens,
                    "kl_beta": self.kl_beta,
                    "kl_estimator": self.kl_estimator,
                    "ref_ema": self.ref_ema,
                    "clip_ratio": self.clip_ratio,
                    "entropy_weight": self.entropy_weight,
                    "steps_total": self.steps,
                    "batch_size": self.batch_size,
                }
                try:
                    from .lora_init import init_lora_if_needed
                    auto_adapter_path = init_lora_if_needed(
                        rec.model_instance,
                        self.model_path or rec.path_or_hf_id,
                        self.checkpoint_dir or "./checkpoints",
                        rank=self.lora_rank,
                        num_layers=self.lora_layers,
                        dropout=self.lora_dropout,
                        scale=self.lora_scale,
                        adapter_format=self.adapter_format,
                        extra_meta=auto_meta,
                    )
                except Exception as e:
                    logging.warning(f"LoRA auto-init skipped due to error: {e}")
                    auto_adapter_path = None

                if auto_adapter_path:
                    logging.info(f"Auto-initialized LoRA adapters at: {auto_adapter_path}")
                    self.lora_path = auto_adapter_path
                    rec.adapter_path = auto_adapter_path

            if self.lora_path:
                logging.info(f"Applying adapter to active model: {self.lora_path}")
                apply_lora_update_for_record(rec, self.lora_path, lock=weight_update_lock)

        # Smoke training: use Atropos (mock or real). No optimizer yet; compute metrics and exercise in-process updates
        buffer = RolloutBuffer(maxlen=1024)
        
        # Get the policy record
        policy_rec = get_active_record()
        
        if self.atropos_url is None and policy_rec is not None:
            provider = MockAtroposClient(
                base_url=f"http://{self.host}:{self.port}", 
                model_id=self.model_path or "", 
                tokenizer=policy_rec.tokenizer_instance
            )
        elif self.atropos_url is not None:
            provider = AtroposClient(self.atropos_url)
        else:
            # Fallback mock client without tokenizer
            provider = MockAtroposClient(
                base_url=f"http://{self.host}:{self.port}", 
                model_id=self.model_path or "", 
                tokenizer=None
            )
        if policy_rec is None:
            logging.warning("No active model found; training will log only.")
        else:
            # Tokenizer from policy record
            tok = policy_rec.tokenizer_instance
            if tok is None:
                logging.warning("No tokenizer on active model; training will log only.")
                policy_rec = None

        if policy_rec and policy_rec.tokenizer_instance is not None:
            # Build a frozen reference model snapshot for KL if desired
            ref_record = None
            try:
                if self.model_path:
                    from mlx_parallm.utils import load as load_model_and_tokenizer
                    ref_model, _ = load_model_and_tokenizer(
                        self.model_path,
                        adapter_path=self.lora_path,
                        lazy=False,
                    )
                    # Keep ref frozen
                    try:
                        ref_model.eval()
                    except Exception:
                        pass
                    from mlx_parallm.server.schemas import InternalModelRecord, ModelStatus
                    ref_record = InternalModelRecord(
                        id=f"{policy_rec.id}__ref",
                        path_or_hf_id=policy_rec.path_or_hf_id,
                        model_type=policy_rec.model_type,
                        status=ModelStatus.LOADED,
                        adapter_path=self.lora_path,
                        model_instance=ref_model,
                        tokenizer_instance=policy_rec.tokenizer_instance,
                    )
            except Exception as e:
                logging.warning(f"Failed to build ref model; continuing without: {e}")

            # Build hyperparam metadata for checkpoints
            hp_meta: Dict[str, Any] = {
                "algorithm": self.algorithm,
                "learning_rate": self.learning_rate,
                "max_tokens": self.max_tokens,
                "kl_beta": self.kl_beta,
                "kl_estimator": self.kl_estimator,
                "ref_ema": self.ref_ema,
                "clip_ratio": self.clip_ratio,
                "entropy_weight": self.entropy_weight,
                "adapter_format": self.adapter_format,
                "steps_total": self.steps,
                "batch_size": self.batch_size,
            }

            trainer = GRPOTrainer(
                policy_record=policy_rec,
                ref_record=ref_record,
                tokenizer=policy_rec.tokenizer_instance,
                cfg=GRPOConfig(
                    kl_beta=self.kl_beta,
                    entropy_weight=self.entropy_weight,
                    max_tokens=self.max_tokens,
                    learning_rate=self.learning_rate,
                    kl_estimator=self.kl_estimator,
                    ref_ema=self.ref_ema,
                    clip_ratio=self.clip_ratio,
                ),
                checkpoint_dir=self.checkpoint_dir,
                save_every_step=self.save_every_step,
                adapter_format=self.adapter_format,
                adapter_meta=hp_meta,
            )
        else:
            class _NoOpTrainer(RLTrainerBase):
                def step(self, batch):
                    logging.info(f"[train] batch={len(batch)} no active model")
                    return {"loss": 0.0}
            trainer = _NoOpTrainer(provider)

        # Register with Atropos if using real provider
        trainer_uuid = None
        if isinstance(provider, AtroposClient):
            try:
                trainer_uuid = provider.register({
                    "wandb_group": "local",
                    "wandb_project": "mlx-parallm",
                    "batch_size": self.batch_size,
                    "max_token_len": 2048,
                    "checkpoint_dir": self.checkpoint_dir or "./checkpoints",
                    "save_checkpoint_interval": 500,
                    "starting_step": 0,
                    "num_steps": self.steps,
                })
                logging.info(f"Registered with Atropos, UUID: {trainer_uuid}")
            except Exception as e:
                logging.warning(f"Atropos registration failed: {e}")

        for i in range(self.steps):
            if isinstance(provider, AtroposClient):
                batch = provider.fetch_batch(self.batch_size, token_budget=self.token_budget, require_same_env=False)
            else:
                batch = list(provider.fetch(self.batch_size))
            
            # Process batch through GRPO trainer
            metrics = trainer.step(batch)
            logging.info(f"[train] step={i+1} metrics={metrics}")
            if self.lora_path and not self.save_every_step:
                rec = get_active_record()
                if rec:
                    apply_lora_update_for_record(rec, self.lora_path, lock=weight_update_lock)
            time.sleep(0.25)
            
            # Optional checkpointing
            if self.checkpoint_dir and (i + 1) % max(1, self.checkpoint_interval) == 0:
                # Save metadata
                save_checkpoint(
                    self.checkpoint_dir,
                    step=i + 1,
                    config={
                        "model_path": self.model_path,
                        "lora_path": self.lora_path,
                        "algorithm": self.algorithm,
                        "learning_rate": self.learning_rate,
                        "max_tokens": self.max_tokens,
                        "kl_beta": self.kl_beta,
                        "kl_estimator": self.kl_estimator,
                        "ref_ema": self.ref_ema,
                        "clip_ratio": self.clip_ratio,
                        "entropy_weight": self.entropy_weight,
                        "steps_total": self.steps,
                        "batch_size": self.batch_size,
                        "adapter_format": self.adapter_format,
                    },
                    adapter_path=self.lora_path,
                )
                # Save adapter-only weights snapshot
                if policy_rec and policy_rec.model_instance is not None:
                    save_adapter_checkpoint(
                        self.checkpoint_dir,
                        policy_rec.model_instance,
                        step=i + 1,
                        extra_meta={
                            "model": self.model_path,
                            "lora": self.lora_path,
                            "algorithm": self.algorithm,
                            "learning_rate": self.learning_rate,
                            "max_tokens": self.max_tokens,
                            "kl_beta": self.kl_beta,
                            "kl_estimator": self.kl_estimator,
                            "ref_ema": self.ref_ema,
                            "clip_ratio": self.clip_ratio,
                            "entropy_weight": self.entropy_weight,
                            "batch_size": self.batch_size,
                            "adapter_format": self.adapter_format,
                        },
                        format=self.adapter_format,
                    )

        logging.info("Smoke training completed.")


def train_cli_runner():
    run_and_exit(TrainCLIArgs, description="MLX ParaLLM unified trainer (scaffold)", version="0.1.0")
