import logging
import os
import threading
import time
from typing import Optional

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
from mlx_parallm.rl_training.checkpoint import save_checkpoint

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
    dry_run: bool = Field(False, description="Skip launching server/training; validate config only.", cli=["--dry-run"])

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
        if rec and self.lora_path:
            logging.info(f"Applying initial LoRA to active model: {self.lora_path}")
            apply_lora_update_for_record(rec, self.lora_path, lock=weight_update_lock)

        # Smoke training: use Atropos (mock or real). No optimizer yet; compute metrics and exercise in-process updates
        buffer = RolloutBuffer(maxlen=1024)
        provider = MockAtroposClient() if self.atropos_url is None else AtroposClient(self.atropos_url)

        # Build GRPO trainer with active policy model; for smoke, ref=policy
        policy_rec = get_active_record()
        if policy_rec is None:
            logging.warning("No active model found; training will log only.")
        else:
            # Tokenizer from policy record
            tok = policy_rec.tokenizer_instance
            if tok is None:
                logging.warning("No tokenizer on active model; training will log only.")
                policy_rec = None

        if policy_rec and policy_rec.tokenizer_instance is not None:
            trainer = GRPOTrainer(policy_record=policy_rec, ref_record=policy_rec, tokenizer=policy_rec.tokenizer_instance, cfg=GRPOConfig())
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
            if self.lora_path:
                rec = get_active_record()
                if rec:
                    apply_lora_update_for_record(rec, self.lora_path, lock=weight_update_lock)
            time.sleep(0.25)
            
            # Optional checkpointing
            if self.checkpoint_dir and (i + 1) % max(1, (10)) == 0:
                save_checkpoint(self.checkpoint_dir, step=i + 1, config={
                    "model_path": self.model_path,
                    "lora_path": self.lora_path,
                }, adapter_path=self.lora_path)

        logging.info("Smoke training completed.")


def train_cli_runner():
    run_and_exit(TrainCLIArgs, description="MLX ParaLLM unified trainer (scaffold)", version="0.1.0")
