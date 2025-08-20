from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    batch_size: int = 8
    max_concurrent_requests: int = 100


@dataclass
class ModelConfig:
    base_path: Optional[str] = None
    lora_path: Optional[str] = None


@dataclass
class RLTrainingConfig:
    algorithm: str = "grpo"
    learning_rate: float = 1e-5
    batch_size: int = 32
    update_epochs: int = 1
    kl_beta: float = 0.05
    entropy_weight: float = 0.0
    clip_ratio: float = 0.2
    total_steps: int = 1000
    checkpoint_interval: int = 100

    atropos_url: Optional[str] = None


@dataclass
class Config:
    server: ServerConfig
    model: ModelConfig
    rl_training: RLTrainingConfig


def load_config(path: str) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    server = data.get("server", {})
    model = data.get("model", {})
    rl = data.get("rl_training", {})

    return Config(
        server=ServerConfig(
            host=server.get("host", "127.0.0.1"),
            port=int(server.get("port", 8000)),
            batch_size=int(server.get("batch_size", 8)),
            max_concurrent_requests=int(server.get("max_concurrent_requests", 100)),
        ),
        model=ModelConfig(
            base_path=model.get("base_path"),
            lora_path=model.get("lora_path"),
        ),
        rl_training=RLTrainingConfig(
            algorithm=rl.get("algorithm", "grpo"),
            learning_rate=float(rl.get("learning_rate", 1e-5)),
            batch_size=int(rl.get("batch_size", 32)),
            update_epochs=int(rl.get("update_epochs", 1)),
            kl_beta=float(rl.get("kl_beta", 0.05)),
            entropy_weight=float(rl.get("entropy_weight", 0.0)),
            clip_ratio=float(rl.get("clip_ratio", 0.2)),
            total_steps=int(rl.get("total_steps", 1000)),
            checkpoint_interval=int(rl.get("checkpoint_interval", 100)),
            atropos_url=rl.get("atropos", {}).get("api_url") if isinstance(rl.get("atropos"), dict) else rl.get("atropos_url"),
        ),
    )

