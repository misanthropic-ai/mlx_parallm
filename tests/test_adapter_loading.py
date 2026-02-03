from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import requests

from mlx_parallm.rl_training.lora_init import init_lora_if_needed
from mlx_parallm.utils import load as load_model_and_tokenizer

from tests.helpers import build_tiny_model, start_server, stop_server, ServerHandle


class AdapterLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="mlx_parallm_tests_adapter_")
        cls._tmp = Path(cls._tmpdir.name)
        cls.model_dir = build_tiny_model(cls._tmp / "tiny_model_adapter")

        # Create an MLX-LM compatible adapter dir on disk.
        model, _tok = load_model_and_tokenizer(str(cls.model_dir), lazy=False)
        adapter_dir = init_lora_if_needed(
            model,
            model_path=str(cls.model_dir),
            checkpoint_dir=str(cls._tmp / "ckpt_adapter"),
            adapter_format="npz",
        )
        if adapter_dir is None:
            raise RuntimeError("Expected init_lora_if_needed to create adapter dir for quantized tiny model")
        cls.adapter_dir = Path(adapter_dir)

        # Start server with adapter preloaded.
        cls.server: ServerHandle = start_server(model_path=cls.model_dir, lora_path=cls.adapter_dir, batch_timeout=0.05, max_batch_size=8)

    @classmethod
    def tearDownClass(cls) -> None:
        stop_server(cls.server)
        cls._tmpdir.cleanup()

    def test_adapter_artifacts_present(self) -> None:
        self.assertTrue((self.adapter_dir / "adapter_config.json").exists())
        self.assertTrue((self.adapter_dir / "adapters.safetensors").exists())

    def test_server_runs_with_adapter(self) -> None:
        r = requests.get(f"{self.server.base_url}/v1/models", timeout=10)
        self.assertEqual(r.status_code, 200)
        data = r.json().get("data", [])
        self.assertTrue(any(m.get("id") == str(self.model_dir) and m.get("status") == "loaded" for m in data))

        payload = {
            "model": str(self.model_dir),
            "prompt": "Say hello.",
            "max_tokens": 8,
            "temperature": 0.0,
        }
        r2 = requests.post(f"{self.server.base_url}/v1/completions", json=payload, timeout=120)
        self.assertEqual(r2.status_code, 200, msg=r2.text[:500])
