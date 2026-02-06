from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlx_parallm.rl_training.checkpoint import save_adapter_checkpoint
from mlx_parallm.rl_training.lora_init import init_lora_if_needed
from mlx_parallm.rl_training.weight_updater import apply_lora_update_for_record
from mlx_parallm.server.schemas import InternalModelRecord, ModelStatus
from mlx_parallm.server.state import model_registry, get_active_record
from mlx_parallm.utils import load as load_model_and_tokenizer

from tests.helpers import build_tiny_model


class CheckpointAndWeightUpdateTest(unittest.TestCase):
    def test_save_checkpoint_and_apply_update(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlx_parallm_tests_ckpt_") as td:
            tmp = Path(td)
            model_dir = build_tiny_model(tmp / "tiny_model_ckpt")

            model, tok = load_model_and_tokenizer(str(model_dir), lazy=False)

            adapter_dir = init_lora_if_needed(
                model,
                model_path=str(model_dir),
                checkpoint_dir=str(tmp / "ckpt_root"),
                adapter_format="npz",
            )
            self.assertIsNotNone(adapter_dir)

            step_dir = save_adapter_checkpoint(
                tmp / "ckpt_root",
                model,
                step=1,
                extra_meta={"lora_path": adapter_dir},
                format="npz",
            )
            self.assertTrue((step_dir / "adapters.safetensors").exists())
            self.assertTrue((step_dir / "adapter_config.json").exists())

            # Create a record and apply the update (normal MLX-LM load_adapters path).
            rec = InternalModelRecord(
                id=str(model_dir),
                path_or_hf_id=str(model_dir),
                model_type="causal_lm",
                status=ModelStatus.LOADED,
                model_instance=model,
                tokenizer_instance=tok,
            )
            apply_lora_update_for_record(rec, str(step_dir))
            self.assertEqual(rec.adapter_path, str(step_dir))

            # Force fallback path: remove adapter_config.json so load_adapters fails,
            # and ensure weight_updater still applies weights from adapters.safetensors.
            (step_dir / "adapter_config.json").unlink()
            apply_lora_update_for_record(rec, str(step_dir))
            self.assertEqual(rec.adapter_path, str(step_dir))

    def test_get_active_record(self) -> None:
        model_registry.clear()
        rec = InternalModelRecord(
            id="m",
            path_or_hf_id="m",
            model_type="causal_lm",
            status=ModelStatus.LOADED,
            model_instance=object(),
            tokenizer_instance=object(),
        )
        model_registry["m"] = rec
        active = get_active_record()
        self.assertIsNotNone(active)
        self.assertEqual(active.id, "m")

