from __future__ import annotations

import concurrent.futures
import tempfile
import unittest
from pathlib import Path

import requests

from tests.helpers import build_tiny_model, get_metrics, start_server, stop_server, ServerHandle


class ServerBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="mlx_parallm_tests_batch_")
        cls._tmp = Path(cls._tmpdir.name)
        cls.model_dir = build_tiny_model(cls._tmp / "tiny_model_batch")
        # Use a slightly larger batch window so concurrent requests have time to coalesce.
        cls.server: ServerHandle = start_server(model_path=cls.model_dir, batch_timeout=0.2, max_batch_size=8)

    @classmethod
    def tearDownClass(cls) -> None:
        stop_server(cls.server)
        cls._tmpdir.cleanup()

    def test_concurrent_completions_increase_batch_metrics(self) -> None:
        before = get_metrics(self.server.base_url)
        before_batches = int(before.get("batches_processed", 0))

        payloads = [
            {
                "model": str(self.model_dir),
                "prompt": f"Request {i}: Say hello.",
                "max_tokens": 8,
                "temperature": 0.0,
            }
            for i in range(8)
        ]

        def do_req(p):
            r = requests.post(f"{self.server.base_url}/v1/completions", json=p, timeout=120)
            r.raise_for_status()
            return r.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(do_req, p) for p in payloads]
            for f in futs:
                j = f.result()
                self.assertIn("choices", j)

        after = get_metrics(self.server.base_url)
        after_batches = int(after.get("batches_processed", 0))
        self.assertGreaterEqual(after_batches, before_batches + 1, msg=f"metrics before={before} after={after}")
