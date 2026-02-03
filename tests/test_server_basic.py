from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import requests

from tests.helpers import build_tiny_model, start_server, stop_server, ServerHandle


class ServerBasicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="mlx_parallm_tests_basic_")
        cls._tmp = Path(cls._tmpdir.name)
        cls.model_dir = build_tiny_model(cls._tmp / "tiny_model_basic")
        cls.server: ServerHandle = start_server(model_path=cls.model_dir, batch_timeout=0.05, max_batch_size=8)

    @classmethod
    def tearDownClass(cls) -> None:
        stop_server(cls.server)
        cls._tmpdir.cleanup()

    def test_health(self) -> None:
        r = requests.get(f"{self.server.base_url}/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "ok")

    def test_models_list_loaded(self) -> None:
        r = requests.get(f"{self.server.base_url}/v1/models", timeout=10)
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j.get("object"), "list")
        data = j.get("data", [])
        self.assertTrue(any(m.get("id") == str(self.model_dir) and m.get("status") == "loaded" for m in data))

    def test_single_completion(self) -> None:
        payload = {
            "model": str(self.model_dir),
            "prompt": "Say hello in one word.",
            "max_tokens": 8,
            "temperature": 0.0,
        }
        r = requests.post(f"{self.server.base_url}/v1/completions", json=payload, timeout=60)
        self.assertEqual(r.status_code, 200, msg=r.text[:500])
        j = r.json()
        self.assertIn("choices", j)
        self.assertEqual(len(j["choices"]), 1)
        self.assertIsInstance(j["choices"][0].get("text"), str)

    def test_chat_completion_n2(self) -> None:
        payload = {
            "model": str(self.model_dir),
            "messages": [{"role": "user", "content": "Return exactly one word."}],
            "max_tokens": 8,
            "temperature": 0.7,
            "top_p": 0.95,
            "n": 2,
        }
        r = requests.post(f"{self.server.base_url}/v1/chat/completions", json=payload, timeout=120)
        self.assertEqual(r.status_code, 200, msg=r.text[:500])
        j = r.json()
        self.assertIn("choices", j)
        self.assertEqual(len(j["choices"]), 2)
        for c in j["choices"]:
            msg = c.get("message", {})
            self.assertEqual(msg.get("role"), "assistant")
            self.assertIsInstance(msg.get("content"), str)

    def test_streaming_chat_completes(self) -> None:
        payload = {
            "model": str(self.model_dir),
            "messages": [{"role": "user", "content": "In one sentence, describe a tree."}],
            "max_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": True,
        }
        with requests.post(
            f"{self.server.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as r:
            self.assertEqual(r.status_code, 200)
            saw_done = False
            # Read a bounded number of lines to avoid hanging if streaming breaks
            for _, line in zip(range(200), r.iter_lines(decode_unicode=True)):
                if not line:
                    continue
                if line.strip() == "data: [DONE]":
                    saw_done = True
                    break
                if line.startswith("data: "):
                    # Basic sanity: should be JSON chunks
                    payload_str = line[len("data: ") :].strip()
                    try:
                        json.loads(payload_str)
                    except Exception:
                        # tolerate non-json (e.g. error payloads) but fail loudly
                        self.fail(f"Non-JSON stream payload: {payload_str[:200]}")
            self.assertTrue(saw_done, msg="Did not observe data: [DONE] in stream")
