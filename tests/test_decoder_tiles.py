"""Run with .venv-vllm/bin/python -m unittest discover -s tests -p test_decoder_tiles.py."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/paddle_vllm_plugin"))
from paddle_native_precision.decoder_tiles import (
    ENVIRONMENT_KEY, decoder_shapes, install_decoder_tiles,
)


@unittest.skipUnless(importlib.util.find_spec("vllm"), "Requires the vLLM environment")
class DecoderTileIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from vllm.model_executor.determinism import batch_invariant as invariant
        from vllm.model_executor.determinism import batch_invariant_configs as configs
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
            raise unittest.SkipTest("Tile screen targets the existing Ada GPUs")
        cls.torch, cls.invariant, cls.configs = torch, invariant, configs
        hf = SimpleNamespace(**json.loads(
            (Path.home() / "huggingface/PaddleOCR-VL-1.6/config.json").read_text()))
        cls.cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf, dtype=torch.bfloat16),
            parallel_config=SimpleNamespace(tensor_parallel_size=1))
        cls.shapes = decoder_shapes(hf)
        cls.table = configs._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"]

    def setUp(self):
        self.before = dict(self.table)
        self.environment = patch.dict(os.environ, {
            ENVIRONMENT_KEY: "1", "VLLM_BATCH_INVARIANT": "1"})
        self.environment.start()

    def tearDown(self):
        self.table.clear()
        self.table.update(self.before)
        self.environment.stop()

    def test_disabled_path_does_not_touch_configuration(self):
        with patch.dict(os.environ, {ENVIRONMENT_KEY: "0"}):
            self.assertEqual(install_decoder_tiles(object()), ())
        self.assertEqual(self.table, self.before)

    def test_requires_invariance_and_supported_parallelism(self):
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0"}):
            with self.assertRaisesRegex(ValueError, "batch invariance"):
                install_decoder_tiles(self.cfg)
        cfg = SimpleNamespace(model_config=self.cfg.model_config,
                              parallel_config=SimpleNamespace(tensor_parallel_size=2))
        with self.assertRaisesRegex(ValueError, "parallel size one"):
            install_decoder_tiles(cfg)
        self.assertEqual(self.table, self.before)

    def test_existing_tuning_conflict_does_not_partially_modify_table(self):
        self.table[self.shapes[-1]] = object()
        before = dict(self.table)
        with self.assertRaisesRegex(ValueError, "already tunes"):
            install_decoder_tiles(self.cfg)
        self.assertEqual(self.table, before)

    def test_actual_kernel_preserves_outputs_across_tile_boundaries(self):
        torch = self.torch
        self.invariant.init_batch_invariance()
        base = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8}
        for n, k in self.shapes:
            self.assertEqual(self.configs._get_matmul_config(
                5, n, k, torch.bfloat16, base), base)
        torch.manual_seed(123)
        a = torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16).t()
        expected = self.invariant.matmul_persistent(a, b)
        self.assertEqual(install_decoder_tiles(self.cfg), self.shapes)
        self.assertEqual(install_decoder_tiles(self.cfg), self.shapes)  # Idempotent.
        for m in (1, 5, 16, 17, 40, 64, 65, 128):
            for n, k in self.shapes:
                cfg = self.configs._get_matmul_config(m, n, k, torch.bfloat16, base)
                self.assertEqual(cfg["BLOCK_SIZE_K"], 64)
                self.assertEqual(cfg["BLOCK_SIZE_N"], 128)
                self.assertEqual(cfg["BLOCK_SIZE_M"], 16 if m <= 16 else 32 if m <= 64 else 128)
                self.assertEqual(self.configs._get_matmul_config(
                    m, n, k, torch.float32, base), base)
            actual = self.invariant.matmul_persistent(a[:m], b)
            self.assertTrue(torch.equal(actual, expected[:m]), f"Batch boundary {m}")

    def test_fixed_tiles_survive_dynamic_compilation(self):
        self._assert_compiled_tiles("32", 1024, 2048, 128)

    def test_shape_columns_survive_dynamic_compilation(self):
        for n, k, columns in ((1024, 2048, 64), (2560, 1024, 256)):
            with self.subTest(shape=(n, k), columns=columns):
                self.table.clear()
                self.table.update(self.before)
                self._assert_compiled_tiles("shape_columns", n, k, columns)

    def _assert_compiled_tiles(self, mode, n, k, columns):
        torch = self.torch
        torch._dynamo.reset()
        self.invariant.init_batch_invariance()
        torch.manual_seed(321)
        a = torch.randn(2048, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
        expected = self.invariant.matmul_persistent(a, b)
        with patch.dict(os.environ, {ENVIRONMENT_KEY: mode}):
            install_decoder_tiles(self.cfg)
        base = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8}
        for m in (1, 5, 64, 128, 2048, 16384):
            for shape_n, shape_k in self.shapes:
                selected = self.configs._get_matmul_config(m, shape_n, shape_k, torch.bfloat16, base)
                self.assertEqual(selected["BLOCK_SIZE_M"], 32)
                expected_columns = (256 if shape_n in {2560, 6144} else 64) if mode == "shape_columns" else 128
                self.assertEqual(selected["BLOCK_SIZE_N"], expected_columns)
                self.assertEqual(selected["BLOCK_SIZE_K"], 64)
                self.assertEqual(selected["num_warps"], 4)
        compiled = torch.compile(self.invariant.matmul_persistent, dynamic=True, fullgraph=True)
        # Trace with a prefill-sized input before running short decode inputs.
        for m in (2048, 5, 65, 128):
            self.assertTrue(torch.equal(compiled(a[:m], b), expected[:m]), f"Compiled M={m}")
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profile:
            compiled(a[:5], b)
            compiled(a[:65], b)
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory(prefix="decoder-tiles-") as directory:
            path = Path(directory) / "trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        kernels = [event for event in events if event.get("cat") == "kernel"
                   and "matmul_kernel_persistent" in event.get("name", "")]
        self.assertEqual(len(kernels), 2)
        self.assertTrue(all(event["args"]["block"] == [128, 1, 1] for event in kernels))
        expected_grids = [
            [min(58, ((m + 31) // 32) * ((n + columns - 1) // columns)), 1, 1]
            for m in (5, 65)]
        self.assertEqual([event["args"]["grid"] for event in kernels], expected_grids)


if __name__ == "__main__":
    unittest.main()
