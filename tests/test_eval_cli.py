"""Exercise the documented eval/ commands from the repository root using a test-only adapter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_capture_evaluate_and_compare_from_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, adapters, reports = (root / name for name in ("data", "adapters", "reports"))
            data.mkdir()
            adapters.mkdir()
            reports.mkdir()
            (data / "document.pdf").write_bytes((ROOT / "eval/fixtures/synthetic.pdf").read_bytes())
            (data / "manifest.json").write_text(json.dumps({
                "purpose": "cli-unit-test",
                "cases": [{"id": "document", "pdf": "document.pdf",
                           "reference": "references/document.json", "pages": 1}],
            }))
            # This deliberately avoids models and GPUs. These temporary reports
            # only exercise the CLI protocol and are never benchmark artifacts.
            (adapters / "cli_baseline.py").write_text(
                "def embed_pdf(path: str) -> dict:\n"
                "    with open(path, 'rb') as source:\n"
                "        if not source.read(1024).startswith(b'%PDF-'):\n"
                "            raise ValueError('Malformed PDF')\n"
                "    return {'ocr_text': 'Hello world\\n', 'chunks': "
                "[{'id': 'chunk-000', 'embedding': [1.0, 0.0, 0.0]}]}\n")
            (adapters / "cli_optimized.py").write_text("from cli_baseline import embed_pdf\n")
            environment = {
                "hardware_id": "unit-test-only", "gpu_model": "mock", "gpu_count": 1,
                "driver_version": "mock", "concurrency": 1,
            }
            child_environment = {**os.environ, "PYTHONPATH": str(adapters)}

            def run(script, *arguments, expected=0):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "eval" / script), *map(str, arguments)],
                    cwd=ROOT, env=child_environment, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return result

            inputs = ("--manifest", data / "manifest.json", "--root", data)
            run("capture_reference.py", "--module", "cli_baseline", *inputs)
            reference = data / "references/document.json"
            original_reference = reference.read_bytes()
            self.assertEqual(json.loads(original_reference)["ocr_text"], "Hello world\n")
            # A repeated capture must refuse to overwrite the frozen reference.
            run("capture_reference.py", "--module", "cli_baseline", *inputs, expected=1)
            self.assertEqual(reference.read_bytes(), original_reference)

            for name in ("baseline", "optimized"):
                env_path = reports / (name + "-env.json")
                env_path.write_text(json.dumps({**environment, "engine": "test-" + name}))
                run("evaluate.py", "--module", "cli_" + name, *inputs,
                    "--environment", env_path, "--out", reports / (name + ".json"))
                report = json.loads((reports / (name + ".json")).read_text())
                self.assertTrue(report["pass"])
                self.assertEqual(report["configuration"]["warmup_per_case"], 1)
                self.assertEqual(report["configuration"]["repeats_per_case"], 5)
                self.assertEqual(report["configuration"]["timeout_seconds"], 600)
                self.assertEqual(report["summary"]["measured_samples"], 5)
                self.assertEqual([r["phase"] for r in report["cases"][0]["runs"]],
                                 ["warmup"] + ["measured"] * 5)
                self.assertGreater(report["cold_start"]["process_to_first_result_ms"],
                                   report["cold_start"]["worker_startup_ms"])
                self.assertGreater(report["cases"][0]["p50_ms"], 0)
                self.assertGreaterEqual(report["cases"][0]["p95_ms"],
                                        report["cases"][0]["p50_ms"])

            comparison = reports / "comparison.json"
            run("compare.py", "--baseline", reports / "baseline.json",
                "--optimized", reports / "optimized.json", "--out", comparison)
            result = json.loads(comparison.read_text())
            self.assertTrue(result["pass"])
            for values in result["metrics"].values():
                self.assertGreater(values["speedup"], 0)


if __name__ == "__main__":
    unittest.main()
