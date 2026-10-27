"""compare.py refuses runs that differ in hardware, dataset or settings (eval/ stays unmodified)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.compare import compare
from eval.evaluate import evaluate
from eval.tests.test_reports import FakeWorker

ROOT = Path(__file__).resolve().parents[1]


class CompareGuardTests(unittest.TestCase):
    def report(self):
        with tempfile.TemporaryDirectory() as temp, patch('eval.evaluate.Worker', FakeWorker):
            manifest = Path(temp) / 'manifest.json'
            source = json.loads((ROOT / 'eval/fixtures/manifest.json').read_text())
            source['purpose'] = 'unit-test'
            manifest.write_text(json.dumps(source))
            return evaluate('mocked.adapter', manifest, ROOT / 'eval/fixtures',
                            warmup=1, repeats=5, minimum=.99,
                            environment={'hardware_id': 'unit-test', 'gpu_model': 'mock',
                                         'gpu_count': 1, 'driver_version': 'mock',
                                         'concurrency': 1})

    def test_compare_requires_matching_hardware_dataset_and_settings(self):
        baseline = self.report()
        for field, replacement in [
            ('hardware_id', 'different'), ('gpu_model', 'different'), ('gpu_count', 2),
            ('driver_version', 'different'), ('concurrency', 2),
        ]:
            with self.subTest(environment=field):
                optimized = copy.deepcopy(baseline)
                optimized['environment'][field] = replacement
                with self.assertRaisesRegex(ValueError, 'Environments differ'):
                    compare(baseline, optimized)
                optimized['environment'][field] = ''
                with self.assertRaisesRegex(ValueError, 'is missing'):
                    compare(baseline, optimized)
        optimized = copy.deepcopy(baseline)
        optimized['dataset_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different PDFs or reference outputs'):
            compare(baseline, optimized)
        for field, value in [('warmup_per_case', 2), ('repeats_per_case', 6),
                             ('minimum_cosine', .995)]:
            with self.subTest(setting=field):
                optimized = copy.deepcopy(baseline)
                optimized['configuration'][field] = value
                with self.assertRaisesRegex(ValueError, 'Settings differ'):
                    compare(baseline, optimized)

    def test_compare_requires_sequential_calls_warmup_and_cosine_threshold(self):
        report = self.report()
        report['environment']['concurrency'] = 2
        with self.assertRaisesRegex(ValueError, 'concurrency 1'):
            compare(report, report)
        report = self.report()
        report['configuration']['warmup_per_case'] = 0
        with self.assertRaisesRegex(ValueError, 'one warm-up'):
            compare(report, report)
        report = self.report()
        report['configuration']['minimum_cosine'] = .98
        with self.assertRaisesRegex(ValueError, 'at least 0.99'):
            compare(report, report)


if __name__ == '__main__':
    unittest.main()
